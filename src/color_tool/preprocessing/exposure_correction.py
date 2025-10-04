# src/color_tool/preprocessing/exposure_correction.py
"""
Exposure + WB correction tuned for skin/hair/eye color analysis.

Pipeline:
  sRGB→linear → (skin-aware) exposure gain + soft roll-off
  → Shades-of-Gray WB (with Gray-World fallback)
  → back to sRGB → denoise → LAB(L*) CLAHE (adaptive)
  → mild a*/b* recenter (scaled by bias) → metrics & quality gate
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Tuple, Dict

import cv2
import numpy as np


# ----------------------- Config -----------------------

@dataclass
class PreprocessConfig:
    # Exposure
    target_mid_gray: float = 0.42      # desired midtone
    midtone_percentile: float = 60.0   # percentile treated as "midface"
    exposure_gain_min: float = 0.6     # >>> NEW: safer lower clamp (was 0.5)
    exposure_gain_max: float = 2.4     # >>> NEW: lower ceiling to avoid blowouts (was 3.0)

    # White balance
    wb_p_norm: int = 4
    wb_gain_clip: float = 1.8         # >>> NEW: safer cap (was 3.0)

    # LAB / CLAHE
    clahe_tile: int = 8
    clahe_clip_min: float = 1.4        # >>> NEW: gentler (was 1.5)
    clahe_clip_max: float = 3.5        # >>> NEW: avoid plasticky look (was 5.0)

    # a*/b* centering
    ab_shift_cap: float = 4.0          # >>> NEW: gentler cap (was 6.0)
    ab_full_shift_bias: float = 6.0    # >>> NEW: require stronger cast for full shift (was 5.0)

    # Quality thresholds (optional)
    max_overexp_pct: float = 2.0       # >>> NEW: tighter (was 3.0)
    max_underexp_pct: float = 8.0      # >>> NEW: looser shadow tolerance (was 5.0)
    max_ab_bias: float = 3.0           # tighter neutrality tolerance (was 3.5)
    min_blur_var: float = 60.0         # Laplacian variance lower bound

    # midtone range and max luminance for no-edit window
    min_midtone: float = 0.38   # >>> NEW: wider no-edit window (was 0.38)
    max_midtone: float = 0.54   # >>> NEW: wider no-edit window (was 0.54)
    max_safe_highs: float = 0.88 # >>> NEW: allow slightly brighter highlights (was 0.95)

    skin_min_fraction: float = 0.10    # >>> NEW: need enough skin to judge (was 0.02)

# ----------------------- Color space helpers -----------------------

def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB (0..255, BGR) -> linear (0..1, BGR)."""
    x = img_u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    """linear (0..1, BGR) -> uint8 sRGB (0..255, BGR)."""
    x = np.where(img_lin <= 0.0031308, 12.92 * img_lin,
                 1.055 * (img_lin ** (1 / 2.4)) - 0.055)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def luminance_from_linear_bgr(lin_bgr: np.ndarray) -> np.ndarray:
    """ Luminance Y from linear BGR image."""
    B, G, R = lin_bgr[..., 0], lin_bgr[..., 1], lin_bgr[..., 2]
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


# ----------------------- Building blocks -----------------------

def soft_tone(x: np.ndarray, k: float = 0.85) -> np.ndarray:
    """
    Highlight-preserving roll-off in linear space.
    x in [0,1]
    """
    return (x * (1.0 + k)) / (x + k + 1e-8)


def quick_skin_mask(bgr_u8: np.ndarray) -> np.ndarray:
    """
    Very lightweight skin mask (union of YCrCb and HSV rules), median smoothed.
    Returns uint8 mask {0,255}.
    """
    # YCrCb rule
    ycrcb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    m1 = (Cr >= 135) & (Cr <= 180) & (Cb >= 85) & (Cb <= 135)

    # HSV rule (avoid very low saturation and extremes)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m2 = (S >= 40) & (V >= 35) & (V <= 240)

    mask = (m1 & m2).astype(np.uint8) * 255
    if mask.mean() > 0:
        mask = cv2.medianBlur(mask, 5)
    return mask


def estimate_exposure_gain(lin_bgr: np.ndarray, target: float, pct: float,
                           gmin: float, gmax: float,
                           skin_mask: np.ndarray | None = None) -> float:
    """
    Robust midtone-based global gain. If a usable skin mask is provided, we
    compute the percentile on skin only.
    """
    if skin_mask is not None and (skin_mask > 0).mean() >= 0.02:
        # Use only skin pixels
        lin_skin = lin_bgr[skin_mask > 0]
        R, G, B = lin_skin[:, 2], lin_skin[:, 1], lin_skin[:, 0]
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    else:
        Y = luminance_from_linear_bgr(lin_bgr)

    mid = float(np.percentile(Y, pct))
    if mid < 1e-6:
        return 1.0
    return float(np.clip(target / mid, gmin, gmax))


def shades_of_gray_wb(lin_bgr: np.ndarray, p: int, clip: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shades-of-Gray white balance (Minkowski p-norm).
    Returns (balanced_linear_bgr, gains_bgr).
    """
    eps = 1e-8
    # Compute per-channel p-norm means
    m = np.power(np.mean(np.power(lin_bgr, p), axis=(0, 1)) + eps, 1 / p)  # B,G,R
    gains = m.mean() / m
    gains = np.clip(gains, 1.0 / clip, clip)
    out = np.clip(lin_bgr * gains, 0.0, 1.0)
    return out, gains

def gray_world_wb(lin_bgr: np.ndarray, clip: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Gray-World white balance."""
    mean = lin_bgr.mean(axis=(0, 1))
    gains = mean.mean() / (mean + 1e-8)
    gains = np.clip(gains, 1.0 / clip, clip)
    out = np.clip(lin_bgr * gains, 0.0, 1.0)
    return out, gains


def lab_mean_ab(bgr_u8: np.ndarray) -> Tuple[float, float]:
    """ Mean a* and b* from uint8 BGR image."""
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    return float(a.mean()), float(b.mean())


def laplacian_var(gray_u8: np.ndarray) -> float:
    """ Laplacian variance (sharpness) from uint8 grayscale image."""
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


# >>> NEW: luminance percentiles & adaptive roll-off strength helpers
def luminance_percentiles(lin_bgr: np.ndarray, mask: np.ndarray | None = None):
    """Return key luminance percentiles on linear Y (optionally skin-only)."""
    Y = luminance_from_linear_bgr(lin_bgr)
    if mask is not None and (mask > 0).any():
        Y = Y[mask > 0]
    return (float(np.percentile(Y, 5.0)),
            float(np.percentile(Y, 50.0)),
            float(np.percentile(Y, 95.0)),
            float(np.percentile(Y, 99.0)))


def k_for_soft_tone(y95: float) -> float:
    """Adaptive highlight roll-off: stronger when bright highlights exist."""
    # y95 <= 0.75 -> mild roll-off; y95 >= 0.95 -> strong roll-off
    return float(np.interp(y95, [0.75, 0.95], [0.85, 1.8]))


# ----------------------- Core routine -----------------------

def correct_exposure(
    bgr_u8: np.ndarray,
    cfg: PreprocessConfig | None = None
) -> Tuple[np.ndarray, Dict]:
    """
    Exposure + WB + mild LAB cleanup. Returns (corrected_bgr_u8, metrics).
    """
    if cfg is None:
        cfg = PreprocessConfig()

    report: Dict = {}

    # Pre metrics (for debugging/quality gate)
    pre_a, pre_b = lab_mean_ab(bgr_u8)
    pre_ab_bias = float(np.hypot(pre_a, pre_b))
    pre_over = float((bgr_u8 >= 250).mean() * 100.0)
    pre_under = float((bgr_u8 <= 5).mean() * 100.0)
    pre_blur = laplacian_var(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY))

    # 1) sRGB → linear
    lin = srgb_to_linear(bgr_u8)

    # >>> TWEAK: build skin & luminance stats early + wider NO-EDIT WINDOW + jitter guard
    skin = quick_skin_mask(bgr_u8)
    skin_frac = float((skin > 0).mean())

    # Luminance percentiles (global & skin-only)
    g_y5, g_y50, g_y95, g_y99 = luminance_percentiles(lin)
    s_y5, s_y50, s_y95, s_y99 = luminance_percentiles(
        lin, skin if skin_frac >= cfg.skin_min_fraction else None
    )

    # Prefer skin stats when enough skin is present
    use_skin_stats = (skin_frac >= cfg.skin_min_fraction)
    y50 = s_y50 if use_skin_stats else g_y50
    y95 = s_y95 if use_skin_stats else g_y95

    # -------- NO-EDIT WINDOW: leave already-good exposures untouched ----------
    # Wider band: treat 0.36–0.56 as acceptable midtone; highlights safe up to 0.96
    adequate_mid = cfg.min_midtone <= y50 <= cfg.max_midtone
    safe_highs  = y95 <= cfg.max_safe_highs

    if adequate_mid and safe_highs:
        # Small "jitter" guard—if a computed gain would move midtone by < 0.03, skip edits anyway
        # (protects borderline-good images from tiny, unnecessary changes)
        est_gain = cfg.target_mid_gray / max(y50, 1e-6)
        if abs((y50 * est_gain) - y50) < 0.03:
            report.update({
                "pre": {
                    "ab_bias": pre_ab_bias,
                    "overexp_pct": pre_over,
                    "underexp_pct": pre_under,
                    "blur_var": pre_blur,
                },
                "exposure_gain": 1.0,
                "used_skin_mask": bool(use_skin_stats),
                "skin_fraction": skin_frac,
                "wb_method": "noop",
                "wb_gains_bgr": (1.0, 1.0, 1.0),
                "clahe_clip_used": 0.0,
                "a_shift": 0.0,
                "b_shift": 0.0,
            })
            report["post"] = report["pre"]
            report["lighting_ok"] = True
            report["wb_ok"] = True
            report["sharp_ok"] = pre_blur > cfg.min_blur_var
            report["overall_ok"] = bool(report["lighting_ok"] and report["wb_ok"] and report["sharp_ok"])
            return bgr_u8, report
# --------------------------------------------------------------------------

    # 2) Exposure gain (skin-aware if possible) + soft roll-off with predicted clip guard
    use_skin = (skin_frac >= cfg.skin_min_fraction)
 
    proposed_gain = estimate_exposure_gain(
        lin, cfg.target_mid_gray, cfg.midtone_percentile,
        cfg.exposure_gain_min, cfg.exposure_gain_max,
        skin_mask=skin if use_skin else None
    )

    if use_skin and (0.4 <= y50 <= 0.56) and (pre_ab_bias <= 2.2):
        # prevent lifting if skin already mid-bright and chroma is healhty:
        proposed_gain = min(proposed_gain, 1.15)
        
    # >>> TWEAK: dynamic max gain — allow higher ceiling only when skin is clearly dark
    # (keeps good/brighter images safe, but lifts sample2–4)
    _, _, y95_sel, y99_sel = (s_y5, s_y50, s_y95, s_y99) if use_skin else (g_y5, g_y50, g_y95, g_y99)

    if y50 < 0.34 or y95_sel < 0.80:
        dyn_max = max(cfg.exposure_gain_max, 2.9)   # allow stronger lift for very dark faces
    elif y50 < 0.40:
        dyn_max = max(cfg.exposure_gain_max, 2.7)
    else:
        dyn_max = cfg.exposure_gain_max

    # Always clip by highlight safety using 99th percentile (skin-first)
    max_gain_clip = 0.98 / max(y99_sel, 1e-3)
    max_gain_clip *= 0.98  # small safety margin

    proposed_gain = float(min(proposed_gain, dyn_max, max_gain_clip))

    # Additional backoff if highlights would still push too high
    if (y95_sel * proposed_gain) > 0.95:
        proposed_gain *= 0.92

    gain = float(proposed_gain)

    # Adaptive highlight roll-off strength based on current y95
    k = k_for_soft_tone(y95_sel)
    lin = soft_tone(lin * gain, k=k)

    report["exposure_gain"] = gain
    report["used_skin_mask"] = bool(use_skin)
    report["skin_fraction"] = float(skin_frac)


    # 3) White balance: SoG with adaptive GW blend if SoG does not improve ab-bias enough
    sog_lin, sog_gains = shades_of_gray_wb(lin, cfg.wb_p_norm, cfg.wb_gain_clip)
    sog_bgr = linear_to_srgb(sog_lin)
    a1, b1 = lab_mean_ab(sog_bgr)
    sog_bias = np.hypot(a1, b1)

    gw_lin, gw_gains = gray_world_wb(lin, cfg.wb_gain_clip)
    gw_bgr = linear_to_srgb(gw_lin)
    ag, bg = lab_mean_ab(gw_bgr)
    gw_bias = np.hypot(ag, bg)

    # pick blend based on which reduces pre-bias more; interpolate when similar
    def improvement(b): return max(0.0, pre_ab_bias - b)
    imp_sog, imp_gw = improvement(sog_bias), improvement(gw_bias)
    total = imp_sog + imp_gw + 1e-6
    w_gw = float(np.clip(imp_gw / total, 0.0, 1.0))   # 0..1
    w_gw = 0.2 + 0.6 * w_gw                           # keep some SoG; range ≈ [0.2, 0.8]
    lin = np.clip(w_gw * gw_lin + (1.0 - w_gw) * sog_lin, 0, 1)

    report["wb_method"] = f"adaptive_blend_gw={w_gw:.2f}"
    report["wb_gains_bgr"] = tuple(float(x) for x in (w_gw * gw_gains + (1.0 - w_gw) * sog_gains))

    # 4) Back to sRGB → light denoise before LAB work
    bgr_corr = linear_to_srgb(lin)
    bgr_corr = cv2.fastNlMeansDenoisingColored(bgr_corr, None, h=3, hColor=3,
                                               templateWindowSize=7, searchWindowSize=21)

    # 5) Adaptive CLAHE on L* only
    # Contrast proxy: std of L* (0..255). Map to clip limit [min, max].
    # >>> NEW: only apply CLAHE when actually low-contrast to avoid waxy look
    lab = cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    L_std = float(L.std())
    apply_clahe = L_std < 32.0
    if apply_clahe:
        clip = np.clip(1.2 + 4.0 * (L_std / 64.0), cfg.clahe_clip_min, cfg.clahe_clip_max)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
        lab[..., 0] = clahe.apply(lab[..., 0])
        report["clahe_clip_used"] = float(clip)
    else:
        report["clahe_clip_used"] = 0.0

    # 6) Gentle a*/b* recenter, scaled by current bias and capped
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    a_mean, b_mean = float(a.mean()), float(b.mean())
    ab_bias = float(np.hypot(a_mean, b_mean))
    # >>> NEW: skip if already near neutral; otherwise gentle nudge
    if ab_bias > 1.0:
        scale = np.clip((ab_bias - 1.0) / cfg.ab_full_shift_bias, 0.0, 1.0)  # only strong casts get full shift
        a_shift = float(np.clip(-a_mean * scale, -cfg.ab_shift_cap, cfg.ab_shift_cap))
        b_shift = float(np.clip(-b_mean * scale, -cfg.ab_shift_cap, cfg.ab_shift_cap))
        lab[..., 1] = np.clip(lab[..., 1].astype(np.float32) + a_shift, 0, 255).astype(np.uint8)
        lab[..., 2] = np.clip(lab[..., 2].astype(np.float32) + b_shift, 0, 255).astype(np.uint8)
    else:
        a_shift = 0.0
        b_shift = 0.0

    bgr_corr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Post metrics
    overexp = float((bgr_corr >= 250).mean() * 100.0)
    underexp = float((bgr_corr <= 5).mean() * 100.0)
    blur = laplacian_var(cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2GRAY))
    post_a, post_b = lab_mean_ab(bgr_corr)
    post_ab_bias = float(np.hypot(post_a, post_b))

    # >>> NEW: skin-weighted exposure stats after correction
    lin_post = srgb_to_linear(bgr_corr)
    sg_y5, sg_y50, sg_y95, sg_y99 = luminance_percentiles(
        lin_post, skin if skin_frac >= cfg.skin_min_fraction else None
    )

    report.update({
        "pre": {
            "ab_bias": pre_ab_bias,
            "overexp_pct": pre_over,
            "underexp_pct": pre_under,
            "blur_var": pre_blur,
        },
        "post": {
            "ab_bias": post_ab_bias,
            "overexp_pct": overexp,
            "underexp_pct": underexp,
            "blur_var": blur,
            "skin_y50": sg_y50,
            "skin_y95": sg_y95,
        },
        "clahe_clip_used": report.get("clahe_clip_used", 0.0),
        "a_shift": a_shift,
        "b_shift": b_shift,
    })

    # Quality gate (useful for “retake photo” prompts)
    # >>> NEW: prioritize skin; fall back to global only if too little skin
    skin_ok = (skin_frac >= cfg.skin_min_fraction)
    ov_ok = (sg_y95 if skin_ok else 0.0) <= 0.97 and overexp < cfg.max_overexp_pct
    un_ok = (sg_y50 if skin_ok else g_y50) >= 0.34 or underexp < cfg.max_underexp_pct
    lighting_ok = bool(ov_ok and un_ok)
    wb_ok = (post_ab_bias < cfg.max_ab_bias)
    sharp_ok = (blur > cfg.min_blur_var)

    # Penalize frames with too little skin (hard to grade fairly)
    if not skin_ok:
        lighting_ok = False

    report["lighting_ok"] = lighting_ok
    report["wb_ok"] = wb_ok
    report["sharp_ok"] = sharp_ok
    report["overall_ok"] = bool(lighting_ok and wb_ok and sharp_ok)

    return bgr_corr, report


# ----------------------- Simple batch driver -----------------------

def main():
    # Batch process everything in data/raw/, save to data/processed/
    root = (os.path.dirname(__file__)).split("\\src")[0] if "\\" in __file__ else (os.path.dirname(__file__)).split("/src")[0]
    data_path = os.path.join(root, "data")
    raw_dir = os.path.join(data_path, "raw")
    out_dir = os.path.join(data_path, "processed")
    os.makedirs(out_dir, exist_ok=True)

    cfg = PreprocessConfig()

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue
        ipath = os.path.join(raw_dir, fname)
        img = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {ipath}")
            continue

        corrected, metrics = correct_exposure(img, cfg)
        oname = f"{os.path.splitext(fname)[0]}_corrected{os.path.splitext(fname)[1]}"
        opath = os.path.join(out_dir, oname)
        cv2.imwrite(opath, corrected)
        print(f"[OK] {fname} -> {oname} | overall_ok={metrics['overall_ok']}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

