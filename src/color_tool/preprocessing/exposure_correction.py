# src/color_tool/preprocessing/exposure_correction.py
"""
Minimal, safe exposure + white-balance correction for skin/hair/eye color work.

Design goals:
- Keep transforms global and near-linear so CIELAB/ΔE remain meaningful.
- Adjust exposure in *linear RGB* (undo gamma, scale, reapply gamma).
- White balance with Shades-of-Gray (p-norm); no per-channel histogram tricks.
- Light contrast lift on L* only (CLAHE).
- Small, capped centering of a*, b* to reduce residual cast.
- Return a metrics dict you can log or feed to a quality gate.

Dependencies: numpy, opencv-python
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import Tuple, Dict

import os 
import cv2
import numpy as np


# ----------------------- Config -----------------------

@dataclass
class PreprocessConfig:
    # Exposure
    target_mid_gray: float = 0.40      # desired midtone (0..1 in linear RGB)
    midtone_percentile: float = 55.0   # percentile treated as "midtone"
    exposure_gain_min: float = 0.5     # clamp to avoid artifacts
    exposure_gain_max: float = 2.5

    # White balance (Shades-of-Gray)
    wb_p_norm: int = 6                 # p=6 is a good robust default
    wb_gain_clip: float = 3.0          # max gain per channel

    # LAB adjustments
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    ab_shift_cap: float = 4.0          # max |Δa*|, |Δb*| centering

    # Quality thresholds (optional, for convenience)
    max_overexp_pct: float = 2.5
    max_underexp_pct: float = 2.5
    max_ab_bias: float = 3.0           # √(ā² + b̄²) tolerance
    min_blur_var: float = 60.0         # Laplacian variance lower bound


# ----------------------- Helpers -----------------------

def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB (0..255, BGR) -> linear (0..1, BGR)."""
    x = img_u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    """linear (0..1, BGR) -> uint8 sRGB (0..255, BGR)."""
    x = np.where(img_lin <= 0.0031308, 12.92 * img_lin,
                 1.055 * (img_lin ** (1 / 2.4)) - 0.055)
    return np.clip((x * 255.0).round(), 0, 255).astype(np.uint8)


def estimate_exposure_gain(img_lin_bgr: np.ndarray, target: float, pct: float,
                           gmin: float, gmax: float) -> float:
    """Compute a global exposure gain from a robust midtone percentile."""
    # Luminance from *linear* RGB (remember OpenCV stores BGR)
    B, G, R = img_lin_bgr[..., 0], img_lin_bgr[..., 1], img_lin_bgr[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    mid = float(np.percentile(Y, pct))
    if mid < 1e-6:
        return 1.0
    gain = np.clip(target / mid, gmin, gmax)
    return float(gain)


def shades_of_gray_wb(img_lin_bgr: np.ndarray, p: int, clip: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shades-of-Gray white balance (Minkowski p-norm).
    Returns (balanced_linear_bgr, gains_bgr).
    """
    eps = 1e-8
    # Compute per-channel p-norm means
    m = np.power(np.mean(np.power(img_lin_bgr, p), axis=(0, 1)) + eps, 1 / p)  # B,G,R
    gains = m.mean() / m
    gains = np.clip(gains, 1.0 / clip, clip)
    out = np.clip(img_lin_bgr * gains, 0.0, 1.0)
    return out, gains


def lab_mean_ab(bgr_u8: np.ndarray) -> Tuple[float, float]:
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    return float(a.mean()), float(b.mean())


def laplacian_var(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


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

    # 1) sRGB -> linear
    lin = srgb_to_linear(bgr_u8)

    # 2) global exposure gain in linear
    gain = estimate_exposure_gain(
        lin, cfg.target_mid_gray, cfg.midtone_percentile,
        cfg.exposure_gain_min, cfg.exposure_gain_max
    )
    lin = np.clip(lin * gain, 0.0, 1.0)
    report["exposure_gain"] = gain

    # 3) white balance (Shades-of-Gray)
    lin, gains_bgr = shades_of_gray_wb(lin, cfg.wb_p_norm, cfg.wb_gain_clip)
    report["wb_gains_bgr"] = tuple(float(x) for x in gains_bgr)

    # 4) back to sRGB for LAB work
    bgr_corr = linear_to_srgb(lin)

    # 5) CLAHE on L* only
    lab = cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    lab[..., 0] = clahe.apply(lab[..., 0])

    # 6) mild a*, b* centering (capped)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    a_mean, b_mean = float(a.mean()), float(b.mean())
    ab_bias = float(np.hypot(a_mean, b_mean))
    a_shift = float(np.clip(-a_mean, -cfg.ab_shift_cap, cfg.ab_shift_cap))
    b_shift = float(np.clip(-b_mean, -cfg.ab_shift_cap, cfg.ab_shift_cap))
    lab[..., 1] = np.clip(lab[..., 1].astype(np.float32) + a_shift, 0, 255).astype(np.uint8)
    lab[..., 2] = np.clip(lab[..., 2].astype(np.float32) + b_shift, 0, 255).astype(np.uint8)
    bgr_corr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 7) simple quality metrics
    overexp = float((bgr_u8 >= 250).mean() * 100.0)
    underexp = float((bgr_u8 <= 5).mean() * 100.0)
    blur = laplacian_var(cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2GRAY))

    report.update({
        "mean_a": a_mean, "mean_b": b_mean, "ab_bias": ab_bias,
        "overexp_pct": overexp, "underexp_pct": underexp,
        "blur_var": blur,
        "lighting_ok": (overexp < cfg.max_overexp_pct and underexp < cfg.max_underexp_pct),
        "wb_ok": (ab_bias < cfg.max_ab_bias and blur > cfg.min_blur_var),
    })

    return bgr_corr, report


# ----------------------- CLI -----------------------

# def _build_argparser():
#     ap = argparse.ArgumentParser(description="Exposure + WB correction")
#     ap.add_argument("--input", "-i", required=True, help="Path to input image")
#     ap.add_argument("--out", "-o", required=True, help="Path to save corrected image")
#     ap.add_argument("--json", default=None, help="Path to save metrics JSON (optional)")
#     return ap


def main():
    
    data_path = os.path.join((os.path.dirname(__file__)).split("\src")[0],"data")
    raw_img_path = os.path.join(data_path, "raw")
    processed_img_path = os.path.join(data_path, "processed")
    json_path = os.path.join(data_path, "json")

    # ap = _build_argparser()
    # args = ap.parse_args()
   
    input_img = os.path.join(raw_img_path, "sample.png") # path to input image
    output_img = os.path.join(processed_img_path , "sample_corrected.png") # path to save corrected image

    for img_name in os.listdir(raw_img_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        input_img = os.path.join(raw_img_path, img_name)

        img = cv2.imread(input_img, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read: {input_img}")

        corrected, metrics = correct_exposure(img)
        output_img = os.path.join(processed_img_path, img_name.split(".")[0] + "_corrected." + img_name.split(".")[-1])
        print(output_img)
        ok = cv2.imwrite(output_img, corrected)
        if not ok:
            raise SystemExit(f"Could not write: {output_img}")

        print(f"[OK] Saved corrected image -> {output_img}")
        print(json.dumps(metrics, indent=2))

        # create new json file with same name as input img
        # if not os.path.exists(json_path):
        #     os.makedirs(json_path)
        # with open(json_path, "w", encoding="utf-8") as f:
        #     json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

