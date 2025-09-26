"""
Face & Eye Gate (class-based)
- Fair, tone-agnostic gate to decide if a portrait is usable for eye-colour analysis.
- Prefers geometry & image-quality signals over appearance attributes.

Dependencies:
  - Required: numpy, opencv-python
  - Optional: mediapipe (for Face Mesh + Iris). If unavailable, falls back to OpenCV cascades.

Public API:
  gate = FaceEyeGate(config=FaceEyeGateConfig())
  result = gate.evaluate(bgr_u8)
  result = {
    "ok": bool, "reason": str, "confidence": float,
    "used_mediapipe": bool, "iod_px": float, "pose_ok": bool, "crop_ok": bool,
    "eye_quality": {
        "blur_var": float, "underexp_pct": float, "overexp_pct": float, "highlight_pct": float
    },
    "iris": {
        "pixels": int, "L_med": float, "a_med": float, "b_med": float, "h_deg": float
    }
  }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math
import numpy as np
import cv2

# ------------------------------ Optional MediaPipe ---------------------------------
_HAS_MEDIAPIPE = True
try:
    import mediapipe as mp
    _MP_FACE_MESH = mp.solutions.face_mesh
except Exception:
    _HAS_MEDIAPIPE = False
    _MP_FACE_MESH = None


# ------------------------------ Config --------------------------------------------

@dataclass
class FaceEyeGateConfig:
    # Geometry gates
    min_interocular_px: int = 90          # min distance between eye corners
    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 15.0
    max_roll_deg: float = 10.0

    # Eye quality gates (computed on iris region if available, else tight eye ROI)
    min_blur_var: float = 120.0           # Laplacian variance threshold (sharpness)
    max_underexp_pct: float = 10.0        # % pixels with L* < 10 allowed
    max_overexp_pct: float = 2.0          # % pixels with L* > 95 allowed
    max_highlight_pct: float = 10.0       # % HSV V>240 allowed (speculars)

    # Iris mask cleanup
    erode_px: int = 2
    min_mask_pixels: int = 200

    # Confidence weights (simple linear combiner)
    w_geom: float = 0.45
    w_quality: float = 0.35
    w_area: float = 0.20


# ------------------------------ Helpers -------------------------------------------

def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _lab(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return L, a, b

def _exposure_stats_eye(bgr: np.ndarray) -> Tuple[float, float, float]:
    L, _, _ = _lab(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    under = float((L < 10).mean() * 100.0)
    over  = float((L > 95).mean() * 100.0)
    highlights = float((hsv[..., 2] > 240).mean() * 100.0)
    return under, over, highlights

def _hue_deg(a_med: float, b_med: float) -> float:
    angle = math.degrees(math.atan2(b_med, a_med))
    return (angle + 360.0) % 360.0

def _poly_mask(h: int, w: int, pts_xy: np.ndarray) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts_xy.astype(np.int32)], 255)
    return m


# ------------------------------ Main class ----------------------------------------

class FaceEyeGate:
    def __init__(self, config: Optional[FaceEyeGateConfig] = None):
        self.cfg = config or FaceEyeGateConfig()
        self.used_mediapipe = False

        # Fallback detectors
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # MediaPipe FaceMesh (if available)
        self._mp_ctx = None
        if _HAS_MEDIAPIPE:
            self._mp_ctx = _MP_FACE_MESH.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,  # gives iris landmarks
                max_num_faces=1,
                min_detection_confidence=0.5
            )

    # -------------------- Public entry point --------------------

    def evaluate(self, bgr_u8: np.ndarray) -> Dict:
        """
        Returns a dictionary with 'ok', 'reason', 'confidence', and detailed fields.
        """
        h, w = bgr_u8.shape[:2]

        # Try MediaPipe path first
        if self._mp_ctx is not None:
            mp_result = self._mediapipe_path(bgr_u8)
            if mp_result is not None:
                self.used_mediapipe = True
                return mp_result

        # Fallback to OpenCV cascades
        self.used_mediapipe = False
        return self._opencv_fallback_path(bgr_u8)

    # -------------------- MediaPipe path ------------------------

    def _mediapipe_path(self, bgr: np.ndarray) -> Optional[Dict]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._mp_ctx.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark

        # landmarks we’ll use (MediaPipe indices):
        # 33, 263 ~ eye outer corners; 133, 362 ~ eye inner corners
        pts = {
            "L_outer": lm[33], "R_outer": lm[263],
            "L_inner": lm[133], "R_inner": lm[362],
            "nose_tip": lm[1],
            "left_ear": lm[234], "right_ear": lm[454]
        }
        def to_xy(p): return np.array([p.x * w, p.y * h])
        L_out, R_out = to_xy(pts["L_outer"]), to_xy(pts["R_outer"])
        IOD = float(np.linalg.norm(L_out - R_out))

        # Pose (coarse): yaw from ear symmetry; roll from eye slope; pitch from nose-eye vertical
        yaw = self._estimate_yaw(pts, w, h)
        roll = self._estimate_roll(pts, w, h)
        pitch = self._estimate_pitch(pts, w, h)

        pose_ok = (abs(yaw) <= self.cfg.max_yaw_deg and
                   abs(pitch) <= self.cfg.max_pitch_deg and
                   abs(roll) <= self.cfg.max_roll_deg)

        # Iris polygons (if refine_landmarks=True)
        # Left iris: indices 469..473; Right iris: 474..478
        L_iris = [lm[i] for i in [469,470,471,472,473] if i < len(lm)]
        R_iris = [lm[i] for i in [474,475,476,477,478] if i < len(lm)]
        if not L_iris or not R_iris:
            return None

        L_poly = np.array([[p.x*w, p.y*h] for p in L_iris], np.float32)
        R_poly = np.array([[p.x*w, p.y*h] for p in R_iris], np.float32)

        mask_L = _poly_mask(h, w, L_poly)
        mask_R = _poly_mask(h, w, R_poly)
        mask = cv2.bitwise_or(mask_L, mask_R)
        if self.cfg.erode_px > 0:
            mask = cv2.erode(mask, np.ones((self.cfg.erode_px, self.cfg.erode_px), np.uint8))

        crop_ok = int(mask.sum() > 0) and mask.sum() >= self.cfg.min_mask_pixels

        # Eye-region quality
        eye_q = self._eye_quality_metrics(bgr, mask)

        # Iris colour stats (robust)
        iris_stats = self._iris_stats(bgr, mask)

        # Geometry gate
        geom_ok = (IOD >= self.cfg.min_interocular_px) and pose_ok and crop_ok

        # Decision
        quality_ok = (eye_q["blur_var"] >= self.cfg.min_blur_var and
                      eye_q["underexp_pct"] <= self.cfg.max_underexp_pct and
                      eye_q["overexp_pct"]  <= self.cfg.max_overexp_pct and
                      eye_q["highlight_pct"] <= self.cfg.max_highlight_pct)

        ok = bool(geom_ok and quality_ok)
        reason = "OK"
        if not geom_ok:
            if IOD < self.cfg.min_interocular_px: reason = "Move closer (face too small)"
            elif not pose_ok: reason = "Turn to camera (pose too oblique)"
            elif not crop_ok: reason = "Eyes partially cropped/occluded"
        elif not quality_ok:
            reason = self._quality_reason(eye_q)

        confidence = self._confidence(IOD, pose_ok, crop_ok, eye_q, mask)

        return {
            "ok": ok, "reason": reason, "confidence": float(confidence),
            "used_mediapipe": True,
            "iod_px": IOD, "pose_ok": pose_ok, "crop_ok": bool(crop_ok),
            "eye_quality": eye_q,
            "iris": iris_stats
        }

    # -------------------- OpenCV fallback path ------------------

    def _opencv_fallback_path(self, bgr: np.ndarray) -> Dict:
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(80, 80))
        if len(faces) == 0:
            return {
                "ok": False, "reason": "No face detected", "confidence": 0.0,
                "used_mediapipe": False, "iod_px": 0.0, "pose_ok": False, "crop_ok": False,
                "eye_quality": {"blur_var": 0.0, "underexp_pct": 0.0, "overexp_pct": 0.0, "highlight_pct": 0.0},
                "iris": {"pixels": 0, "L_med": 0.0, "a_med": 0.0, "b_med": 0.0, "h_deg": 0.0}
            }

        # Use the largest face
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        roi_gray = gray[y:y+fh, x:x+fw]
        roi_bgr  = bgr[y:y+fh, x:x+fw]

        # Eyes within face ROI
        eyes = self._eye_cascade.detectMultiScale(roi_gray, 1.15, 6, minSize=(20, 15))
        if len(eyes) < 1:
            return {
                "ok": False, "reason": "Face detected but eyes not found", "confidence": 0.1,
                "used_mediapipe": False, "iod_px": 0.0, "pose_ok": False, "crop_ok": False,
                "eye_quality": {"blur_var": 0.0, "underexp_pct": 0.0, "overexp_pct": 0.0, "highlight_pct": 0.0},
                "iris": {"pixels": 0, "L_med": 0.0, "a_med": 0.0, "b_med": 0.0, "h_deg": 0.0}
            }

        # Build a tight eye mask by combining eye rectangles and shrinking
        mask = np.zeros((h, w), np.uint8)
        iod_px = 0.0
        centers = []
        for (ex, ey, ew, eh) in eyes[:2]:  # top two detections
            ex0, ey0 = x + ex, y + ey
            roi_rect = (ex0, ey0, ew, eh)
            # shrink rectangle to inner 60% to avoid lashes
            k = 0.6
            cx, cy = ex0 + ew//2, ey0 + eh//2
            rw, rh = int(ew * k), int(eh * k)
            rx, ry = int(cx - rw/2), int(cy - rh/2)
            cv2.rectangle(mask, (rx, ry), (rx+rw, ry+rh), 255, -1)
            centers.append((cx, cy))

        if len(centers) >= 2:
            c1, c2 = centers[:2]
            iod_px = float(math.hypot(c1[0]-c2[0], c1[1]-c2[1]))

        if self.cfg.erode_px > 0:
            mask = cv2.erode(mask, np.ones((self.cfg.erode_px, self.cfg.erode_px), np.uint8))

        crop_ok = int(mask.sum() > 0) and mask.sum() >= self.cfg.min_mask_pixels
        pose_ok = True  # can’t reliably estimate pose without landmarks

        # Quality & stats
        eye_q = self._eye_quality_metrics(bgr, mask)
        iris_stats = self._iris_stats(bgr, mask)

        geom_ok = (iod_px >= self.cfg.min_interocular_px) and pose_ok and crop_ok
        quality_ok = (eye_q["blur_var"] >= self.cfg.min_blur_var and
                      eye_q["underexp_pct"] <= self.cfg.max_underexp_pct and
                      eye_q["overexp_pct"]  <= self.cfg.max_overexp_pct and
                      eye_q["highlight_pct"] <= self.cfg.max_highlight_pct)

        ok = bool(geom_ok and quality_ok)
        reason = "OK"
        if not geom_ok:
            if iod_px < self.cfg.min_interocular_px: reason = "Move closer (face too small)"
            elif not crop_ok: reason = "Eyes partially cropped/occluded"
        elif not quality_ok:
            reason = self._quality_reason(eye_q)

        confidence = self._confidence(iod_px, pose_ok, crop_ok, eye_q, mask)

        return {
            "ok": ok, "reason": reason, "confidence": float(confidence),
            "used_mediapipe": False,
            "iod_px": iod_px, "pose_ok": pose_ok, "crop_ok": bool(crop_ok),
            "eye_quality": eye_q,
            "iris": iris_stats
        }

    # -------------------- Subroutines --------------------------

    def _eye_quality_metrics(self, bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {"blur_var": 0.0, "underexp_pct": 100.0, "overexp_pct": 0.0, "highlight_pct": 0.0}

        x, y, w, h = cv2.boundingRect(mask)
        roi = bgr[y:y+h, x:x+w]
        m  = mask[y:y+h, x:x+w]
        # masked crop
        roi_masked = cv2.bitwise_and(roi, roi, mask=m)
        gray = cv2.cvtColor(roi_masked, cv2.COLOR_BGR2GRAY)
        blur = _lap_var(gray[m > 0])

        # robust exposure stats only within mask
        under, over, highlights = _exposure_stats_eye(roi_masked[m > 0].reshape(-1, 3).reshape(-1, 1, 3)
                                                      if False else roi_masked[m > 0].reshape(-1, 3))

        # Workaround: compute stats directly from masked pixels
        px = roi_masked[m > 0]
        if px.size == 0:
            return {"blur_var": 0.0, "underexp_pct": 100.0, "overexp_pct": 0.0, "highlight_pct": 0.0}
        L, _, _ = _lab(px.reshape(-1, 1, 3))
        hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        under = float((L < 10).mean() * 100.0)
        over  = float((L > 95).mean() * 100.0)
        highlights = float((hsv[..., 2] > 240).mean() * 100.0)

        return {"blur_var": float(blur), "underexp_pct": under, "overexp_pct": over, "highlight_pct": highlights}

    def _iris_stats(self, bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        px = bgr[mask > 0]
        if px.size < self.cfg.min_mask_pixels:
            return {"pixels": int(px.shape[0]), "L_med": 0.0, "a_med": 0.0, "b_med": 0.0, "h_deg": 0.0}
        L, A, B = _lab(px.reshape(-1, 1, 3))
        Lm = float(np.median(L))
        am = float(np.median(A))
        bm = float(np.median(B))
        return {
            "pixels": int(px.shape[0]),
            "L_med": Lm, "a_med": am, "b_med": bm,
            "h_deg": _hue_deg(am, bm)
        }

    def _quality_reason(self, q: Dict[str, float]) -> str:
        if q["blur_var"] < self.cfg.min_blur_var: return "Image too blurry (hold still / focus)"
        if q["underexp_pct"] > self.cfg.max_underexp_pct: return "Too dark (increase light on face)"
        if q["overexp_pct"] > self.cfg.max_overexp_pct: return "Too bright (reduce glare/backlight)"
        if q["highlight_pct"] > self.cfg.max_highlight_pct: return "Strong reflections on eyes"
        return "Quality below threshold"

    def _confidence(self, iod_px: float, pose_ok: bool, crop_ok: bool,
                    q: Dict[str, float], mask: np.ndarray) -> float:
        # Normalize features to 0..1 and combine
        g = min(1.0, iod_px / (self.cfg.min_interocular_px * 1.2)) * (1.0 if pose_ok and crop_ok else 0.5)
        blur = min(1.0, q["blur_var"] / (self.cfg.min_blur_var * 1.5))
        exp = 1.0
        if q["underexp_pct"] > 0: exp *= max(0.0, 1.0 - q["underexp_pct"]/ (self.cfg.max_underexp_pct*2))
        if q["overexp_pct"]  > 0: exp *= max(0.0, 1.0 - q["overexp_pct"] / (self.cfg.max_overexp_pct*2))
        hl  = max(0.0, 1.0 - q["highlight_pct"]/(self.cfg.max_highlight_pct*2))
        quality = 0.6*blur + 0.3*exp + 0.1*hl
        area = min(1.0, float((mask > 0).sum()) / float(self.cfg.min_mask_pixels*2))

        conf = self.cfg.w_geom*g + self.cfg.w_quality*quality + self.cfg.w_area*area
        return float(np.clip(conf, 0.0, 1.0))

    # ---- Pose estimation helpers (coarse; MediaPipe only) -----

    def _estimate_roll(self, pts, w, h) -> float:
        # roll: slope between outer eye corners
        def xy(i): return np.array([pts[i].x * w, pts[i].y * h])
        L, R = xy(33), xy(263)
        dy, dx = (R - L)[1], (R - L)[0]
        return math.degrees(math.atan2(dy, dx))

    def _estimate_yaw(self, pts, w, h) -> float:
        # yaw: relative ear horizontal positions (~zero when frontal)
        left = pts[234].x * w
        right = pts[454].x * w
        nose = pts[1].x * w
        # normalize by face width
        fw = abs(right - left) + 1e-6
        yaw_norm = ((nose - (left+right)/2.0) / fw) * 100.0
        return float(np.clip(yaw_norm, -45, 45))

    def _estimate_pitch(self, pts, w, h) -> float:
        # pitch: nose vertical relative to eyes
        nose_y = pts[1].y * h
        eye_y  = 0.5 * (pts[33].y * h + pts[263].y * h)
        fh = abs((pts[10].y - pts[152].y) * h) + 1e-6  # approx face height
        pitch_norm = ((eye_y - nose_y) / fh) * 180.0
        return float(np.clip(pitch_norm, -45, 45))


# ------------------------------ Example usage -------------------------------------
if __name__ == "__main__":
    # Manual, non-CLI usage example
    import os
    import cv2
    print(cv2.__file__)

    # cfg = FaceEyeGateConfig()
    # gate = FaceEyeGate(cfg)

    # # Adjust path to a test image in your repo
    # img_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "sample5.png")
    # img_path = os.path.abspath(img_path)
    # print(f"Testing FaceEyeGate on image: {img_path}")
    # bgr = cv2.imread(img_path)
    # if bgr is None:
    #     raise SystemExit(f"Could not read test image at {img_path}")

    # result = gate.evaluate(bgr)
    # print(result)
