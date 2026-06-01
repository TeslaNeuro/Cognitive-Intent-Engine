"""Vision pipeline.

Captures from a webcam in its own thread and continuously emits
`VisionFeatures` + an `EmotionDistribution` snapshot to the `StateStore`.

Feature extraction is done with MediaPipe FaceMesh when available (468
landmarks), with an OpenCV Haar-cascade fallback so the system still runs
in minimal environments.

Geometric features (all interpretable):
    - eye aspect ratio (EAR) — fatigue / blink
    - mouth curvature        — smile vs frown vs neutral
    - brow tension           — frustration / concentration
    - head yaw / pitch / roll
    - gaze direction (eyeball-centre vs eye-corner heuristic)
    - blink rate
    - attention score        — derived from gaze + head stability

The FER CNN is fed a 48x48 grayscale crop of the face.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from ..models.vision_emotion import VisionEmotionModel
from ..state.store import StateStore
from ..utils.config import AppConfig
from ..utils.logging import get_logger
from ..utils.schemas import EmotionDistribution, VisionFeatures

log = get_logger("vision")


# MediaPipe is optional; degrade gracefully if missing.
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _MP_AVAILABLE = False


# Selected landmark indices from MediaPipe FaceMesh (refined topology).
# Eyes — outer/inner corners + upper/lower lids (we use these for EAR).
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # outer, top1, top2, inner, bot2, bot1
_LEFT_EYE = [263, 387, 385, 362, 380, 373]
# Mouth corners + upper/lower lip midpoints.
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_UPPER_LIP = 13
_LOWER_LIP = 14
# Brows.
_LEFT_BROW = [70, 63, 105, 66, 107]
_RIGHT_BROW = [336, 296, 334, 293, 300]
# Nose tip & forehead anchor for head pose.
_NOSE_TIP = 1
_FOREHEAD = 10
_CHIN = 152


@dataclass
class _LatestFrame:
    rgb: Optional[np.ndarray] = None  # original frame for overlay
    overlay: Optional[np.ndarray] = None  # frame with drawings
    ts: float = 0.0


class VisionPipeline:
    def __init__(self, cfg: AppConfig, store: StateStore) -> None:
        self.cfg = cfg
        self.store = store
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._fer = VisionEmotionModel(weights_path=cfg.vision.emotion_model)
        self._mp_face = None
        if _MP_AVAILABLE and cfg.vision.use_mediapipe:
            try:
                self._mp_face = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                log.warning(f"MediaPipe FaceMesh init failed: {e}; using Haar cascade")
                self._mp_face = None
        self._haar = None
        if self._mp_face is None:
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # Blink + head-pose history for derived metrics.
        self._blink_state: bool = False
        self._blink_times: Deque[float] = deque(maxlen=20)
        self._head_pose_history: Deque[np.ndarray] = deque(maxlen=30)
        self._latest_emotion: EmotionDistribution = EmotionDistribution()
        self._latest_emotion_ts: float = 0.0
        self._emotion_every_n: int = 3   # FER inference every 3rd frame
        self._frame_counter: int = 0

        self._latest = _LatestFrame()
        self._latest_lock = threading.Lock()

    # ---------- lifecycle ----------
    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.cfg.vision.camera_index)
        if not self._cap.isOpened():
            log.warning(
                f"Could not open camera {self.cfg.vision.camera_index}; "
                f"vision pipeline will idle."
            )
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.vision.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.vision.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.cfg.vision.fps)
        self._thread = threading.Thread(target=self._run, name="VisionPipeline", daemon=True)
        self._thread.start()
        log.info("Vision pipeline started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        log.info("Vision pipeline stopped")

    # ---------- public ----------
    def latest_emotion(self) -> EmotionDistribution:
        return self._latest_emotion

    def get_latest_frame_jpeg(self, quality: int = 70) -> Optional[bytes]:
        """Used by the dashboard /video stream endpoint."""
        with self._latest_lock:
            frame = self._latest.overlay if self._latest.overlay is not None else self._latest.rgb
            if frame is None:
                return None
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return bytes(buf)

    # ---------- main loop ----------
    def _run(self) -> None:
        last_t = time.time()
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                # Synthesize an empty frame so downstream still gets ticks.
                self._publish_no_face()
                time.sleep(0.1)
                continue
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.01)
                continue
            if self.cfg.vision.flip_horizontal:
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            ts = time.time()
            feats, overlay, face_crop = self._extract(frame_rgb, ts)
            self.store.update_vision(feats)

            # FER inference every Nth frame on the cropped face (CPU-friendly).
            self._frame_counter += 1
            if face_crop is not None and (self._frame_counter % self._emotion_every_n == 0):
                gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (VisionEmotionModel.INPUT_SIZE,) * 2)
                self._latest_emotion = self._fer.predict(gray)
                self._latest_emotion_ts = ts

            with self._latest_lock:
                self._latest = _LatestFrame(rgb=frame_rgb, overlay=overlay, ts=ts)

            # Optional pacing — try to stay around configured FPS.
            elapsed = time.time() - last_t
            target = 1.0 / max(1, self.cfg.vision.fps)
            if elapsed < target:
                time.sleep(target - elapsed)
            last_t = time.time()

    # ---------- feature extraction ----------
    def _extract(
        self, rgb: np.ndarray, ts: float
    ) -> Tuple[VisionFeatures, np.ndarray, Optional[np.ndarray]]:
        h, w = rgb.shape[:2]
        overlay = rgb.copy()
        feats = VisionFeatures(ts=ts)

        face_crop: Optional[np.ndarray] = None

        if self._mp_face is not None:
            try:
                result = self._mp_face.process(rgb)
            except Exception:
                result = None
            if result and result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
                bbox = _bbox_from_pts(pts, w, h)
                feats.face_detected = True
                feats.bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
                feats.ear = _ear(pts, _LEFT_EYE) * 0.5 + _ear(pts, _RIGHT_EYE) * 0.5
                feats.mouth_curvature = _mouth_curvature(pts)
                feats.brow_tension = _brow_tension(pts)
                yaw, pitch, roll = _head_pose(pts, w, h)
                feats.head_yaw, feats.head_pitch, feats.head_roll = yaw, pitch, roll
                feats.gaze_x, feats.gaze_y = _gaze(pts)
                feats.blink_rate_hz = self._update_blinks(feats.ear, ts)
                feats.attention_score = self._attention(yaw, pitch, feats.gaze_x, feats.gaze_y)

                x, y, bw, bh = bbox
                pad = int(0.15 * max(bw, bh))
                x0 = max(0, x - pad); y0 = max(0, y - pad)
                x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
                if x1 > x0 and y1 > y0:
                    face_crop = rgb[y0:y1, x0:x1].copy()

                if self.cfg.vision.draw_overlays:
                    _draw_overlay(overlay, bbox, pts, feats, self._latest_emotion)
        elif self._haar is not None:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces):
                x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])
                feats.face_detected = True
                feats.bbox = [x / w, y / h, bw / w, bh / h]
                feats.attention_score = 0.6   # no landmarks → uninformed prior
                face_crop = rgb[y : y + bh, x : x + bw].copy()
                if self.cfg.vision.draw_overlays:
                    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        if not feats.face_detected and self.cfg.vision.draw_overlays:
            cv2.putText(
                overlay, "No face", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
            )

        return feats, overlay, face_crop

    def _publish_no_face(self) -> None:
        feats = VisionFeatures()
        self.store.update_vision(feats)

    # ---------- helpers ----------
    def _update_blinks(self, ear: float, ts: float) -> float:
        closed = ear < 0.20
        if closed and not self._blink_state:
            self._blink_times.append(ts)
        self._blink_state = closed
        if not self._blink_times:
            return 0.0
        window_s = 10.0
        recent = [t for t in self._blink_times if (ts - t) <= window_s]
        return len(recent) / window_s

    def _attention(self, yaw: float, pitch: float, gx: float, gy: float) -> float:
        # Penalize off-axis head and far gaze; combine into 0..1.
        head_pen = min(1.0, (abs(yaw) + abs(pitch)) / 60.0)
        gaze_pen = min(1.0, np.hypot(gx, gy))
        score = max(0.0, 1.0 - 0.6 * head_pen - 0.4 * gaze_pen)

        # Add stability bonus: less variation in recent head pose -> more attention.
        self._head_pose_history.append(np.array([yaw, pitch], dtype=np.float32))
        if len(self._head_pose_history) >= 5:
            var = float(np.var(np.stack(self._head_pose_history), axis=0).mean())
            stability = max(0.0, 1.0 - var / 100.0)
            score = 0.7 * score + 0.3 * stability
        return float(np.clip(score, 0.0, 1.0))


# --------------------------------------------------------------------------
# Pure geometry helpers
# --------------------------------------------------------------------------

def _bbox_from_pts(pts: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x = int(max(0, x_min))
    y = int(max(0, y_min))
    bw = int(min(w - x, x_max - x_min))
    bh = int(min(h - y, y_max - y_min))
    return x, y, bw, bh


def _ear(pts: np.ndarray, idx: list[int]) -> float:
    """Tereza-Soukupová eye-aspect-ratio."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    den = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return float(num / den)


def _mouth_curvature(pts: np.ndarray) -> float:
    """>0 if mouth corners are above the lip midline (smile), <0 if below."""
    lc = pts[_MOUTH_LEFT]; rc = pts[_MOUTH_RIGHT]
    up = pts[_UPPER_LIP]; lo = pts[_LOWER_LIP]
    midline_y = 0.5 * (up[1] + lo[1])
    avg_corner_y = 0.5 * (lc[1] + rc[1])
    mouth_w = np.linalg.norm(lc - rc) + 1e-6
    # negative y = up in image coords, so corners above midline -> midline_y - avg_corner_y > 0
    return float((midline_y - avg_corner_y) / mouth_w)


def _brow_tension(pts: np.ndarray) -> float:
    """Distance between inner brows + how lowered they are over the eyes."""
    lb = pts[_LEFT_BROW].mean(axis=0)
    rb = pts[_RIGHT_BROW].mean(axis=0)
    # Inner-brow separation (frown corrugator)
    inner_l = pts[_LEFT_BROW[0]]
    inner_r = pts[_RIGHT_BROW[0]]
    sep = np.linalg.norm(inner_l - inner_r)
    # Brow above eye distance (lowered brow = tension)
    eye_l = pts[_LEFT_EYE[0]]; eye_r = pts[_RIGHT_EYE[0]]
    brow_eye = 0.5 * ((eye_l[1] - lb[1]) + (eye_r[1] - rb[1]))
    # Normalize by inter-ocular distance
    iod = np.linalg.norm(eye_l - eye_r) + 1e-6
    inv_sep = max(0.0, 1.0 - (sep / iod))      # smaller separation -> more tension
    inv_be = max(0.0, 1.0 - (brow_eye / iod))  # smaller distance -> more tension
    return float(0.5 * inv_sep + 0.5 * inv_be)


def _head_pose(pts: np.ndarray, w: int, h: int) -> Tuple[float, float, float]:
    """SolvePnP-based head pose. Returns yaw/pitch/roll in degrees."""
    # 3D model points (in mm) for nose tip, chin, eye corners, mouth corners.
    model_pts = np.array([
        [0.0,   0.0,   0.0],     # nose
        [0.0, -63.6, -12.5],     # chin
        [-43.3, 32.7, -26.0],    # left eye outer
        [43.3,  32.7, -26.0],    # right eye outer
        [-28.9,-28.9, -24.1],    # mouth left
        [28.9, -28.9, -24.1],    # mouth right
    ], dtype=np.float64)
    image_pts = np.array([
        pts[_NOSE_TIP],
        pts[_CHIN],
        pts[_LEFT_EYE[0]],
        pts[_RIGHT_EYE[0]],
        pts[_MOUTH_LEFT],
        pts[_MOUTH_RIGHT],
    ], dtype=np.float64)

    focal = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy > 1e-6:
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        roll = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
    else:
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw = float(np.degrees(np.arctan2(-rmat[0, 1], rmat[1, 1])))
        roll = 0.0
    return yaw, pitch, roll


def _gaze(pts: np.ndarray) -> Tuple[float, float]:
    """Rough gaze direction from eye-corner geometry. Returns normalized (-1..1)."""
    # Use the eyeball-center landmarks if present (refined topology gives them).
    # Fallback to center of eye corners.
    try:
        left_center = pts[468]   # left iris center
        right_center = pts[473]  # right iris center
    except IndexError:
        left_center = (pts[_LEFT_EYE[0]] + pts[_LEFT_EYE[3]]) / 2.0
        right_center = (pts[_RIGHT_EYE[0]] + pts[_RIGHT_EYE[3]]) / 2.0

    left_outer = pts[_LEFT_EYE[0]]; left_inner = pts[_LEFT_EYE[3]]
    right_outer = pts[_RIGHT_EYE[0]]; right_inner = pts[_RIGHT_EYE[3]]
    left_w = np.linalg.norm(left_outer - left_inner) + 1e-6
    right_w = np.linalg.norm(right_outer - right_inner) + 1e-6
    left_mid = (left_outer + left_inner) / 2.0
    right_mid = (right_outer + right_inner) / 2.0
    gx = 0.5 * ((left_center[0] - left_mid[0]) / left_w + (right_center[0] - right_mid[0]) / right_w)
    # Crude vertical gaze: position of iris between top/bottom lid
    top_l = pts[_LEFT_EYE[1]]; bot_l = pts[_LEFT_EYE[5]]
    eye_h = np.linalg.norm(top_l - bot_l) + 1e-6
    gy = (left_center[1] - 0.5 * (top_l[1] + bot_l[1])) / eye_h
    return float(np.clip(gx * 2.0, -1.0, 1.0)), float(np.clip(gy * 2.0, -1.0, 1.0))


def _draw_overlay(
    overlay: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pts: np.ndarray,
    feats: VisionFeatures,
    emo: EmotionDistribution,
) -> None:
    x, y, bw, bh = bbox
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    top_label, top_p = emo.top()
    cv2.putText(
        overlay,
        f"{top_label} {top_p:.0%}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        overlay,
        f"EAR={feats.ear:.2f} attn={feats.attention_score:.2f}",
        (x, y + bh + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1
    )
    # Draw a couple of key landmarks
    for i in (_NOSE_TIP, _CHIN, _MOUTH_LEFT, _MOUTH_RIGHT):
        p = pts[i].astype(int)
        cv2.circle(overlay, tuple(p), 2, (255, 255, 0), -1)
