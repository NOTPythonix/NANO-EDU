from __future__ import annotations

import base64
import os
from pathlib import Path
from urllib.request import urlopen
from dataclasses import dataclass
from typing import Literal, Optional

ObstacleState = Literal["clear", "left", "right", "center", "unknown"]


@dataclass
class InferenceResult:
    obstacle: ObstacleState = "unknown"
    throttle: float = 0.0
    steer: float = 0.0
    label: str = "—"
    confidence: float = 0.0


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


_MODEL_DIR = Path(__file__).resolve().parent / "models"
_PROTO_PATH = _MODEL_DIR / "mobilenet_ssd_deploy.prototxt"
_WEIGHTS_PATH = _MODEL_DIR / "mobilenet_ssd_deploy.caffemodel"
_PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
_WEIGHTS_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Treat dynamic agents and major furniture as obstacles for indoor robot navigation.
_OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "bus",
    "car",
    "motorbike",
    "chair",
    "diningtable",
    "sofa",
    "tvmonitor",
    "pottedplant",
    "dog",
    "cat",
}

_NET = None
_NET_ERR: str = ""


def _download_to(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=20) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"empty download from {url}")
    path.write_bytes(data)


def _load_net(cv2):
    global _NET, _NET_ERR
    if _NET is not None:
        return _NET
    if _NET_ERR:
        return None

    allow_download = str(os.environ.get("ROBOT_OD_ALLOW_DOWNLOAD", "1")).strip() not in ("0", "false", "False")

    try:
        if allow_download:
            if not _PROTO_PATH.exists():
                _download_to(_PROTO_PATH, _PROTO_URL)
            if not _WEIGHTS_PATH.exists():
                _download_to(_WEIGHTS_PATH, _WEIGHTS_URL)

        if not _PROTO_PATH.exists() or not _WEIGHTS_PATH.exists():
            _NET_ERR = "detector model files missing"
            return None

        _NET = cv2.dnn.readNetFromCaffe(str(_PROTO_PATH), str(_WEIGHTS_PATH))
        return _NET
    except Exception as e:
        _NET = None
        _NET_ERR = str(e)
        return None


def detect_obstacle_from_bgr(frame) -> tuple[ObstacleState, str, float]:
    """DNN object detection: classify obstacle direction from highest-confidence obstacle class."""

    cv2 = _try_import_cv2()
    if cv2 is None:
        return "unknown", "opencv-missing", 0.0

    net = _load_net(cv2)
    if net is None:
        return "unknown", ("model-unavailable" if not _NET_ERR else _NET_ERR[:42]), 0.0

    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        det = net.forward()

        conf_thr = max(0.15, min(0.9, float(os.environ.get("ROBOT_OD_CONF", "0.45"))))
        min_area_ratio = max(0.001, min(0.5, float(os.environ.get("ROBOT_OD_MIN_AREA", "0.015"))))

        best = None
        best_score = -1.0
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < conf_thr:
                continue

            cls_idx = int(det[0, 0, i, 1])
            if cls_idx < 0 or cls_idx >= len(_CLASSES):
                continue
            label = _CLASSES[cls_idx]
            if label not in _OBSTACLE_CLASSES:
                continue

            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area_ratio = float(bw * bh) / float(max(1, w * h))
            if area_ratio < min_area_ratio:
                continue

            # Prefer confident and visually significant detections.
            score = conf * (1.0 + min(1.0, area_ratio * 10.0))
            if score > best_score:
                best_score = score
                best = (x1, y1, x2, y2, label, conf)

        if best is None:
            return "clear", "none", 0.0

        x1, y1, x2, y2, label, conf = best
        cx = (x1 + x2) * 0.5
        nx = cx / float(max(1, w))
        if nx < 0.40:
            return "left", str(label), float(conf)
        if nx > 0.60:
            return "right", str(label), float(conf)
        return "center", str(label), float(conf)
    except Exception:
        return "unknown", "dnn-error", 0.0


def infer_from_jpeg_b64(jpeg_b64: str) -> InferenceResult:
    cv2 = _try_import_cv2()
    if cv2 is None:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="opencv-missing", confidence=0.0)

    try:
        raw = base64.b64decode(jpeg_b64.encode("ascii"), validate=False)
    except Exception:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="decode-error", confidence=0.0)

    try:
        import numpy as np  # type: ignore

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="imdecode-none", confidence=0.0)
    except Exception:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="numpy-error", confidence=0.0)

    obs, label, conf = detect_obstacle_from_bgr(frame)

    # Map obstacle -> throttle/steer.
    if obs == "clear":
        throttle, steer = 0.9, 0.0
    elif obs == "left":
        throttle, steer = 0.0, 1.0
    elif obs == "right":
        throttle, steer = 0.0, -1.0
    elif obs == "center":
        throttle, steer = -0.2, -1.0
    else:
        throttle, steer = 0.0, 0.0

    return InferenceResult(
        obstacle=obs,
        throttle=float(throttle),
        steer=float(steer),
        label=str(label),
        confidence=float(conf),
    )
