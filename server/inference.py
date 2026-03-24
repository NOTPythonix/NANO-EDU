from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

ObstacleState = Literal["clear", "left", "right", "center", "unknown"]


@dataclass
class InferenceResult:
    obstacle: ObstacleState = "unknown"
    throttle: float = 0.0
    steer: float = 0.0


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def detect_obstacle_from_bgr(frame) -> ObstacleState:
    """Legacy heuristic (ported from old client camera code)."""

    cv2 = _try_import_cv2()
    if cv2 is None:
        return "unknown"

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "clear"
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 6000:
            return "clear"
        x, y, w, h = cv2.boundingRect(largest)
        center_x = x + w // 2
        if center_x < 150:
            return "left"
        if center_x > 350:
            return "right"
        return "center"
    except Exception:
        return "unknown"


def infer_from_jpeg_b64(jpeg_b64: str) -> InferenceResult:
    cv2 = _try_import_cv2()
    if cv2 is None:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0)

    try:
        raw = base64.b64decode(jpeg_b64.encode("ascii"), validate=False)
    except Exception:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0)

    try:
        import numpy as np  # type: ignore

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0)
    except Exception:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0)

    obs = detect_obstacle_from_bgr(frame)

    # Map obstacle -> throttle/steer (same behavior as legacy autonomy)
    if obs == "clear":
        throttle, steer = 1.0, 0.0
    elif obs == "left":
        throttle, steer = 0.0, 1.0
    elif obs == "right":
        throttle, steer = 0.0, -1.0
    elif obs == "center":
        throttle, steer = 0.0, -1.0
    else:
        throttle, steer = 0.0, 0.0

    return InferenceResult(obstacle=obs, throttle=float(throttle), steer=float(steer))
