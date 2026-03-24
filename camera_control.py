from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ObstacleState = Literal["clear", "left", "right", "center"]


@dataclass
class CameraConfig:
    camera_index: int = 0


def detect_obstacle(frame) -> ObstacleState:
    """Detect obstacle location in a frame.

    This matches the heuristic used in the legacy monolithic script.
    """

    import cv2  # type: ignore

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


class CameraController:
    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cv2 = None
        self._cap = None

        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            self._cap = cv2.VideoCapture(cfg.camera_index)
        except Exception:
            self._cv2 = None
            self._cap = None

    @property
    def available(self) -> bool:
        return bool(self._cap is not None)

    def read_obstacle(self) -> Optional[ObstacleState]:
        if not self._cap:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        try:
            return detect_obstacle(frame)
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
