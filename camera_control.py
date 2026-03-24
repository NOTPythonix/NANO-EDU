from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    """Camera capture settings for streaming to the server."""

    camera_index: int = 0
    width: int = 320
    height: int = 240
    jpeg_quality: int = 55


class CameraController:
    """Captures frames and encodes them as JPEG bytes.

    This replaces the legacy on-device obstacle heuristic. The intended design is:
    - client (Pi) captures frames
    - server performs inference
    - server sends back driving commands / annotations
    """

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cv2 = None
        self._cap = None

        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            cap = cv2.VideoCapture(cfg.camera_index)
            # Best-effort; some backends ignore this.
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
            except Exception:
                pass
            self._cap = cap
        except Exception:
            self._cv2 = None
            self._cap = None

    @property
    def available(self) -> bool:
        return bool(self._cap is not None)

    def read_jpeg(self) -> Optional[bytes]:
        """Return a single JPEG frame, or None if unavailable."""

        if not self._cap or not self._cv2:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None

        cv2 = self._cv2
        try:
            # Resize for bandwidth stability.
            frame = cv2.resize(frame, (int(self._cfg.width), int(self._cfg.height)))
        except Exception:
            pass

        q = max(10, min(95, int(self._cfg.jpeg_quality)))
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return None
        try:
            return bytes(enc.tobytes())
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
