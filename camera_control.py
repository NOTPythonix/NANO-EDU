from __future__ import annotations

import os
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
        self.last_error: str = ""

        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as e:
            self._cv2 = None
            self._cap = None
            self.last_error = f"opencv import failed: {e}"
            return

        try:
            cap = None
            if os.name == "nt":
                try:
                    cap = self._cv2.VideoCapture(cfg.camera_index, self._cv2.CAP_DSHOW)
                except Exception:
                    cap = None
            if cap is None:
                cap = self._cv2.VideoCapture(cfg.camera_index)
        except Exception as e:
            self._cap = None
            self.last_error = f"camera init failed for index {cfg.camera_index}: {e}"
            return

        if cap is None or not cap.isOpened():
            self._cap = None
            self.last_error = f"cannot open camera index {cfg.camera_index}"
            return

        # Best-effort; some backends ignore this.
        try:
            self._cv2.setUseOptimized(True)
        except Exception:
            pass
        try:
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
        except Exception:
            pass
        try:
            # Keep capture queue shallow to avoid stale-frame latency.
            cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._cap = cap
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self._cap is not None)

    def read_jpeg(self) -> Optional[bytes]:
        """Return a single JPEG frame, or None if unavailable."""

        if not self._cap or not self._cv2:
            if not self.last_error:
                self.last_error = "camera unavailable"
            return None
        ret, frame = self._cap.read()
        if not ret:
            self.last_error = "camera read failed"
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
            self.last_error = "jpeg encode failed"
            return None
        try:
            data = bytes(enc.tobytes())
            self.last_error = ""
            return data
        except Exception:
            self.last_error = "jpeg conversion failed"
            return None

    def close(self) -> None:
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
