from __future__ import annotations

import select
from dataclasses import dataclass
from typing import Optional, Tuple

Command = Tuple[str, Optional[int]]


@dataclass
class IRConfig:
    device_path: str | None = None
    device_name_contains: str = "ir"


class IRController:
    """Optional IR controller using evdev.

    If evdev isn't installed or no suitable input device is available, this controller becomes a no-op.
    """

    def __init__(self, cfg: IRConfig):
        self._cfg = cfg
        self._evdev = None
        self._device = None
        self._mode = "off"  # off | device
        self.last_error: str = ""

        try:
            import evdev  # type: ignore

            self._evdev = evdev
            self._device = self._open_device(evdev)
            if self._device is None:
                self._mode = "off"
                if not self.last_error:
                    self.last_error = "no evdev IR device found"
                return

        except Exception as e:
            self._evdev = None
            self._device = None
            self._mode = "off"
            self.last_error = f"evdev import failed: {e}"

    def _open_device(self, evdev):
        patterns = []
        if self._cfg.device_path:
            patterns = [str(self._cfg.device_path)]
        else:
            try:
                patterns = list(evdev.list_devices())
            except Exception as e:
                self.last_error = f"evdev list_devices failed: {e}"
                return None

        hints = [
            str(self._cfg.device_name_contains or "").strip().lower(),
            "ir",
            "remote",
            "lirc",
            "receiver",
        ]

        candidates: list[str] = []
        for path in patterns:
            try:
                dev = evdev.InputDevice(path)
            except Exception as e:
                candidates.append(f"{path}: {e}")
                continue

            name = str(getattr(dev, "name", "") or "").lower()
            phys = str(getattr(dev, "phys", "") or "").lower()
            if self._cfg.device_path:
                self.last_error = ""
                return dev

            if any(h and (h in name or h in phys) for h in hints):
                self.last_error = ""
                return dev

            try:
                dev.close()
            except Exception:
                pass

        if candidates and not self.last_error:
            self.last_error = "no matching evdev IR device found"
        return None

    @property
    def available(self) -> bool:
        return self._mode == "device" and self._device is not None

    def poll_codes(self) -> list[str]:
        if not self._device or not self._evdev:
            if not self.last_error:
                self.last_error = "ir unavailable"
            return []
        try:
            device = self._device
            if device is None:
                return []

            ready, _, _ = select.select([device.fd], [], [], 0)
            if not ready:
                return []

            codes: list[str] = []
            for event in device.read():
                if event.type != self._evdev.ecodes.EV_KEY:
                    continue
                if int(getattr(event, "value", 0)) not in (1, 2):
                    continue
                keycode = self._evdev.categorize(event).keycode
                if isinstance(keycode, list):
                    codes.extend(str(c) for c in keycode)
                else:
                    codes.append(str(keycode))

            if codes:
                self.last_error = ""
            return codes
        except Exception as e:
            self.last_error = f"ir poll failed: {e}"
            return []

    @staticmethod
    def parse_code(code: str) -> Command | None:
        k = str(code).lower()
        if any(x in k for x in ("up", "vol+", "volumeup", "volume_up")):
            return ("forward", None)
        if any(x in k for x in ("down", "vol-", "volumedown", "volume_down")):
            return ("backward", None)
        if any(x in k for x in ("left", "rewind", "prev")):
            return ("left", None)
        if any(x in k for x in ("right", "forward", "next")):
            return ("right", None)
        if any(x in k for x in ("play", "pause", "ok", "enter")):
            return ("stop", None)
        if "power" in k:
            return ("stop", None)
        if "autonomous" in k or "mode" in k:
            return ("autonomous_toggle", None)
        if k.startswith("key_") and k[4:].isdigit():
            n = int(k[4:])
            val = max(0, min(100, int(n * 11)))
            return ("speed", val)
        if k.isdigit():
            n = int(k)
            val = max(0, min(100, int(n * 11)))
            return ("speed", val)
        return None

    def close(self) -> None:
        device = self._device
        if not device:
            return
        try:
            device.close()
        except Exception:
            pass
