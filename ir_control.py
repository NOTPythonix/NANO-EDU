from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Command = Tuple[str, Optional[int]]


@dataclass
class IRConfig:
    lirc_socket_name: str = "robot"


class IRController:
    """Optional IR controller using python-lirc.

    If python-lirc isn't installed or LIRC isn't configured, this controller becomes a no-op.
    """

    def __init__(self, cfg: IRConfig):
        self._cfg = cfg
        self._lirc = None
        self._client = None

        try:
            import lirc  # type: ignore

            self._lirc = lirc

            # Prefer modern Client API if available.
            try:
                self._client = lirc.Client(cfg.lirc_socket_name)
            except Exception:
                try:
                    lirc.init(cfg.lirc_socket_name, blocking=False)
                    self._client = None
                except Exception:
                    self._lirc = None
                    self._client = None
        except Exception:
            self._lirc = None
            self._client = None

    @property
    def available(self) -> bool:
        return self._lirc is not None

    def poll_codes(self) -> list[str]:
        if not self._lirc:
            return []
        try:
            if self._client:
                codes = self._client.next(timeout=0)
                if not codes:
                    return []
                return [str(c) for c in codes]
            codes = self._lirc.nextcode()
            return [str(c) for c in (codes or [])]
        except Exception:
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
        lirc = self._lirc
        if not lirc:
            return
        try:
            if hasattr(lirc, "deinit"):
                lirc.deinit()
        except Exception:
            pass
