from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

Command = Tuple[str, Optional[int]]


@dataclass
class VoiceConfig:
    mic_device_index: int | None = None


class VoiceController:
    """Optional voice controller using SpeechRecognition.

    If SpeechRecognition (and its audio deps) aren't installed, this controller becomes a no-op.
    """

    def __init__(self, command_queue: "queue.Queue[Command]", cfg: VoiceConfig):
        self._q = command_queue
        self._cfg = cfg

        self._sr = None
        self._recognizer = None
        self._mic = None
        self._stop_listening = None

        self._lock = threading.Lock()
        self._last_text: str | None = None
        self._last_cmd: Command | None = None
        self._last_ts: float | None = None

        try:
            import speech_recognition as sr  # type: ignore

            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._mic = sr.Microphone(device_index=cfg.mic_device_index) if cfg.mic_device_index is not None else sr.Microphone()
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception:
            self._sr = None
            self._recognizer = None
            self._mic = None

    @property
    def available(self) -> bool:
        return bool(self._sr and self._recognizer and self._mic)

    def start(self) -> None:
        if not self.available:
            return
        self._stop_listening = self._recognizer.listen_in_background(self._mic, self._callback, phrase_time_limit=4)

    def stop(self) -> None:
        if self._stop_listening:
            try:
                self._stop_listening(wait_for_stop=False)
            except Exception:
                pass
            self._stop_listening = None

    def _callback(self, recognizer, audio) -> None:
        sr = self._sr
        if sr is None:
            return
        try:
            text = recognizer.recognize_google(audio)
            text = str(text).lower().strip()
            cmd = self.parse_command(text)

            with self._lock:
                self._last_text = text
                self._last_cmd = cmd
                self._last_ts = time.time()

            if cmd:
                self._q.put(cmd)
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass
        except Exception:
            pass

    def last_event(self) -> tuple[str | None, Command | None, float | None]:
        """Return (last_text, last_cmd, last_timestamp)."""

        with self._lock:
            return self._last_text, self._last_cmd, self._last_ts

    @staticmethod
    def parse_command(text: str) -> Command | None:
        t = text.lower().strip()
        if any(kw in t for kw in ("forward", "go forward", "move forward")):
            return ("forward", None)
        if any(kw in t for kw in ("back", "backward", "go back", "move back")):
            return ("backward", None)
        if any(kw in t for kw in ("left", "turn left", "go left")):
            return ("left", None)
        if any(kw in t for kw in ("right", "turn right", "go right")):
            return ("right", None)
        if "stop" in t or "halt" in t:
            return ("stop", None)
        if "autonomous on" in t or "autonomy on" in t or "autonomous mode" in t:
            return ("autonomous_on", None)
        if "autonomous off" in t or "autonomy off" in t:
            return ("autonomous_off", None)
        if "speed" in t:
            for w in t.split():
                if w.isdigit():
                    return ("speed", int(w))
        return None
