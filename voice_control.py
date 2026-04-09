from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple

Command = Tuple[str, Optional[int]]


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"


def _tail_words(text: str, limit: int = 7) -> str:
    parts = str(text).strip().split()
    if len(parts) <= limit:
        return " ".join(parts)
    return " ".join(parts[-limit:])


@dataclass
class VoiceConfig:
    mic_device_index: int | None = None
    model_path: str | None = None
    sample_rate: int = 16000
    blocksize: int = 4000


class VoiceController:
    """Optional voice controller using Vosk partial streaming.

    If Vosk or sounddevice aren't installed, this controller becomes a no-op.
    """

    def __init__(self, command_queue: "queue.Queue[Command]", cfg: VoiceConfig):
        self._q = command_queue
        self._cfg = cfg

        self._sd = None
        self._vosk = None
        self._model = None
        self._recognizer = None
        self._stream = None
        self._audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=32)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._last_text: str | None = None
        self._last_cmd: Command | None = None
        self._last_ts: float | None = None
        self._last_emitted_cmd: Command | None = None
        self.last_error: str = ""

        try:
            from vosk import KaldiRecognizer, Model  # type: ignore
            import sounddevice as sd  # type: ignore

            self._sd = sd
            self._vosk = (KaldiRecognizer, Model)

            model_path = Path(cfg.model_path) if cfg.model_path else DEFAULT_MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(f"model not found: {model_path}")

            self._model = Model(str(model_path))
            self._recognizer = KaldiRecognizer(self._model, float(cfg.sample_rate))
            self._recognizer.SetWords(True)
            self.last_error = ""
        except Exception as e:
            self._sd = None
            self._vosk = None
            self._model = None
            self._recognizer = None
            self.last_error = f"vosk init failed: {e}"

    @property
    def available(self) -> bool:
        return bool(self._sd and self._vosk and self._model and self._recognizer)

    def start(self) -> None:
        if not self.available:
            if not self.last_error:
                self.last_error = "voice unavailable"
            return
        try:
            if self._stream is not None:
                return

            self._stop_event.clear()

            def callback(indata, frames, time_info, status) -> None:
                if status:
                    self.last_error = f"audio status: {status}"
                try:
                    self._audio_q.put_nowait(bytes(indata))
                except queue.Full:
                    pass

            self._stream = self._sd.RawInputStream(
                samplerate=float(self._cfg.sample_rate),
                blocksize=int(self._cfg.blocksize),
                device=self._cfg.mic_device_index,
                dtype="int16",
                channels=1,
                callback=callback,
            )
            self._stream.start()

            self._worker = threading.Thread(target=self._worker_loop, name="VoiceController", daemon=True)
            self._worker.start()
            self.last_error = ""
        except Exception as e:
            self._close_stream()
            self.last_error = f"voice start failed: {e}"

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._audio_q.put_nowait(b"")
        except queue.Full:
            pass
        self._close_stream()
        if self._worker is not None:
            try:
                self._worker.join(timeout=0.5)
            except Exception:
                pass
            self._worker = None

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._stop_event.is_set() or not data:
                continue

            recognizer = self._recognizer
            if recognizer is None:
                continue

            try:
                recognizer.AcceptWaveform(data)
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                partial = str(partial).strip()
                cmd = self.parse_command(partial) if partial else None
                display_text = _tail_words(partial) if partial else None

                with self._lock:
                    self._last_text = display_text or self._last_text
                    self._last_cmd = cmd or self._last_cmd
                    self._last_ts = time.time()

                if cmd and cmd != self._last_emitted_cmd:
                    self._last_emitted_cmd = cmd
                    self._q.put(cmd)
                self.last_error = ""
            except Exception as e:
                self.last_error = f"voice callback failed: {e}"

    def last_event(self) -> tuple[str | None, Command | None, float | None]:
        """Return (last_partial_text, last_cmd, last_timestamp)."""

        with self._lock:
            return _tail_words(self._last_text or "") or None, self._last_cmd, self._last_ts

    @staticmethod
    def parse_command(text: str) -> Command | None:
        t = str(text).lower().strip()
        if not t:
            return None

        candidates: list[tuple[int, Command]] = []

        def add_last(cmd: Command, patterns: list[str]) -> None:
            best = -1
            for pattern in patterns:
                for match in re.finditer(pattern, t):
                    best = max(best, match.start())
            if best >= 0:
                candidates.append((best, cmd))

        add_last(("forward", None), [r"\bmove forward\b", r"\bgo forward\b", r"\bturn forward\b", r"\bforward\b", r"\bforwards\b"])
        add_last(("backward", None), [r"\bmove back\b", r"\bgo back\b", r"\bturn back\b", r"\bbackward\b", r"\bbackwards\b", r"\breverse\b", r"\bback\b"])
        add_last(("left", None), [r"\bturn left\b", r"\bgo left\b", r"\bleft\b"])
        add_last(("right", None), [r"\bturn right\b", r"\bgo right\b", r"\bright\b"])
        add_last(("stop", None), [r"\bstop\b", r"\bhalt\b"])
        add_last(("autonomous_on", None), [r"\bautonomous on\b", r"\bautonomy on\b", r"\bautonomous mode\b"])
        add_last(("autonomous_off", None), [r"\bautonomous off\b", r"\bautonomy off\b"])

        for match in re.finditer(r"\bspeed\s+(\d{1,3})\b", t):
            try:
                candidates.append((match.start(), ("speed", int(match.group(1)))))
            except Exception:
                pass

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]


class AudioStreamClient:
    """Capture PCM16 mono audio chunks for network streaming (no local recognition)."""

    def __init__(self, cfg: VoiceConfig):
        self._cfg = cfg
        self._sd = None
        self._stream = None
        self._audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=12)
        self.last_error: str = ""
        self._last_chunk_ts: Optional[float] = None

        try:
            import sounddevice as sd  # type: ignore

            self._sd = sd
            self.last_error = ""
        except Exception as e:
            self._sd = None
            self.last_error = f"audio init failed: {e}"

    @property
    def available(self) -> bool:
        return bool(self._sd is not None)

    @property
    def sample_rate(self) -> int:
        return int(self._cfg.sample_rate)

    @property
    def last_chunk_ts(self) -> Optional[float]:
        return self._last_chunk_ts

    def start(self) -> None:
        if not self.available:
            if not self.last_error:
                self.last_error = "audio unavailable"
            return
        try:
            if self._stream is not None:
                return

            def callback(indata, frames, time_info, status) -> None:
                if status:
                    # Input overflow is common under load on SBCs; keep streaming and
                    # avoid latching a permanent error banner for a recoverable condition.
                    overflow = bool(getattr(status, "input_overflow", False))
                    if not overflow:
                        self.last_error = f"audio status: {status}"
                try:
                    self._audio_q.put_nowait(bytes(indata))
                    self._last_chunk_ts = time.time()
                    if str(self.last_error).startswith("audio status:"):
                        self.last_error = ""
                except queue.Full:
                    # Keep freshest audio by dropping oldest chunk first.
                    try:
                        _ = self._audio_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._audio_q.put_nowait(bytes(indata))
                        self._last_chunk_ts = time.time()
                    except Exception:
                        pass

            self._stream = self._sd.RawInputStream(
                samplerate=float(self._cfg.sample_rate),
                blocksize=int(self._cfg.blocksize),
                device=self._cfg.mic_device_index,
                dtype="int16",
                channels=1,
                callback=callback,
            )
            self._stream.start()
            self.last_error = ""
        except Exception as e:
            self.stop()
            self.last_error = f"audio start failed: {e}"

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def poll_chunks(self, *, max_chunks: int = 4) -> list[bytes]:
        out: list[bytes] = []
        limit = max(1, int(max_chunks))
        for _ in range(limit):
            try:
                out.append(self._audio_q.get_nowait())
            except queue.Empty:
                break
        return out


class ServerVoiceRecognizer:
    """Server-side Vosk recognizer fed by streamed PCM16 chunks."""

    def __init__(self, cfg: VoiceConfig):
        self._cfg = cfg
        self._recognizer = None
        self._lock = threading.Lock()
        self._last_text: Optional[str] = None
        self._last_cmd: Optional[Command] = None
        self._last_ts: Optional[float] = None
        self._last_emitted_cmd: Optional[Command] = None
        self._last_partial: str = ""
        self.last_error: str = ""

        try:
            from vosk import KaldiRecognizer, Model  # type: ignore

            model_path = Path(cfg.model_path) if cfg.model_path else DEFAULT_MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(f"model not found: {model_path}")

            model = Model(str(model_path))
            self._recognizer = KaldiRecognizer(model, float(cfg.sample_rate))
            self._recognizer.SetWords(True)
            self.last_error = ""
        except Exception as e:
            self._recognizer = None
            self.last_error = f"vosk init failed: {e}"

    @property
    def available(self) -> bool:
        return self._recognizer is not None

    def feed_chunk(self, data: bytes) -> Optional[Command]:
        rec = self._recognizer
        if rec is None or not data:
            return None

        try:
            rec.AcceptWaveform(data)

            partial_txt = ""
            try:
                partial_txt = str(json.loads(rec.PartialResult()).get("partial", "") or "").strip()
            except Exception:
                partial_txt = ""
            if partial_txt:
                self._last_partial = partial_txt

            text = partial_txt or self._last_partial
            if text:
                with self._lock:
                    self._last_text = _tail_words(text)
                    self._last_ts = time.time()

            cmd = VoiceController.parse_command(text) if text else None
            if cmd:
                with self._lock:
                    self._last_cmd = cmd
                    self._last_ts = time.time()
                if cmd != self._last_emitted_cmd:
                    self._last_emitted_cmd = cmd
                    return cmd

            self.last_error = ""
            return None
        except Exception as e:
            self.last_error = f"voice callback failed: {e}"
            return None

    def last_event(self) -> tuple[str | None, Command | None, float | None]:
        with self._lock:
            return self._last_text, self._last_cmd, self._last_ts
