from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path

from voice_control import VoiceController


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal Vosk partial-stream transcription test")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "vosk-model-small-en-us-0.15"),
        help="Path to the Vosk model directory",
    )
    parser.add_argument("--device", type=int, default=None, help="Optional input device index")
    parser.add_argument("--samplerate", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--blocksize", type=int, default=4000, help="Audio block size")
    args = parser.parse_args()

    try:
        from vosk import KaldiRecognizer, Model
    except Exception as exc:
        print(f"Missing dependency: vosk ({exc})", file=sys.stderr)
        return 1

    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"Missing dependency: sounddevice ({exc})", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading model: {model_path}")
    model = Model(str(model_path))
    recognizer = KaldiRecognizer(model, float(args.samplerate))

    audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=32)

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        try:
            audio_q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    print("Speak now. Ctrl+C to stop.")
    print(f"Input device: {'default' if args.device is None else args.device}")
    print("-" * 60)

    last_partial = ""
    last_cmd = None

    try:
        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            device=args.device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    recognizer.Result()

                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                partial = str(partial).strip()
                if partial and partial != last_partial:
                    print(f"\rpartial: {partial}   ", end="", flush=True)
                    last_partial = partial

                cmd = VoiceController.parse_command(partial) if partial else None
                if cmd and cmd != last_cmd:
                    print(f"\ncmd: {cmd[0]}")
                    last_cmd = cmd
    except KeyboardInterrupt:
        print("\nStopping.")
        return 0
    except Exception as exc:
        print(f"\nStreaming failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())