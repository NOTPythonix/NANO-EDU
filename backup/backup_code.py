#!/usr/bin/env python3
"""backup_code.py

Backup of the original monolithic robot controller.

This file is also used as the single source of truth for motor pinout constants:
- `MOTORS`
- `DIR_CORRECT`

To keep development workable on non-Raspberry Pi machines (e.g., Windows), this module
avoids importing `RPi.GPIO`, `cv2`, `termios`, etc. at import time.

Run the legacy controller (Raspberry Pi / Linux only):
  python3 backup_code.py
"""

from __future__ import annotations

from typing import Any

# --- Motor pins (BCM GPIO) - user-provided mapping ---
# User confirmed these are BCM GPIO numbers on the real Pi.
# If any are wrong, the code should fail fast rather than auto-detect.
MOTORS = {
    "lf": {"in1": 11, "in2": 14, "en": 15},
    "rr": {"in1": 20, "in2": 16, "en": 23},
    "rf": {"in1": 26, "in2": 19, "en": 13},
    "lr": {"in1": 31, "in2": 29, "en": 12},
}

# --- Direction correction ---
# True  => forward means (IN1=HIGH, IN2=LOW)
# False => swap direction for that motor
DIR_CORRECT = {
    "lf": True,
    "lr": True,
    "rf": True,
    "rr": True,
}

PWM_FREQ = 1000
DEFAULT_SPEED = 100

# IR receiver GPIO (data pin)
IR_GPIO = 17


def _run_backup_main(argv: list[str] | None = None) -> int:
    """Run the legacy monolithic controller.

    This is intentionally self-contained so importing this module for pin constants
    doesn't require Pi-only dependencies.
    """

    import argparse
    import atexit
    import queue
    import select
    import sys
    import time

    # POSIX-only terminal handling.
    try:
        import termios  # type: ignore
        import tty  # type: ignore
    except Exception as e:
        raise RuntimeError("This backup script must be run in a POSIX terminal (Linux/Raspberry Pi).") from e

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required to run the backup script (camera mode).") from e

    try:
        import RPi.GPIO as GPIO  # type: ignore
    except Exception as e:
        raise RuntimeError("RPi.GPIO required to run the backup script on Raspberry Pi.") from e

    # Optional voice dependencies
    try:
        import speech_recognition as sr  # type: ignore

        VOICE_AVAILABLE = True
    except Exception:
        VOICE_AVAILABLE = False
        sr = None  # type: ignore

    # Optional LIRC for IR remote decoding
    try:
        import lirc  # type: ignore

        LIRC_AVAILABLE = True
    except Exception:
        LIRC_AVAILABLE = False
        lirc = None  # type: ignore

    class KeyReader:
        def __init__(self):
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.restore)

        def restore(self):
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass

        def get_key(self):
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return sys.stdin.read(1)
            return None

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    pwms: dict[str, Any] = {}

    def setup_motors():
        for name, pins in MOTORS.items():
            GPIO.setup(pins["in1"], GPIO.OUT)
            GPIO.setup(pins["in2"], GPIO.OUT)
            GPIO.setup(pins["en"], GPIO.OUT)
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.LOW)
            pwm = GPIO.PWM(pins["en"], PWM_FREQ)
            pwm.start(0)
            pwms[name] = pwm

    def set_motor_direction(name, forward=True):
        pins = MOTORS[name]
        real_forward = forward if DIR_CORRECT.get(name, True) else not forward
        if real_forward:
            GPIO.output(pins["in1"], GPIO.HIGH)
            GPIO.output(pins["in2"], GPIO.LOW)
        else:
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.HIGH)

    def set_speed(speed):
        s = max(0, min(100, int(speed)))
        for pwm in pwms.values():
            pwm.ChangeDutyCycle(s)

    def stop_all():
        for name, pins in MOTORS.items():
            GPIO.output(pins["in1"], GPIO.LOW)
            GPIO.output(pins["in2"], GPIO.LOW)
            pwms[name].ChangeDutyCycle(0)

    def cleanup():
        stop_all()
        for pwm in pwms.values():
            pwm.stop()
        GPIO.cleanup()

    def forward(speed):
        for m in MOTORS:
            set_motor_direction(m, True)
        set_speed(speed)

    def backward(speed):
        for m in MOTORS:
            set_motor_direction(m, False)
        set_speed(speed)

    def turn_right(speed):
        for m in ("rf", "lr"):
            set_motor_direction(m, False)
        for m in ("rr", "lf"):
            set_motor_direction(m, True)
        set_speed(speed)

    def turn_left(speed):
        for m in ("rf", "lr"):
            set_motor_direction(m, True)
        for m in ("rr", "lf"):
            set_motor_direction(m, False)
        set_speed(speed)

    def detect_obstacle(frame):
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

    class VoiceController:
        def __init__(self, command_queue, mic_device_index=None):
            self.command_queue = command_queue
            self.recognizer = None
            self.mic = None
            self.stop_listening = None
            self.mic_device_index = mic_device_index
            if VOICE_AVAILABLE:
                self._setup()

        def _setup(self):
            try:
                self.recognizer = sr.Recognizer()  # type: ignore[union-attr]
                if self.mic_device_index is None:
                    self.mic = sr.Microphone()  # type: ignore[union-attr]
                else:
                    self.mic = sr.Microphone(device_index=self.mic_device_index)  # type: ignore[union-attr]
                with self.mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception as e:
                print("VoiceController: microphone setup failed:", e)
                self.mic = None
                self.recognizer = None

        def start(self):
            if not VOICE_AVAILABLE or not self.mic or not self.recognizer:
                return
            self.stop_listening = self.recognizer.listen_in_background(self.mic, self._callback, phrase_time_limit=4)

        def stop(self):
            if self.stop_listening:
                self.stop_listening(wait_for_stop=False)
                self.stop_listening = None

        def _callback(self, recognizer, audio):
            try:
                text = recognizer.recognize_google(audio)
                text = text.lower().strip()
                cmd = self._parse_command(text)
                if cmd:
                    self.command_queue.put(cmd)
            except sr.UnknownValueError:  # type: ignore[union-attr]
                pass
            except sr.RequestError:  # type: ignore[union-attr]
                pass

        def _parse_command(self, text):
            if any(kw in text for kw in ("forward", "go forward", "move forward")):
                return ("forward", None)
            if any(kw in text for kw in ("back", "backward", "go back", "move back")):
                return ("backward", None)
            if any(kw in text for kw in ("left", "turn left", "go left")):
                return ("left", None)
            if any(kw in text for kw in ("right", "turn right", "go right")):
                return ("right", None)
            if "stop" in text or "halt" in text:
                return ("stop", None)
            if "autonomous on" in text or "autonomy on" in text or "autonomous mode" in text:
                return ("autonomous_on", None)
            if "autonomous off" in text or "autonomy off" in text:
                return ("autonomous_off", None)
            if "speed" in text:
                words = text.split()
                for w in words:
                    if w.isdigit():
                        return ("speed", int(w))
            return None

    class IRController:
        def __init__(self, lirc_socket_name="robot"):
            self.lirc_socket_name = lirc_socket_name
            self.client = None
            if LIRC_AVAILABLE:
                try:
                    self.client = lirc.Client(self.lirc_socket_name)  # type: ignore[union-attr]
                except Exception:
                    try:
                        lirc.init(self.lirc_socket_name, blocking=False)  # type: ignore[union-attr]
                        self.client = None
                    except Exception:
                        self.client = None

        def poll(self):
            if not LIRC_AVAILABLE:
                return []
            try:
                if hasattr(self, "client") and self.client:
                    codes = self.client.next(timeout=0)
                    if not codes:
                        return []
                    return codes
                codes = lirc.nextcode()  # type: ignore[union-attr]
                return codes or []
            except Exception:
                return []

        def parse_ir_key(self, key: str):
            k = str(key).lower()
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
                return ("autonomous_on", None)
            if k.startswith("key_") and k[4:].isdigit():
                n = int(k[4:])
                return ("speed", max(0, min(100, int(n * 11))))
            if k.isdigit():
                n = int(k)
                return ("speed", max(0, min(100, int(n * 11))))
            return None

    def legacy_main(mic_device_index=None):
        setup_motors()
        kr = KeyReader()
        speed = DEFAULT_SPEED
        autonomous = False

        cam = cv2.VideoCapture(0)
        voice_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        vc = VoiceController(voice_queue, mic_device_index=mic_device_index)
        vc.start()

        irc = IRController()

        print("Hold W/A/S/D to move manually")
        print("Press O = autonomous mode ON")
        print("Press Q = quit program")

        try:
            while True:
                key = kr.get_key()
                if key:
                    key = key.lower()
                    if key == "q":
                        break
                    if key == "o":
                        autonomous = True
                        print("Autonomous mode enabled")
                    if key in ("w", "a", "s", "d"):
                        autonomous = False
                    if not autonomous:
                        if key == "w":
                            forward(speed)
                        elif key == "s":
                            backward(speed)
                        elif key == "a":
                            turn_left(speed)
                        elif key == "d":
                            turn_right(speed)

                try:
                    while True:
                        cmd, val = voice_queue.get_nowait()
                        if cmd == "forward":
                            autonomous = False
                            forward(speed)
                        elif cmd == "backward":
                            autonomous = False
                            backward(speed)
                        elif cmd == "left":
                            autonomous = False
                            turn_left(speed)
                        elif cmd == "right":
                            autonomous = False
                            turn_right(speed)
                        elif cmd == "stop":
                            autonomous = False
                            stop_all()
                        elif cmd == "autonomous_on":
                            autonomous = True
                        elif cmd == "autonomous_off":
                            autonomous = False
                        elif cmd == "speed" and isinstance(val, int):
                            speed = max(0, min(100, val))
                            set_speed(speed)
                except queue.Empty:
                    pass

                if LIRC_AVAILABLE:
                    for code in irc.poll():
                        parsed = irc.parse_ir_key(code)
                        if parsed:
                            cmd, val = parsed
                            if cmd == "forward":
                                autonomous = False
                                forward(speed)
                            elif cmd == "backward":
                                autonomous = False
                                backward(speed)
                            elif cmd == "left":
                                autonomous = False
                                turn_left(speed)
                            elif cmd == "right":
                                autonomous = False
                                turn_right(speed)
                            elif cmd == "stop":
                                autonomous = False
                                stop_all()
                            elif cmd == "autonomous_on":
                                autonomous = True
                            elif cmd == "autonomous_off":
                                autonomous = False
                            elif cmd == "speed" and isinstance(val, int):
                                speed = max(0, min(100, val))
                                set_speed(speed)

                if autonomous:
                    ret, frame = cam.read()
                    if not ret:
                        stop_all()
                        continue
                    state = detect_obstacle(frame)
                    if state == "clear":
                        forward(speed)
                    elif state == "left":
                        turn_right(speed)
                    else:
                        turn_left(speed)

                if not key and not autonomous:
                    stop_all()

                time.sleep(0.02)

        finally:
            vc.stop()
            try:
                if LIRC_AVAILABLE and hasattr(lirc, "deinit"):
                    lirc.deinit()  # type: ignore[union-attr]
            except Exception:
                pass
            cam.release()
            cleanup()

    parser = argparse.ArgumentParser(description="Robot control (backup monolithic).")
    parser.add_argument("--mic", type=int, default=None, help="Microphone device index.")
    args = parser.parse_args(argv)
    legacy_main(mic_device_index=args.mic)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_backup_main())
