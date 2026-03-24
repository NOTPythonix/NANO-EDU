from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from camera_control import CameraConfig, CameraController
from ir_control import IRConfig, IRController
from movement import DriveConfig, SkidSteerDrive, build_motors, load_motor_pins, mix_throttle_steer, run_motor_test
from voice_control import VoiceConfig, VoiceController


@dataclass
class RuntimeState:
    speed_setting: float = 0.65  # 0..1
    autonomous: bool = False
    manual_throttle: float = 0.0
    manual_steer: float = 0.0


class PynputKeys:
    def __init__(self):
        try:
            from pynput import keyboard as pynput_keyboard  # type: ignore
        except Exception as e:
            raise RuntimeError("pynput is required for keyboard control. Install with: pip install pynput") from e

        self._kb = pynput_keyboard
        self._pressed: Set[str] = set()
        self._lock = threading.Lock()
        self._listener = None

        self._events: "queue.Queue[Tuple[str, str]]" = queue.Queue()

    def start(self) -> None:
        kb = self._kb

        def norm(key) -> Optional[str]:
            try:
                if hasattr(key, "char") and key.char:
                    return str(key.char).lower()
            except Exception:
                pass
            if key == kb.Key.space:
                return "space"
            return None

        def on_press(key) -> None:
            k = norm(key)
            if k is None:
                return
            with self._lock:
                was_down = k in self._pressed
                self._pressed.add(k)
            if not was_down:
                self._events.put(("down", k))

        def on_release(key) -> None:
            k = norm(key)
            if k is None:
                return
            with self._lock:
                self._pressed.discard(k)
            self._events.put(("up", k))

        self._listener = kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

    def snapshot(self) -> Set[str]:
        with self._lock:
            return set(self._pressed)

    def poll_event(self) -> Optional[Tuple[str, str]]:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None


def _apply_command(state: RuntimeState, cmd: Tuple[str, Optional[int]]) -> None:
    name, val = cmd

    if name == "forward":
        state.autonomous = False
        state.manual_throttle = 1.0
        state.manual_steer = 0.0
        return

    if name == "backward":
        state.autonomous = False
        state.manual_throttle = -1.0
        state.manual_steer = 0.0
        return

    if name == "left":
        state.autonomous = False
        state.manual_steer = -1.0
        return

    if name == "right":
        state.autonomous = False
        state.manual_steer = 1.0
        return

    if name == "stop":
        state.autonomous = False
        state.manual_throttle = 0.0
        state.manual_steer = 0.0
        return

    if name == "autonomous_on":
        state.autonomous = True
        state.manual_throttle = 0.0
        state.manual_steer = 0.0
        return

    if name == "autonomous_off":
        state.autonomous = False
        return

    if name == "speed" and isinstance(val, int):
        # Accept 0..100 from voice/IR.
        state.speed_setting = max(0.0, min(1.0, float(val) / 100.0))
        return


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Robot controller (refactored modules; pynput keyboard only)")
    parser.add_argument("--test", action="store_true", help="run motor test (forward+reverse ramp per motor)")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="run without GPIO; print motor commands")
    parser.add_argument(
        "--full-speed",
        "--full_speed",
        action="store_true",
        help="disable PWM/speed control (useful if EN jumpers are left on); turning cuts one side",
    )
    parser.add_argument(
        "--mic",
        type=int,
        default=None,
        help="microphone device index for voice control",
    )
    args = parser.parse_args(argv)

    motors_cfg, left_names, right_names = load_motor_pins()
    motor_map, stop_all, cleanup = build_motors(
        motors_cfg,
        dry_run=bool(args.dry_run),
        pwm_frequency_hz=DriveConfig().pwm_frequency_hz,
    )

    if args.test:
        try:
            return run_motor_test(
                list(motor_map.values()),
                peak_speed=0.65,
                on_time_s=1.0,
                off_time_s=0.5,
            )
        finally:
            try:
                cleanup()
            except Exception:
                pass

    cfg = DriveConfig()
    drive = SkidSteerDrive(
        motor_map,
        left_names=left_names,
        right_names=right_names,
        cfg=cfg,
        full_speed=bool(args.full_speed),
    )

    keys = PynputKeys()
    keys.start()

    voice_q: "queue.Queue[Tuple[str, Optional[int]]]" = queue.Queue()
    ir_q: "queue.Queue[Tuple[str, Optional[int]]]" = queue.Queue()

    vc = VoiceController(voice_q, VoiceConfig(mic_device_index=args.mic))
    if vc.available:
        vc.start()
        print("Voice: enabled")
    else:
        print("Voice: unavailable (SpeechRecognition/PyAudio not installed)")

    irc = IRController(IRConfig())
    if irc.available:
        print("IR: enabled (requires LIRC configured)")
    else:
        print("IR: unavailable (python-lirc not installed)")

    cam = CameraController(CameraConfig(camera_index=0))
    if cam.available:
        print("Camera: enabled")
    else:
        print("Camera: unavailable (OpenCV not installed or no camera)")

    print("Controls: hold W/A/S/D, Space=stop, O=toggle autonomous, [ / ] speed, Q=quit")

    state = RuntimeState()
    last_t = time.time()

    try:
        while True:
            # Handle key down events for toggles/edges
            while True:
                ev = keys.poll_event()
                if ev is None:
                    break
                kind, k = ev
                if kind == "down" and k == "o":
                    if cam.available:
                        state.autonomous = not state.autonomous
                        if state.autonomous:
                            state.manual_throttle = 0.0
                            state.manual_steer = 0.0
                        print(f"Autonomous: {'ON' if state.autonomous else 'OFF'}")
                    else:
                        print("Autonomous requested, but camera is unavailable")
                elif kind == "down" and k == "[":
                    state.speed_setting = max(0.0, state.speed_setting - 0.05)
                elif kind == "down" and k == "]":
                    state.speed_setting = min(1.0, state.speed_setting + 0.05)

            pressed = keys.snapshot()
            if "q" in pressed:
                break

            if "space" in pressed:
                drive.emergency_stop()
                state.manual_throttle = 0.0
                state.manual_steer = 0.0
                state.autonomous = False

            # Voice commands
            try:
                while True:
                    cmd = voice_q.get_nowait()
                    _apply_command(state, cmd)
            except queue.Empty:
                pass

            # IR commands
            if irc.available:
                for code in irc.poll_codes():
                    cmd = irc.parse_code(code)
                    if cmd:
                        ir_q.put(cmd)

            try:
                while True:
                    cmd = ir_q.get_nowait()
                    _apply_command(state, cmd)
            except queue.Empty:
                pass

            if state.autonomous:
                obs = cam.read_obstacle()
                if obs is None:
                    target_left = 0.0
                    target_right = 0.0
                else:
                    # Legacy behavior: clear -> forward; left obstacle -> turn right; right obstacle -> turn left.
                    if obs == "clear":
                        throttle, steer = 1.0, 0.0
                    elif obs == "left":
                        throttle, steer = 0.0, 1.0
                    elif obs == "right":
                        throttle, steer = 0.0, -1.0
                    else:
                        throttle, steer = 0.0, -1.0
                    target_left, target_right = mix_throttle_steer(
                        throttle,
                        steer,
                        speed_setting=state.speed_setting,
                        cfg=cfg,
                        full_speed=bool(args.full_speed),
                    )
            else:
                # Keyboard manual (car-game feel)
                throttle = (1.0 if "w" in pressed else 0.0) + (-1.0 if "s" in pressed else 0.0)
                steer = (-1.0 if "a" in pressed else 0.0) + (1.0 if "d" in pressed else 0.0)
                if any(k in pressed for k in ("w", "a", "s", "d")):
                    state.manual_throttle = max(-1.0, min(1.0, throttle))
                    state.manual_steer = max(-1.0, min(1.0, steer))
                else:
                    state.manual_throttle = 0.0
                    state.manual_steer = 0.0

                target_left, target_right = mix_throttle_steer(
                    state.manual_throttle,
                    state.manual_steer,
                    speed_setting=state.speed_setting,
                    cfg=cfg,
                    full_speed=bool(args.full_speed),
                )

            now = time.time()
            dt = max(0.0, now - last_t)
            last_t = now

            drive.update(target_left=target_left, target_right=target_right, dt_s=dt)

            time.sleep(0.02)

        return 0

    finally:
        try:
            keys.stop()
        except Exception:
            pass
        try:
            vc.stop()
        except Exception:
            pass
        try:
            irc.close()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        try:
            stop_all()
        except Exception:
            pass
        try:
            cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
