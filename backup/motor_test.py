"""Simple 4-motor L298N GPIO script.

Default mode: WASD keyboard teleop ("car game" feel)
- Hold W: forward
- Hold S: reverse
- Hold A/D: steer while moving (release to straighten)
- Release keys: auto-stop after a short timeout (no need to press Space)
- Space: emergency stop
- Q: quit
- [: slower, ]: faster

If you leave the L298N EN jumpers installed (no PWM wiring), use `--full-speed`.

Optional: `--test` runs a basic forward/reverse test per motor.
Edit the pin constants to match your wiring.
Run this on a Raspberry Pi (or other Linux SBC) with GPIO.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import signal
import sys
import threading
import time
from typing import List, Sequence, Tuple

# -----------------------
# Wiring constants (EDIT)
# -----------------------
# Pin numbering is BCM (GPIOxx), not physical pin numbers.
#
# L298N uses:
# - EN  (enable / speed PWM)
# - IN1 / IN2 (direction)
#
# For each motor channel, remove the EN jumper on the driver and wire EN to a Raspberry Pi GPIO.
#
# Suggested PWM-capable GPIOs on Raspberry Pi: 12, 13, 18, 19 (hardware PWM). Software PWM also works.

# Front Right
M1_EN = 18
M1_IN1 = 5
M1_IN2 = 6

# Back Right
M2_EN = 19
M2_IN1 = 16
M2_IN2 = 20

# Back Left
M3_EN = 12
M3_IN1 = 21
M3_IN2 = 26

# Front Left
M4_EN = 13
M4_IN1 = 23
M4_IN2 = 24

# Group motors into left/right sides (EDIT if needed).
# Default assumes:
# - left side uses Motor 1 + Motor 2
# - right side uses Motor 3 + Motor 4
LEFT_MOTORS = ("Motor 3", "Motor 4")
RIGHT_MOTORS = ("Motor 1", "Motor 2")

# If a motor spins the wrong way when driving forward, flip its invert flag.
M1_INVERT = False
M2_INVERT = False
M3_INVERT = False
M4_INVERT = False

# Test timing (used by --test)
ON_TIME_S = 1.0
OFF_TIME_S = 0.5
REPEATS_PER_MOTOR = 2

# Teleop loop timing
POLL_INTERVAL_S = 0.01

# PWM and motion tuning
PWM_FREQUENCY_HZ = 200
DEFAULT_SPEED = 0.65  # 0..1 (user speed setting via [ and ])
SPEED_STEP = 0.05

# Max speed scaling (0..1) applied to the mixed left/right commands.
# Set MAX_REV_SPEED higher than MAX_FWD_SPEED if you want reverse to be faster.
MAX_FWD_SPEED = 1.00
MAX_REV_SPEED = 1.00

# Acceleration tuning (units: duty per second).
# Separate forward vs reverse acceleration so reverse can ramp faster.
FWD_ACCEL_PER_S = 1.0
REV_ACCEL_PER_S = 1.0
DECEL_PER_S = 2.5

COAST_ON_STOP = True

# Steering mix: higher = sharper turns while moving.
STEER_GAIN = 0.70

# If True, A/D will pivot-turn in place even when W/S is not held.
# If False, steering only affects motion while moving (car-like feel).
ALLOW_PIVOT_TURN = True

# Pivot turn speed scaling (0..1). Applied when throttle == 0.
PIVOT_SPEED = 0.45

SCRIPT_VERSION = "2026-02-24-wasd-car-controls"


class _PWM:
    def value_set(self, duty_0_to_1: float) -> None: ...
    def close(self) -> None: ...


class _DOut:
    def on(self) -> None: ...
    def off(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class MotorPins:
    name: str
    en: int
    in1: int
    in2: int
    invert: bool = False


class Motor:
    def __init__(self, pins: MotorPins, en_pwm: _PWM, in1: _DOut, in2: _DOut):
        self.pins = pins
        self._en = en_pwm
        self._in1 = in1
        self._in2 = in2
        self._speed = 0.0

    def set(self, speed: float) -> None:
        """Set speed in range [-1..1]."""
        speed = max(-1.0, min(1.0, float(speed)))
        if self.pins.invert:
            speed = -speed

        if abs(speed) < 1e-6:
            if COAST_ON_STOP:
                self._in1.off()
                self._in2.off()
            else:
                self._in1.on()
                self._in2.on()
            self._en.value_set(0.0)
            self._speed = 0.0
            return

        if speed > 0:
            self._in1.on()
            self._in2.off()
        else:
            self._in1.off()
            self._in2.on()

        self._en.value_set(abs(speed))
        self._speed = speed

    def stop(self) -> None:
        self.set(0.0)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            try:
                self._en.close()
            finally:
                try:
                    self._in1.close()
                finally:
                    self._in2.close()


def _build_motors(motors: Sequence[MotorPins]):
    """Create Motor objects using gpiozero (preferred) or RPi.GPIO (fallback)."""

    # Dry-run backend: no GPIO imports, just prints commands.
    # Enabled via --dry-run in main().

    try:
        from gpiozero import DigitalOutputDevice, PWMOutputDevice  # type: ignore

        built: List[Motor] = []
        for m in motors:
            en = PWMOutputDevice(m.en, frequency=PWM_FREQUENCY_HZ, initial_value=0.0)
            in1 = DigitalOutputDevice(m.in1, initial_value=False)
            in2 = DigitalOutputDevice(m.in2, initial_value=False)

            class _GZPWM:
                def __init__(self, dev):
                    self.dev = dev

                def value_set(self, duty_0_to_1: float) -> None:
                    self.dev.value = max(0.0, min(1.0, duty_0_to_1))

                def close(self) -> None:
                    self.dev.close()

            built.append(Motor(m, _GZPWM(en), in1, in2))

        def all_stop():
            for motor in built:
                motor.stop()

        def cleanup():
            for motor in built:
                motor.close()

        return built, all_stop, cleanup

    except Exception:
        pass

    try:
        import RPi.GPIO as GPIO  # type: ignore

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        class _GPIOPWM:
            def __init__(self, pin: int):
                self.pin = pin
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                self._pwm = GPIO.PWM(pin, PWM_FREQUENCY_HZ)
                self._pwm.start(0.0)

            def value_set(self, duty_0_to_1: float) -> None:
                duty = max(0.0, min(1.0, float(duty_0_to_1))) * 100.0
                self._pwm.ChangeDutyCycle(duty)

            def close(self) -> None:
                try:
                    self._pwm.ChangeDutyCycle(0.0)
                finally:
                    self._pwm.stop()

        class _GPIODOut:
            def __init__(self, pin: int):
                self.pin = pin
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

            def on(self) -> None:
                GPIO.output(self.pin, GPIO.HIGH)

            def off(self) -> None:
                GPIO.output(self.pin, GPIO.LOW)

            def close(self) -> None:
                return

        built: List[Motor] = []
        for m in motors:
            built.append(Motor(m, _GPIOPWM(m.en), _GPIODOut(m.in1), _GPIODOut(m.in2)))

        def all_stop():
            for motor in built:
                motor.stop()

        def cleanup():
            try:
                for motor in built:
                    motor.close()
            finally:
                GPIO.cleanup()

        return built, all_stop, cleanup

    except Exception as e:
        raise RuntimeError(
            "No supported GPIO library found. Install gpiozero (recommended) or RPi.GPIO, "
            "and run on a Raspberry Pi/Linux with GPIO access."
        ) from e


def _build_motors_dry_run(motors: Sequence[MotorPins]):
    built: List[Motor] = []

    class _DryPWM:
        def __init__(self, name: str):
            self._name = name
            self._last = None

        def value_set(self, duty_0_to_1: float) -> None:
            duty = max(0.0, min(1.0, float(duty_0_to_1)))
            if duty != self._last:
                print(f"[{self._name}] EN duty={duty:.2f}")
                self._last = duty

        def close(self) -> None:
            return

    class _DryDOut:
        def __init__(self, name: str):
            self._name = name
            self._state = 0

        def on(self) -> None:
            if self._state != 1:
                print(f"[{self._name}] ON")
                self._state = 1

        def off(self) -> None:
            if self._state != 0:
                print(f"[{self._name}] OFF")
                self._state = 0

        def close(self) -> None:
            return

    for m in motors:
        en = _DryPWM(f"{m.name}/EN(GPIO{m.en})")
        in1 = _DryDOut(f"{m.name}/IN1(GPIO{m.in1})")
        in2 = _DryDOut(f"{m.name}/IN2(GPIO{m.in2})")
        built.append(Motor(m, en, in1, in2))

    def all_stop():
        for motor in built:
            motor.stop()

    def cleanup():
        for motor in built:
            motor.close()

    return built, all_stop, cleanup


def _fmt_motor_pins(motors: Sequence[MotorPins]) -> str:
    return ", ".join(f"{m.name}: EN={m.en} IN1={m.in1} IN2={m.in2}{' (inv)' if m.invert else ''}" for m in motors)


@dataclass
class InputState:
    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    space: bool = False
    q: bool = False
    slower: bool = False  # '['
    faster: bool = False  # ']'

class _NoEchoTerminal:
    """Context manager to disable input echo on POSIX terminals.

    Without this, WASD often appears as typed characters in the terminal.
    """

    def __init__(self):
        self._enabled = False
        self._fd = None
        self._old = None

    def __enter__(self):
        if os.name == "nt":
            return self
        if not sys.stdin or not hasattr(sys.stdin, "fileno"):
            return self
        try:
            if not sys.stdin.isatty():
                return self
        except Exception:
            return self

        try:
            import termios
        except Exception:
            return self

        try:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)

            # Local flags: disable echo and canonical mode.
            new[3] = new[3] & ~termios.ECHO & ~termios.ICANON

            # In non-canonical mode, make reads return immediately.
            try:
                new[6][termios.VMIN] = 0
                new[6][termios.VTIME] = 0
            except Exception:
                pass

            # Keep Ctrl+C working (ISIG on by default). Leave it alone.
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            self._enabled = True
            self._fd = fd
            self._old = old
        except Exception:
            pass

        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return False
        try:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass
        return False


class _InputBackend:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def get_state(self) -> InputState:
        raise NotImplementedError

class _PynputBackend(_InputBackend):
    """Uses pynput to capture key down/up events (no repeat needed)."""

    def __init__(self):
        from pynput import keyboard as pynput_keyboard  # type: ignore

        self._pynput = pynput_keyboard
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None

    def start(self) -> None:
        kb = self._pynput

        def norm(key) -> str | None:
            try:
                if hasattr(key, "char") and key.char:
                    return str(key.char).lower()
            except Exception:
                pass
            if key == kb.Key.space:
                return "space"
            return None

        def on_press(key):
            k = norm(key)
            if k is None:
                return
            with self._lock:
                self._pressed.add(k)

        def on_release(key):
            k = norm(key)
            if k is None:
                return
            with self._lock:
                self._pressed.discard(k)

        self._listener = kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

    def get_state(self) -> InputState:
        with self._lock:
            pressed = set(self._pressed)

        return InputState(
            w="w" in pressed,
            a="a" in pressed,
            s="s" in pressed,
            d="d" in pressed,
            space="space" in pressed,
            q="q" in pressed,
            slower="[" in pressed,
            faster="]" in pressed,
        )

def _make_input_backend() -> _InputBackend:
    return _PynputBackend()


def _ramp(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(target, current + max_delta)
    if current > target:
        return max(target, current - max_delta)
    return current


def _ramp_signed(current: float, target: float, dt_s: float) -> float:
    same_dir = (current == 0.0) or (target == 0.0) or ((current > 0.0) == (target > 0.0))

    if not same_dir:
        max_delta = DECEL_PER_S * dt_s
        return _ramp(current, target, max_delta)

    if abs(target) > abs(current):
        accel = FWD_ACCEL_PER_S if target > 0.0 else REV_ACCEL_PER_S
        max_delta = accel * dt_s
        return _ramp(current, target, max_delta)

    max_delta = DECEL_PER_S * dt_s
    return _ramp(current, target, max_delta)


def run_test(motors: List[Motor]) -> int:
    print("Starting L298N motor test (ramp up/down forward + reverse per motor)...")
    print(f"ON_TIME_S={ON_TIME_S}  OFF_TIME_S={OFF_TIME_S}  PEAK_SPEED={DEFAULT_SPEED}")

    # Small timestep for smooth ramping.
    step_s = 0.02

    def ramp_run(motor: Motor, peak_speed: float, total_s: float) -> None:
        total_s = max(0.0, float(total_s))
        if total_s <= 0.0 or abs(peak_speed) < 1e-6:
            motor.stop()
            return

        up_s = total_s / 2.0
        down_s = total_s - up_s

        # Ramp up
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= up_s:
                break
            frac = 0.0 if up_s <= 1e-9 else (t / up_s)
            motor.set(peak_speed * frac)
            time.sleep(step_s)

        motor.set(peak_speed)

        # Ramp down
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= down_s:
                break
            frac = 0.0 if down_s <= 1e-9 else (1.0 - (t / down_s))
            motor.set(peak_speed * frac)
            time.sleep(step_s)

        motor.stop()

    try:
        for motor in motors:
            print(f"\nTesting {motor.pins.name}...")

            print("  FORWARD (ramp up then down)")
            ramp_run(motor, +DEFAULT_SPEED, ON_TIME_S)
            time.sleep(OFF_TIME_S)

            print("  REVERSE (ramp up then down)")
            ramp_run(motor, -DEFAULT_SPEED, ON_TIME_S)
            time.sleep(OFF_TIME_S)

        print("\nDone.")
        return 0
    except KeyboardInterrupt:
        for motor in motors:
            motor.stop()
        print("\nInterrupted. Motors stopped.")
        return 130


def run_teleop(motors: List[Motor], full_speed: bool) -> int:
    name_to_motor = {m.pins.name: m for m in motors}
    left = [name_to_motor[n] for n in LEFT_MOTORS if n in name_to_motor]
    right = [name_to_motor[n] for n in RIGHT_MOTORS if n in name_to_motor]
    if not left or not right:
        raise RuntimeError(
            "LEFT_MOTORS/RIGHT_MOTORS do not match motor names. "
            "Expected motor names: " + ", ".join(sorted(name_to_motor.keys()))
        )

    for m in motors:
        m.stop()

    print("WASD teleop started (press Q to quit)")
    print("  Hold W = forward", flush=True)
    print("  Hold S = reverse", flush=True)
    print("  Hold A/D = steer (while moving)", flush=True)
    print("  Release keys = auto-stop", flush=True)
    print("  Space = emergency stop", flush=True)
    print("  [ or - or Down = slower", flush=True)
    print("  ] or = or Up   = faster", flush=True)
    print("", flush=True)

    backend = _make_input_backend()
    actual_input = "pynput"
    print(f"Input backend: {actual_input}", flush=True)

    backend.start()

    speed_setting = 1.0 if full_speed else DEFAULT_SPEED
    last_printed = None

    target_left = 0.0
    target_right = 0.0
    current_left = 0.0
    current_right = 0.0
    last_t = time.time()

    last_state = InputState()

    def recompute_targets(throttle: float, steer: float) -> None:
        nonlocal target_left, target_right

        if full_speed:
            # Digital drive: no PWM/speed control. Each side is forward/reverse/off.
            # Turning while moving is done by cutting one side.
            if throttle == 0.0:
                if ALLOW_PIVOT_TURN and steer != 0.0:
                    target_left = -1.0 if steer > 0 else 1.0
                    target_right = 1.0 if steer > 0 else -1.0
                else:
                    target_left = 0.0
                    target_right = 0.0
                return

            if steer < 0.0:
                target_left = 0.0
                target_right = 1.0 if throttle > 0 else -1.0
                return
            if steer > 0.0:
                target_left = 1.0 if throttle > 0 else -1.0
                target_right = 0.0
                return

            target_left = 1.0 if throttle > 0 else -1.0
            target_right = 1.0 if throttle > 0 else -1.0
            return

        # Dead simple stop.
        if throttle == 0.0 and (steer == 0.0 or not ALLOW_PIVOT_TURN):
            target_left = 0.0
            target_right = 0.0
            return

        if throttle == 0.0 and ALLOW_PIVOT_TURN:
            # Pivot in place: left and right are opposite directions.
            base = max(0.0, min(1.0, PIVOT_SPEED)) * speed_setting
            # Positive steer (D) should pivot right: left forward, right reverse.
            left_cmd = (steer) * base
            right_cmd = (-steer) * base
        else:
            max_speed = MAX_FWD_SPEED if throttle > 0.0 else MAX_REV_SPEED
            base = max_speed * speed_setting
            # Positive steer (D) should turn right: speed up left / slow down right.
            left_cmd = (throttle + (steer * STEER_GAIN)) * base
            right_cmd = (throttle - (steer * STEER_GAIN)) * base

        target_left = max(-1.0, min(1.0, left_cmd))
        target_right = max(-1.0, min(1.0, right_cmd))

    recompute_targets(0.0, 0.0)

    try:
        with _NoEchoTerminal():
            while True:
                state = backend.get_state()
                if state.q:
                    break

                # Edge-detect speed adjustments so holding [ or ] doesn't spam.
                if not full_speed:
                    if state.slower and not last_state.slower:
                        speed_setting = max(0.0, speed_setting - SPEED_STEP)
                    if state.faster and not last_state.faster:
                        speed_setting = min(1.0, speed_setting + SPEED_STEP)

                if state.space:
                    target_left = 0.0
                    target_right = 0.0
                    current_left = 0.0
                    current_right = 0.0
                    for m in motors:
                        m.stop()
                else:
                    throttle = (1.0 if state.w else 0.0) + (-1.0 if state.s else 0.0)
                    steer = (-1.0 if state.a else 0.0) + (1.0 if state.d else 0.0)
                    throttle = max(-1.0, min(1.0, throttle))
                    steer = max(-1.0, min(1.0, steer))
                    recompute_targets(throttle, steer)

                now = time.time()
                dt = max(0.0, now - last_t)
                last_t = now

                if full_speed:
                    current_left = target_left
                    current_right = target_right
                else:
                    current_left = _ramp_signed(current_left, target_left, dt)
                    current_right = _ramp_signed(current_right, target_right, dt)

                motion = "fwd" if target_left > 0.0 or target_right > 0.0 else ("rev" if target_left < 0.0 or target_right < 0.0 else "stop")
                yaw = current_right - current_left
                if abs(yaw) < 1e-3:
                    turn = "straight"
                else:
                    # Differential-drive convention: right faster => turning left; left faster => turning right.
                    turn = "left" if yaw > 0 else "right"

                state_tuple = (
                    motion,
                    turn,
                    round(speed_setting, 2),
                    actual_input,
                    "full" if full_speed else "pwm",
                )
                if state_tuple != last_printed:
                    mode_str = "FULL" if full_speed else "PWM"
                    print(
                        f"State: {state_tuple[0]} turn:{state_tuple[1]}  Speed: {speed_setting:.2f}  "
                        f"Input: {actual_input}  Mode: {mode_str}"
                    )
                    last_printed = state_tuple

                for m in left:
                    m.set(current_left)
                for m in right:
                    m.set(current_right)

                last_state = state
                time.sleep(POLL_INTERVAL_S)

            return 0
    finally:
        try:
            backend.stop()
        except Exception:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="4-motor GPIO teleop/test")
    parser.add_argument("--test", action="store_true", help="run motor test instead of WASD teleop")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="run without GPIO; print motor commands")
    parser.add_argument(
        "--full-speed",
        "--full_speed",
        action="store_true",
        help="disable PWM/speed control (useful if EN jumpers are left on); turning cuts one side",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"motor_test.py version: {SCRIPT_VERSION}")

    motors_cfg: List[MotorPins] = [
        MotorPins("Motor 1", en=M1_EN, in1=M1_IN1, in2=M1_IN2, invert=M1_INVERT),
        MotorPins("Motor 2", en=M2_EN, in1=M2_IN1, in2=M2_IN2, invert=M2_INVERT),
        MotorPins("Motor 3", en=M3_EN, in1=M3_IN1, in2=M3_IN2, invert=M3_INVERT),
        MotorPins("Motor 4", en=M4_EN, in1=M4_IN1, in2=M4_IN2, invert=M4_INVERT),
    ]

    try:
        if args.dry_run:
            motors, all_off, cleanup = _build_motors_dry_run(motors_cfg)
        else:
            motors, all_off, cleanup = _build_motors(motors_cfg)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    stopping = False

    def _stop(*_args):
        nonlocal stopping
        stopping = True
        try:
            all_off()
        except Exception:
            pass
        # Exit so Ctrl+C / SIGTERM can't leave us trapped in teleop.
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print("Motor pins (BCM): " + _fmt_motor_pins(motors_cfg))
    print(f"Left motors: {LEFT_MOTORS}  Right motors: {RIGHT_MOTORS}")

    try:
        if args.test:
            return run_test(motors)
        return run_teleop(motors, full_speed=args.full_speed)
    finally:
        try:
            cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
