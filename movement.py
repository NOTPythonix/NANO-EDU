from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import robot_config


@dataclass(frozen=True)
class MotorPins:
    name: str
    en: int
    in1: int
    in2: int
    invert: bool


class _PWM:
    def change(self, duty_0_to_1: float) -> None: ...

    def stop(self) -> None: ...


class _DOut:
    def on(self) -> None: ...

    def off(self) -> None: ...


class Motor:
    def __init__(self, pins: MotorPins, en_pwm: _PWM, in1: _DOut, in2: _DOut):
        self.pins = pins
        self._en = en_pwm
        self._in1 = in1
        self._in2 = in2
        self._last_speed = 0.0

    def set(self, speed: float, *, coast_on_stop: bool = True) -> None:
        """Set motor speed in range [-1..1]."""
        s = max(-1.0, min(1.0, float(speed)))
        if self.pins.invert:
            s = -s

        self._last_speed = s

        if abs(s) < 1e-6:
            if coast_on_stop:
                self._in1.off()
                self._in2.off()
            else:
                self._in1.on()
                self._in2.on()
            self._en.change(0.0)
            return

        if s > 0:
            self._in1.on()
            self._in2.off()
        else:
            self._in1.off()
            self._in2.on()

        self._en.change(abs(s))

    def stop(self) -> None:
        self.set(0.0)

    @property
    def last_speed(self) -> float:
        return float(self._last_speed)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            try:
                self._en.stop()
            except Exception:
                pass


@dataclass
class DriveConfig:
    pwm_frequency_hz: int = 200
    steer_gain: float = 1.00
    allow_pivot_turn: bool = True
    pivot_speed: float = 0.85

    max_fwd_speed: float = 1.0
    max_rev_speed: float = 1.0

    fwd_accel_per_s: float = 1.0
    rev_accel_per_s: float = 1.0
    decel_per_s: float = 2.5

    coast_on_stop: bool = True


def load_motor_pins() -> Tuple[List[MotorPins], Tuple[str, str], Tuple[str, str]]:
    motors: List[MotorPins] = []
    for name, pins in robot_config.MOTORS.items():
        invert = not bool(robot_config.DIR_CORRECT.get(name, True))
        motors.append(
            MotorPins(
                name=name,
                en=int(pins["en"]),
                in1=int(pins["in1"]),
                in2=int(pins["in2"]),
                invert=invert,
            )
        )

    # Skid-steer grouping for this robot.
    # Keep this consistent with the intended turning logic for this robot.
    left = ("lf", "rr")
    right = ("rf", "lr")
    return motors, left, right


def build_motors(
    motors: Sequence[MotorPins],
    *,
    dry_run: bool,
    pwm_frequency_hz: int = 200,
    dry_run_output: bool = True,
):
    """Build Motor objects and return (motor_map, stop_all, cleanup)."""

    if dry_run:
        motor_map: Dict[str, Motor] = {}

        class _DryPWM:
            def __init__(self, label: str):
                self._label = label
                self._last: float | None = None

            def change(self, duty_0_to_1: float) -> None:
                duty = max(0.0, min(1.0, float(duty_0_to_1)))
                if dry_run_output and duty != self._last:
                    print(f"[{self._label}] EN duty={duty:.2f}")
                    self._last = duty

            def stop(self) -> None:
                return

        class _DryDOut:
            def __init__(self, label: str):
                self._label = label
                self._state = 0

            def on(self) -> None:
                if self._state != 1:
                    if dry_run_output:
                        print(f"[{self._label}] ON")
                    self._state = 1

            def off(self) -> None:
                if self._state != 0:
                    if dry_run_output:
                        print(f"[{self._label}] OFF")
                    self._state = 0

        for m in motors:
            motor_map[m.name] = Motor(
                m,
                _DryPWM(f"{m.name}/EN({m.en})"),
                _DryDOut(f"{m.name}/IN1({m.in1})"),
                _DryDOut(f"{m.name}/IN2({m.in2})"),
            )

        def stop_all() -> None:
            for mot in motor_map.values():
                mot.stop()

        def cleanup() -> None:
            for mot in motor_map.values():
                mot.close()

        return motor_map, stop_all, cleanup

    try:
        import RPi.GPIO as GPIO  # type: ignore
    except Exception as e:
        raise RuntimeError("RPi.GPIO is required unless --dry-run is set.") from e

    GPIO.setwarnings(False)
    # User requested: these numbers are GPIO pins on the real Pi.
    # Don't auto-detect; use BCM and let invalid pins fail.
    GPIO.setmode(GPIO.BCM)

    class _GPIOPWM:
        def __init__(self, pin: int):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._pwm = GPIO.PWM(pin, pwm_frequency_hz)
            self._pwm.start(0.0)

        def change(self, duty_0_to_1: float) -> None:
            duty = max(0.0, min(1.0, float(duty_0_to_1))) * 100.0
            self._pwm.ChangeDutyCycle(duty)

        def stop(self) -> None:
            try:
                self._pwm.ChangeDutyCycle(0.0)
            finally:
                self._pwm.stop()

    class _GPIODOut:
        def __init__(self, pin: int):
            self._pin = pin
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        def on(self) -> None:
            GPIO.output(self._pin, GPIO.HIGH)

        def off(self) -> None:
            GPIO.output(self._pin, GPIO.LOW)

    motor_map: Dict[str, Motor] = {}
    for m in motors:
        motor_map[m.name] = Motor(m, _GPIOPWM(m.en), _GPIODOut(m.in1), _GPIODOut(m.in2))

    def stop_all() -> None:
        for mot in motor_map.values():
            mot.stop()

    def cleanup() -> None:
        try:
            for mot in motor_map.values():
                mot.close()
        finally:
            GPIO.cleanup()

    return motor_map, stop_all, cleanup


def _ramp(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(target, current + max_delta)
    if current > target:
        return max(target, current - max_delta)
    return current


def _ramp_signed(current: float, target: float, dt_s: float, cfg: DriveConfig) -> float:
    same_dir = (current == 0.0) or (target == 0.0) or ((current > 0.0) == (target > 0.0))

    if not same_dir:
        return _ramp(current, target, cfg.decel_per_s * dt_s)

    if abs(target) > abs(current):
        accel = cfg.fwd_accel_per_s if target > 0.0 else cfg.rev_accel_per_s
        return _ramp(current, target, accel * dt_s)

    return _ramp(current, target, cfg.decel_per_s * dt_s)


def mix_throttle_steer(
    throttle: float,
    steer: float,
    *,
    speed_setting: float,
    cfg: DriveConfig,
    full_speed: bool,
) -> Tuple[float, float]:
    throttle = max(-1.0, min(1.0, float(throttle)))
    steer = max(-1.0, min(1.0, float(steer)))
    # Hardware orientation is mirrored on this robot; invert steering globally.
    steer = -steer
    speed_setting = max(0.0, min(1.0, float(speed_setting)))

    if full_speed:
        if throttle == 0.0:
            if cfg.allow_pivot_turn and steer != 0.0:
                return (-1.0 if steer > 0 else 1.0, 1.0 if steer > 0 else -1.0)
            return (0.0, 0.0)

        if steer < 0.0:
            return (0.0, 1.0 if throttle > 0 else -1.0)
        if steer > 0.0:
            return (1.0 if throttle > 0 else -1.0, 0.0)
        return (1.0 if throttle > 0 else -1.0, 1.0 if throttle > 0 else -1.0)

    if throttle == 0.0 and (steer == 0.0 or not cfg.allow_pivot_turn):
        return (0.0, 0.0)

    if throttle == 0.0 and cfg.allow_pivot_turn:
        base = max(0.0, min(1.0, cfg.pivot_speed)) * speed_setting
        left_cmd = steer * base
        right_cmd = -steer * base
        return (max(-1.0, min(1.0, left_cmd)), max(-1.0, min(1.0, right_cmd)))

    base = (cfg.max_fwd_speed if throttle > 0.0 else cfg.max_rev_speed) * speed_setting
    left_cmd = (throttle + (steer * cfg.steer_gain)) * base
    right_cmd = (throttle - (steer * cfg.steer_gain)) * base

    # Car-like behavior while moving: don't let steering flip one side into reverse.
    # (Pivot turns are handled above when throttle == 0.)
    if throttle > 0.0:
        left_cmd = max(0.0, left_cmd)
        right_cmd = max(0.0, right_cmd)
    elif throttle < 0.0:
        left_cmd = min(0.0, left_cmd)
        right_cmd = min(0.0, right_cmd)
    return (max(-1.0, min(1.0, left_cmd)), max(-1.0, min(1.0, right_cmd)))


class SkidSteerDrive:
    def __init__(
        self,
        motor_map: Dict[str, Motor],
        *,
        left_names: Sequence[str],
        right_names: Sequence[str],
        cfg: DriveConfig,
        full_speed: bool,
    ):
        self._motors = motor_map
        self._left = [motor_map[n] for n in left_names]
        self._right = [motor_map[n] for n in right_names]
        self._cfg = cfg
        self._full_speed = full_speed

        self._current_left = 0.0
        self._current_right = 0.0

    def emergency_stop(self) -> None:
        self._current_left = 0.0
        self._current_right = 0.0
        for m in self._left + self._right:
            m.stop()

    def update(self, *, target_left: float, target_right: float, dt_s: float) -> None:
        if self._full_speed:
            self._current_left = target_left
            self._current_right = target_right
        else:
            self._current_left = _ramp_signed(self._current_left, target_left, dt_s, self._cfg)
            self._current_right = _ramp_signed(self._current_right, target_right, dt_s, self._cfg)

        for m in self._left:
            m.set(self._current_left, coast_on_stop=self._cfg.coast_on_stop)
        for m in self._right:
            m.set(self._current_right, coast_on_stop=self._cfg.coast_on_stop)


def run_motor_test(
    motors: Sequence[Motor],
    *,
    peak_speed: float,
    on_time_s: float,
    off_time_s: float,
    step_s: float = 0.02,
) -> int:
    print("Starting motor test (ramp up/down forward + reverse per motor)...")

    def ramp_run(m: Motor, s: float, total_s: float) -> None:
        total_s = max(0.0, float(total_s))
        if total_s <= 0.0 or abs(s) < 1e-6:
            m.stop()
            return

        up_s = total_s / 2.0
        down_s = total_s - up_s

        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= up_s:
                break
            frac = 0.0 if up_s <= 1e-9 else (t / up_s)
            m.set(s * frac)
            time.sleep(step_s)

        m.set(s)

        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= down_s:
                break
            frac = 0.0 if down_s <= 1e-9 else (1.0 - (t / down_s))
            m.set(s * frac)
            time.sleep(step_s)

        m.stop()

    try:
        for m in motors:
            print(f"\nTesting {m.pins.name}...")
            print("  FORWARD")
            ramp_run(m, +abs(peak_speed), on_time_s)
            time.sleep(off_time_s)
            print("  REVERSE")
            ramp_run(m, -abs(peak_speed), on_time_s)
            time.sleep(off_time_s)
        print("\nDone.")
        return 0
    except KeyboardInterrupt:
        for m in motors:
            m.stop()
        print("\nInterrupted. Motors stopped.")
        return 130
