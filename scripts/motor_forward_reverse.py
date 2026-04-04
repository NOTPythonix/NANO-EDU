"""Standalone forward/reverse motor test.

This file is self-contained: copy it onto the Pi, adjust the pin constants if
needed, and run it directly. Use ``--dry-run`` on a non-Pi machine.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence


# BCM GPIO pin numbers copied from the robot config.
MOTORS = {
    "lf": {"in1": 5, "in2": 6, "en": 12},
    "rr": {"in1": 16, "in2": 20, "en": 13},
    "rf": {"in1": 17, "in2": 27, "en": 18},
    "lr": {"in1": 22, "in2": 23, "en": 19},
}

# True means forward is IN1=HIGH, IN2=LOW for that motor.
DIR_CORRECT = {
    "lf": True,
    "lr": True,
    "rf": True,
    "rr": True,
}


@dataclass(frozen=True)
class MotorPins:
    name: str
    en: int
    in1: int
    in2: int
    invert: bool = False


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

    def set(self, speed: float) -> None:
        speed = max(-1.0, min(1.0, float(speed)))
        if self.pins.invert:
            speed = -speed

        if abs(speed) < 1e-6:
            self._in1.off()
            self._in2.off()
            self._en.change(0.0)
            return

        if speed > 0:
            self._in1.on()
            self._in2.off()
        else:
            self._in1.off()
            self._in2.on()

        self._en.change(abs(speed))

    def stop(self) -> None:
        self.set(0.0)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            try:
                self._en.stop()
            except Exception:
                pass


def load_motor_pins() -> List[MotorPins]:
    motors: List[MotorPins] = []
    for name, pins in MOTORS.items():
        motors.append(
            MotorPins(
                name=name,
                en=int(pins["en"]),
                in1=int(pins["in1"]),
                in2=int(pins["in2"]),
                invert=not bool(DIR_CORRECT.get(name, True)),
            )
        )
    return motors


def build_motors(
    motors: Sequence[MotorPins],
    *,
    dry_run: bool,
    pwm_frequency_hz: int = 200,
):
    if dry_run:
        motor_map: Dict[str, Motor] = {}

        class _DryPWM:
            def __init__(self, label: str):
                self._label = label
                self._last: float | None = None

            def change(self, duty_0_to_1: float) -> None:
                duty = max(0.0, min(1.0, float(duty_0_to_1)))
                if duty != self._last:
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
                    print(f"[{self._label}] ON")
                    self._state = 1

            def off(self) -> None:
                if self._state != 0:
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
    except Exception as exc:
        raise RuntimeError("RPi.GPIO is required unless --dry-run is set.") from exc

    GPIO.setwarnings(False)
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


def run_motor_test(
    motors: Sequence[Motor],
    *,
    peak_speed: float,
    on_time_s: float,
    off_time_s: float,
    step_s: float = 0.02,
) -> int:
    print("Starting motor test (ramp up/down forward + reverse per motor)...")

    def ramp_run(motor: Motor, speed: float, total_s: float) -> None:
        total_s = max(0.0, float(total_s))
        if total_s <= 0.0 or abs(speed) < 1e-6:
            motor.stop()
            return

        up_s = total_s / 2.0
        down_s = total_s - up_s

        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= up_s:
                break
            frac = 0.0 if up_s <= 1e-9 else (elapsed / up_s)
            motor.set(speed * frac)
            time.sleep(step_s)

        motor.set(speed)

        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= down_s:
                break
            frac = 0.0 if down_s <= 1e-9 else (1.0 - (elapsed / down_s))
            motor.set(speed * frac)
            time.sleep(step_s)

        motor.stop()

    try:
        for motor in motors:
            print(f"\nTesting {motor.pins.name}...")
            print("  FORWARD")
            ramp_run(motor, +abs(peak_speed), on_time_s)
            time.sleep(off_time_s)
            print("  REVERSE")
            ramp_run(motor, -abs(peak_speed), on_time_s)
            time.sleep(off_time_s)
        print("\nDone.")
        return 0
    except KeyboardInterrupt:
        for motor in motors:
            motor.stop()
        print("\nInterrupted. Motors stopped.")
        return 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each robot motor forward and reverse using the configured GPIO pins."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the GPIO actions instead of touching real hardware",
    )
    parser.add_argument(
        "--peak-speed",
        type=float,
        default=0.65,
        help="peak motor duty cycle in the range 0..1",
    )
    parser.add_argument(
        "--on-time",
        type=float,
        default=1.0,
        help="seconds to spend ramping up and down for each direction",
    )
    parser.add_argument(
        "--off-time",
        type=float,
        default=0.5,
        help="pause between forward and reverse steps",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    motors_cfg = load_motor_pins()
    motor_map, _, cleanup = build_motors(
        motors_cfg,
        dry_run=bool(args.dry_run),
    )

    try:
        return run_motor_test(
            list(motor_map.values()),
            peak_speed=max(0.0, min(1.0, float(args.peak_speed))),
            on_time_s=max(0.0, float(args.on_time)),
            off_time_s=max(0.0, float(args.off_time)),
        )
    finally:
        try:
            cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())