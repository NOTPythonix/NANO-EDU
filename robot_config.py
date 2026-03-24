"""Robot configuration (current source of truth).

This module contains the Raspberry Pi BCM GPIO pin mappings and per-motor
direction correction flags used by the refactored controller.

It intentionally contains *only* constants so it stays importable on Windows
and other non-Pi machines.
"""

from __future__ import annotations

# --- Motor pins (BCM GPIO) ---
# User confirmed these are BCM GPIO numbers on the real Pi.
# If any are wrong, the GPIO backend should fail fast rather than auto-detect.
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
