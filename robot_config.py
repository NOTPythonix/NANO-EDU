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
    "lf": {"in1": 5, "in2": 6, "en": 12},
    "rr": {"in1": 16, "in2": 20, "en": 13},
    "rf": {"in1": 17, "in2": 27, "en": 18},
    "lr": {"in1": 22, "in2": 23, "en": 19},
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
