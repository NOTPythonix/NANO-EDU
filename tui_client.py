from __future__ import annotations

import os
import time
from typing import Optional

from tui import (
    FeatureFlags,
    _header,
    _select_checklist,
    _select_one,
    NetworkConfig,
    run_live_dashboard_tui,
    run_motor_test_tui,
)


def _require_rich() -> bool:
    try:
        from rich.console import Console  # noqa: F401
        from rich.panel import Panel  # noqa: F401
        from rich.prompt import IntPrompt  # noqa: F401

        return True
    except Exception:
        return False


if not _require_rich():
    raise SystemExit(
        "This TUI requires the 'rich' package.\n\n"
        "Install it with:\n"
        "  pip install rich\n"
    )

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt


def main() -> int:
    console = Console()
    _header(console)

    # Configure peak up-front (before run/operation selection).
    console.print(Panel("Set session peak power (applies to live and test modes).", border_style="bright_blue"))
    peak_pct = IntPrompt.ask("Peak power %", default=65)
    peak = max(0.0, min(1.0, float(peak_pct) / 100.0))

    mode = _select_one(
        console,
        "Run Mode",
        [
            ("dry", "Run without GPIO (safe on Windows)"),
            ("real", "Run on Raspberry Pi GPIO"),
        ],
    )
    dry_run = mode == "dry"

    _header(console)
    action = _select_one(
        console,
        "Operation",
        [
            ("live", "Drive the robot with a live dashboard"),
            ("test", "Run motor test with progress bars"),
        ],
    )

    if action == "test":
        cycles_choice = _select_one(
            console,
            "Cycles Per Motor",
            [
                ("1", "Forward+reverse once"),
                ("2", "Forward+reverse twice"),
                ("3", "Forward+reverse three times"),
                ("4", "Forward+reverse four times"),
                ("5", "Forward+reverse five times"),
            ],
        )
        return run_motor_test_tui(dry_run=dry_run, peak=peak, cycles_per_motor=int(cycles_choice))

    # Live mode
    _header(console)
    console.print(Panel("Motor control is required. Choose optional features:", border_style="bright_blue"))

    picked = _select_checklist(
        console,
        "Optional Features",
        [
            ("voice", "Voice", "Voice control (Vosk + sounddevice)"),
            ("camera", "Camera", "Obstacle detection (OpenCV)"),
        ],
        default_checked=set(),
    )

    features = FeatureFlags(
        motor=True,
        voice="voice" in picked,
        camera="camera" in picked,
        ir=False,
    )

    # Optional network link (client can proceed even if disconnected)
    _header(console)
    console.print(Panel("Optional: connect to a server-side TUI for remote control/inference.", border_style="bright_blue"))
    net_choice = _select_one(
        console,
        "Network Link",
        [
            ("off", "No network (local-only)"),
            ("on", "Connect to server (optional on client)"),
        ],
    )

    net_cfg: Optional[NetworkConfig] = None
    if net_choice == "on":
        default_host = os.environ.get("ROBOT_SERVER_HOST", "127.0.0.1")
        default_port = int(os.environ.get("ROBOT_SERVER_PORT", "8765"))
        host = default_host
        try:
            host = console.input(f"Server host [{default_host}]: ").strip() or default_host
        except Exception:
            host = default_host
        port = IntPrompt.ask("Server port", default=default_port)
        net_cfg = NetworkConfig(enabled=True, host=host, port=port)

    mic_index: Optional[int] = None
    if features.voice:
        ans = _select_one(
            console,
            "Microphone",
            [
                ("auto", "Use default microphone device"),
                ("index", "Specify a microphone device index"),
            ],
        )
        if ans == "index":
            mic_index = IntPrompt.ask("Mic index", default=0)

    _header(console)
    console.print(Panel("Entering dashboard...\n[dim]Press Q to quit.[/]", border_style="bright_green"))
    time.sleep(0.6)

    return run_live_dashboard_tui(dry_run=dry_run, features=features, mic_index=mic_index, peak=peak, net=net_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
