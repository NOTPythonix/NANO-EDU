from __future__ import annotations

import sys
import queue
import threading
import time
import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from camera_control import CameraConfig, CameraController
from ir_control import IRConfig, IRController
from movement import DriveConfig, Motor, SkidSteerDrive, build_motors, load_motor_pins, mix_throttle_steer
from robot_net import JsonLineLink
from ui_layout import allocate_round_robin_heights
from voice_control import VoiceConfig, VoiceController


def _require_rich():
    try:
        from rich import box  # noqa: F401
        from rich.align import Align  # noqa: F401
        from rich.console import Console  # noqa: F401
        from rich.layout import Layout  # noqa: F401
        from rich.live import Live  # noqa: F401
        from rich.panel import Panel  # noqa: F401
        from rich.prompt import IntPrompt  # noqa: F401
        from rich.table import Table  # noqa: F401

        return True
    except Exception:
        return False


if not _require_rich():
    raise SystemExit(
        "This TUI requires the 'rich' package.\n\n"
        "Install it with:\n"
        "  pip install rich\n"
    )

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table


ConsoleCmd = Tuple[str, Optional[int]]


def _motor_label(short: str) -> str:
    return {
        "lf": "Left Front",
        "lr": "Left Rear",
        "rf": "Right Front",
        "rr": "Right Rear",
    }.get(short, short)


def _ellipsize(s: str, max_len: int) -> str:
    s = str(s)
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _drain_stdin() -> None:
    """Best-effort drain of buffered keypresses.

    We use pynput for menus (global hook), but we don't consume stdin. Arrow keys / Enter
    can remain buffered and then get interpreted by the shell after we exit (e.g. PowerShell
    history navigation + accidental command execution). Draining stdin prevents that.
    """

    try:
        if sys.platform.startswith("win"):
            import msvcrt  # type: ignore

            while msvcrt.kbhit():
                try:
                    msvcrt.getwch()
                except Exception:
                    break
            return
    except Exception:
        pass

    # POSIX best-effort
    try:
        import select

        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            sys.stdin.read(1)
    except Exception:
        pass


@dataclass
class FeatureFlags:
    motor: bool = True
    voice: bool = False
    camera: bool = False
    ir: bool = False


@dataclass
class NetworkConfig:
    enabled: bool = False
    host: str = ""
    port: int = 8765


@dataclass
class RuntimeState:
    speed_setting: float = 0.65
    max_speed_setting: float = 1.0
    autonomous: bool = False
    manual_throttle: float = 0.0
    manual_steer: float = 0.0


class PynputKeys:
    def __init__(self):
        try:
            from pynput import keyboard as pynput_keyboard  # type: ignore
        except Exception as e:
            raise RuntimeError("pynput is required for live mode. Install with: pip install pynput") from e

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
            if key == kb.Key.up:
                return "up"
            if key == kb.Key.down:
                return "down"
            if key == kb.Key.left:
                return "left"
            if key == kb.Key.right:
                return "right"
            if key == kb.Key.enter:
                return "enter"
            if key == kb.Key.esc:
                return "esc"
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

        listener_kwargs: Dict[str, Any] = {"on_press": on_press, "on_release": on_release}
        if sys.platform.startswith("linux"):
            # On Linux terminals, arrow keys can also be interpreted by the TTY,
            # causing viewport jitter while the TUI is running.
            listener_kwargs["suppress"] = True

        try:
            self._listener = kb.Listener(**listener_kwargs)
        except TypeError:
            listener_kwargs.pop("suppress", None)
            self._listener = kb.Listener(**listener_kwargs)
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


def _apply_external_command(state: RuntimeState, cmd: ConsoleCmd) -> None:
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
        state.speed_setting = max(0.0, min(state.max_speed_setting, float(val) / 100.0))
        return


def _try_create_menu_keys() -> Optional[PynputKeys]:
    try:
        k = PynputKeys()
        k.start()
        return k
    except Exception:
        return None


def _render_menu(title: str, options: list[tuple[str, str]], selected: int) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("sel", width=2)
    t.add_column("Option", style="bold")
    t.add_column("Description", style="dim")

    for i, (key, desc) in enumerate(options):
        is_sel = i == selected
        marker = "➤" if is_sel else " "
        opt_style = "reverse bold" if is_sel else ""
        t.add_row(marker, f"[{opt_style}]{key}[/]" if opt_style else key, desc)

    help_line = "[dim]↑/↓ to move, Enter to select, Q/Esc to quit[/]"
    return Panel(t, title=title, border_style="bright_blue", subtitle=help_line)


def _select_one(console: Console, title: str, options: list[tuple[str, str]]) -> str:
    # Arrow-key selection via pynput (preferred). If pynput isn't available, fall back.
    keys = _try_create_menu_keys()
    if keys is None:
        table = Table(title=title, box=box.SIMPLE_HEAVY)
        table.add_column("#", style="bold cyan", justify="right")
        table.add_column("Option", style="bold")
        table.add_column("Description", style="dim")
        for i, (key, desc) in enumerate(options, start=1):
            table.add_row(str(i), key, desc)
        console.print(Panel(table, border_style="bright_blue"))
        while True:
            choice = IntPrompt.ask("Select", default=1)
            if 1 <= choice <= len(options):
                return options[choice - 1][0]

    selected = 0
    try:
        console.clear()
        with Live(_render_menu(title, options, selected), console=console, refresh_per_second=30, screen=True) as live:
            while True:
                ev = keys.poll_event()
                if ev is None:
                    time.sleep(0.01)
                    continue
                kind, k = ev
                if kind != "down":
                    continue
                if k in ("q", "esc"):
                    raise SystemExit(130)
                if k == "up":
                    selected = (selected - 1) % len(options)
                elif k == "down":
                    selected = (selected + 1) % len(options)
                elif k == "enter":
                    return options[selected][0]
                live.update(_render_menu(title, options, selected))
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        try:
            keys.stop()
        except Exception:
            pass
        _drain_stdin()


def _render_checklist(title: str, options: list[tuple[str, str, str]], selected: int, checked: set[str]) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("sel", width=2)
    t.add_column("box", width=3)
    t.add_column("Option", style="bold")
    t.add_column("Description", style="dim")

    for i, (key, label, desc) in enumerate(options):
        is_sel = i == selected
        marker = "➤" if is_sel else " "
        box_txt = "[x]" if key in checked else "[ ]"
        opt_style = "reverse bold" if is_sel else ""
        shown_label = f"[{opt_style}]{label}[/]" if opt_style else label
        t.add_row(marker, box_txt, shown_label, desc)

    help_line = "[dim]↑/↓ move, Space toggle, Enter confirm, Q/Esc cancel[/]"
    return Panel(t, title=title, border_style="bright_blue", subtitle=help_line)


def _select_checklist(
    console: Console,
    title: str,
    options: list[tuple[str, str, str]],
    *,
    default_checked: Optional[set[str]] = None,
) -> set[str]:
    keys = _try_create_menu_keys()
    if keys is None:
        # Fallback: no checkbox UI; return nothing selected.
        return set(default_checked or set())

    checked: set[str] = set(default_checked or set())
    selected = 0

    try:
        console.clear()
        with Live(_render_checklist(title, options, selected, checked), console=console, refresh_per_second=30, screen=True) as live:
            while True:
                ev = keys.poll_event()
                if ev is None:
                    time.sleep(0.01)
                    continue
                kind, k = ev
                if kind != "down":
                    continue
                if k in ("q", "esc"):
                    return set(default_checked or set())
                if k == "up":
                    selected = (selected - 1) % len(options)
                elif k == "down":
                    selected = (selected + 1) % len(options)
                elif k == "space":
                    key = options[selected][0]
                    if key in checked:
                        checked.remove(key)
                    else:
                        checked.add(key)
                elif k == "enter":
                    return checked
                live.update(_render_checklist(title, options, selected, checked))
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        try:
            keys.stop()
        except Exception:
            pass
        _drain_stdin()


def _header(console: Console) -> None:
    console.clear()
    console.print(
        Align.center(
            "[bold bright_cyan]HSEF Robot Control[/]  [dim]TUI[/]",
            vertical="middle",
        )
    )


def run_motor_test_tui(*, dry_run: bool, peak: float, cycles_per_motor: int) -> int:
    console = Console()
    _header(console)

    motors_cfg, _left_names, _right_names = load_motor_pins()
    motor_map, _stop_all, cleanup = build_motors(motors_cfg, dry_run=dry_run, dry_run_output=False)

    # Stable order: lf, lr, rf, rr if present.
    preferred = ["lf", "lr", "rf", "rr"]
    motors: list[Motor] = [motor_map[n] for n in preferred if n in motor_map]
    # Any extras
    for name, m in motor_map.items():
        if name not in preferred:
            motors.append(m)

    peak = max(0.0, min(1.0, float(peak)))
    cycles_per_motor = max(1, int(cycles_per_motor))
    on_time_s = 1.0
    off_time_s = 0.5
    step_s = 0.02

    # Shared block style with the dashboard for consistency.
    def bar_fill(power_0_to_1: float, *, sign: float) -> str:
        mag = max(0.0, min(1.0, float(power_0_to_1)))
        blocks = int(round(mag * 10))
        fill = "█" * blocks
        empty = "░" * (10 - blocks)
        color = "green" if sign > 0 else ("red" if sign < 0 else "white")
        return f"[{color}]{fill}{empty}[/]"

    def dir_of(s: float) -> str:
        if abs(s) < 1e-3:
            return "STOP"
        return "FWD" if s > 0 else "REV"

    motor_state: Dict[str, Dict[str, object]] = {}
    invert_map: Dict[str, bool] = {}
    for m in motors:
        motor_state[m.pins.name] = {"phase": "idle", "cmd": 0.0}
        invert_map[m.pins.name] = bool(m.pins.invert)

    def set_state(m: Motor, phase: str) -> None:
        # Pad phase to a fixed width so the bar column doesn't resize.
        motor_state[m.pins.name]["phase"] = f"{phase:<12}"[:12]
        motor_state[m.pins.name]["cmd"] = float(m.last_speed)

    def render_table() -> Table:
        t = Table(box=box.SIMPLE, expand=True)
        t.add_column("Motor", style="bold cyan", width=18)
        t.add_column("Phase", style="magenta", width=12)
        t.add_column("Inv", justify="center", width=3)
        t.add_column("Dir", justify="center", width=4)
        t.add_column("Power", justify="left", width=12)
        t.add_column("%", justify="right", width=4)

        for name in [m.pins.name for m in motors]:
            st = motor_state[name]
            cmd = float(st["cmd"])
            p = max(0.0, min(1.0, abs(cmd)))
            t.add_row(
                f"{_motor_label(name)} ({name})",
                str(st["phase"]),
                "Y" if invert_map.get(name, False) else "N",
                dir_of(cmd),
                bar_fill(p, sign=cmd),
                f"{(p * 100.0):>3.0f}",
            )
        return t

    def ramp_run(m: Motor, signed_peak: float, total_s: float, update_ui) -> None:
        total_s = max(0.0, float(total_s))
        up_s = total_s / 2.0
        down_s = total_s - up_s

        # Ramp up
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= up_s:
                break
            frac = 0.0 if up_s <= 1e-9 else (t / up_s)
            m.set(signed_peak * frac)
            set_state(m, "ramp up")
            update_ui()
            time.sleep(step_s)

        m.set(signed_peak)
        set_state(m, "hold")
        update_ui()

        # Ramp down
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= down_s:
                break
            frac = 0.0 if down_s <= 1e-9 else (1.0 - (t / down_s))
            m.set(signed_peak * frac)
            set_state(m, "ramp down")
            update_ui()
            time.sleep(step_s)

        m.stop()
        set_state(m, "idle")
        update_ui()

    console.print(
        Panel(
            "[bold]Motor Test[/]\n"
            f"Each motor runs forward then reverse with a smooth ramp up/down.\n"
            f"[dim]Peak: {int(round(peak * 100))}%   Cycles/motor: {cycles_per_motor}[/]\n\n"
            "[dim]Press Ctrl+C to abort safely.[/]",
            border_style="bright_green",
        )
    )

    try:
        def panel() -> Panel:
            return Panel(render_table(), title="Motor Test", border_style="bright_green")

        with Live(panel(), console=console, refresh_per_second=20) as live:
            def update_ui() -> None:
                live.update(panel())

            for m in motors:
                set_state(m, "idle")
                update_ui()
                for _ in range(cycles_per_motor):
                    ramp_run(m, +peak, on_time_s, update_ui)
                    time.sleep(off_time_s)
                    ramp_run(m, -peak, on_time_s, update_ui)
                    time.sleep(off_time_s)
        console.print(Panel("[bold bright_green]Done.[/]", border_style="bright_green"))
        return 0
    except KeyboardInterrupt:
        console.print(Panel("[bold yellow]Interrupted. Motors stopped.[/]", border_style="yellow"))
        return 130
    finally:
        try:
            cleanup()
        except Exception:
            pass


def _motor_table(motor_map: Dict[str, Motor]) -> Table:
    # Use available width efficiently. We keep Dir padded to a stable 4 chars so
    # the bar doesn't shift when switching between STOP/FWD/REV.
    t = Table(box=box.SIMPLE, expand=True, pad_edge=False, padding=(0, 1))
    t.add_column("Motor", style="bold cyan", no_wrap=True, overflow="ellipsis")
    t.add_column("Cmd", justify="right", no_wrap=True)
    t.add_column("Inv", justify="center", no_wrap=True)
    t.add_column("Dir", justify="center", no_wrap=True, width=4)
    t.add_column("Power", justify="left", no_wrap=True, width=12)

    def dir_of(s: float) -> str:
        if abs(s) < 1e-3:
            return "STOP"
        return "FWD" if s > 0 else "REV"

    def bar(s: float) -> str:
        # 0..1 magnitude mapped to 10 blocks
        mag = max(0.0, min(1.0, abs(s)))
        blocks = int(round(mag * 10))
        fill = "█" * blocks
        empty = "░" * (10 - blocks)
        color = "green" if s > 0 else ("red" if s < 0 else "white")
        return f"[{color}]{fill}{empty}[/]"

    preferred = ["lf", "lr", "rf", "rr"]
    ordered = [n for n in preferred if n in motor_map] + [n for n in sorted(motor_map.keys()) if n not in preferred]
    for name in ordered:
        m = motor_map[name]
        s = m.last_speed
        d = dir_of(s)
        # Pad to stable width inside the fixed-width column.
        d = f"{d:<4}"[:4]
        t.add_row(f"{_motor_label(name)} ({name})", f"{s:+.2f}", "Y" if m.pins.invert else "N", d, bar(s))

    return t


def run_live_dashboard_tui(
    *,
    dry_run: bool,
    features: FeatureFlags,
    mic_index: Optional[int],
    peak: float,
    net: Optional[NetworkConfig] = None,
) -> int:
    console = Console()
    _header(console)

    motors_cfg, left_names, right_names = load_motor_pins()
    motor_map, stop_all, cleanup = build_motors(motors_cfg, dry_run=dry_run, dry_run_output=False)

    cfg = DriveConfig()
    drive = SkidSteerDrive(
        motor_map,
        left_names=left_names,
        right_names=right_names,
        cfg=cfg,
        full_speed=False,
    )

    keys = PynputKeys()
    keys.start()

    state = RuntimeState(speed_setting=max(0.0, min(1.0, float(peak))), max_speed_setting=max(0.0, min(1.0, float(peak))))
    voice_q: "queue.Queue[ConsoleCmd]" = queue.Queue()
    ir_q: "queue.Queue[ConsoleCmd]" = queue.Queue()

    vc = VoiceController(voice_q, VoiceConfig(mic_device_index=mic_index)) if features.voice else None
    if vc and vc.available:
        vc.start()

    irc = IRController(IRConfig()) if features.ir else None
    cam = CameraController(CameraConfig(camera_index=0, width=320, height=240, jpeg_quality=55)) if features.camera else None

    link: Optional[JsonLineLink] = None
    if net and net.enabled and net.host:
        link = JsonLineLink(host=net.host, port=net.port, role="client", name="robot")
        link.start()

    last_remote_cmd_ts: Optional[float] = None
    remote_state: dict[str, Any] = {}
    last_raw_motors_ts: Optional[float] = None
    raw_motors: dict[str, Any] = {}

    last_telemetry_tx = 0.0
    last_frame_tx = 0.0
    frames_sent = 0

    raw_prev = False

    start_t = time.time()
    last_t = time.time()

    # Layout sizing (tuned for maximum usable vertical space).
    TOP_SIZE = 3
    BOTTOM_SIZE = 3
    STATUS_SIZE = 5

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=TOP_SIZE),
        Layout(name="main"),
        Layout(name="bottom", size=BOTTOM_SIZE),
    )
    # Balance panels so the motor table isn't squished.
    # Simple responsive ratios based on terminal width.
    try:
        w = int(getattr(console, "size").width)
    except Exception:
        w = 120

    if w >= 150:
        motors_ratio, right_ratio = 7, 5
    elif w >= 120:
        motors_ratio, right_ratio = 6, 5
    else:
        motors_ratio, right_ratio = 5, 5

    right_panel_order = ["status", "voice", "camera", "ir"]

    # Layout:
    # - Left column: Motors (short) + Autonomous (fills)
    # - Right column: Status + Voice/Camera/IR
    layout["main"].split_row(Layout(name="left", ratio=motors_ratio), Layout(name="right", ratio=right_ratio))

    try:
        term_h = int(getattr(console, "size").height)
    except Exception:
        term_h = 40

    main_h = max(1, term_h - TOP_SIZE - BOTTOM_SIZE)

    # Motors: give it more height (user request) and steal from Autonomous.
    # Keep Autonomous at least a few rows so it remains useful.
    motors_panel_h = max(10, min(18, main_h // 2))
    motors_panel_h = min(motors_panel_h, max(8, main_h - 6))

    right_heights = allocate_round_robin_heights(main_h, right_panel_order, min_panel_h=5)
    status_panel_h = right_heights["status"]
    voice_panel_h = right_heights["voice"]
    camera_panel_h = right_heights["camera"]
    ir_panel_h = right_heights["ir"]

    layout["left"].split_column(
        Layout(name="motors", size=motors_panel_h),
        Layout(name="auto"),
    )

    layout["right"].split_column(
        Layout(name="status", size=status_panel_h),
        Layout(name="debug_right"),
    )
    layout["debug_right"].split_column(
        Layout(name="voice", size=voice_panel_h),
        Layout(name="camera", size=camera_panel_h),
        Layout(name="ir", size=ir_panel_h),
    )

    def on_off(v: bool) -> str:
        return "[green]ON[/]" if v else "[red]OFF[/]"

    def age_s(ts: Optional[float]) -> str:
        if ts is None:
            return "—"
        return f"{int(time.time() - ts)}s"

    focus_order = ["status", "auto", "voice", "camera", "ir"]
    focused = "status"
    scroll: Dict[str, int] = {k: 0 for k in focus_order}

    def _focus_border(base: str, key: str) -> str:
        return "bright_yellow" if key == focused else base

    def _scrollable_panel(key: str, title: str, base_border: str, rows: list[tuple[str, str]]) -> Panel:
        # Render as a 2-column grid with no wrapping and ellipsis overflow.
        # This prevents Rich from wrapping values like "Age: ..." or "Msg: ..."
        # onto extra lines (which breaks both layout and scrolling).
        try:
            term_h = int(getattr(console, "size").height)
        except Exception:
            term_h = 40

        main_h = max(1, term_h - TOP_SIZE - BOTTOM_SIZE)

        # Estimate per-panel height based on current layout.
        # - Status is the fixed top-right panel.
        # - Autonomous lives under Motors in the left column.
        # - Voice/Camera/IR live under Status in the right column.
        if key == "status":
            panel_h = status_panel_h
        elif key == "auto":
            panel_h = max(3, main_h - motors_panel_h)
        elif key in ("voice", "camera", "ir"):
            if key == "voice":
                panel_h = int(voice_panel_h)
            elif key == "camera":
                panel_h = int(camera_panel_h)
            else:
                panel_h = int(ir_panel_h)
        else:
            panel_h = max(3, main_h // 2)

        # Panel content height ~= total height - top/bottom border.
        window = max(1, panel_h - 2)

        off = max(0, int(scroll.get(key, 0)))
        max_off = max(0, len(rows) - window)
        off = min(off, max_off)
        scroll[key] = off

        more_up = off > 0
        more_down = (off + window) < len(rows)
        view = rows[off : off + window]

        grid = Table.grid(expand=True, pad_edge=False)
        grid.add_column(style="bold", no_wrap=True)
        # Make the value column expand to the remaining width so it doesn't
        # change size when scrolling shows different rows.
        grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1, justify="right")
        for k, v in view:
            grid.add_row(str(k), str(v))

        if key == focused:
            up_ind = "↑" if more_up else " "
            dn_ind = "↓" if more_down else " "
            subtitle = f"[dim]←/→ select  ↑/↓ scroll  {up_ind}{dn_ind}[/]"
        else:
            subtitle = ""

        return Panel(grid, title=title, subtitle=subtitle, border_style=_focus_border(base_border, key))

    last_voice_cmd: Optional[ConsoleCmd] = None
    last_voice_ts: Optional[float] = None
    last_ir_code: Optional[str] = None
    last_ir_cmd: Optional[ConsoleCmd] = None
    last_ir_ts: Optional[float] = None
    # Camera is now streaming to the server; obstacle inference is server-side.
    last_obs: Optional[str] = None
    last_obs_ts: Optional[float] = None
    last_targets: Tuple[float, float] = (0.0, 0.0)
    last_mode: str = "manual"  # manual|autonomous

    def status_panel() -> Panel:
        runtime_s = int(time.time() - start_t)
        cap_pct = int(round(state.max_speed_setting * 100))

        # Actual current output (based on last commanded motor speeds), not the speed setting.
        try:
            out = max(abs(m.last_speed) for m in motor_map.values()) if motor_map else 0.0
        except Exception:
            out = 0.0
        out_pct = int(round(max(0.0, min(1.0, out)) * 100))

        net_state = "[dim]OFF[/]"
        round_trip = "—"
        receive_age = "—"
        peer = "—"
        if link is not None:
            st = link.stats
            now = time.time()
            net_state = "[green]CONNECTED[/]" if st.connected else "[red]DISCONNECTED[/]"
            round_trip = f"{float(st.rtt_ms):.1f} ms" if st.rtt_ms is not None else "—"
            receive_age = f"{int(now - st.last_rx_ts)}s" if st.last_rx_ts is not None else "—"
            peer = _ellipsize(st.peer or (f"{net.host}:{net.port}" if net else "—"), 28)

        rows = [
            ("Run", "DRY" if dry_run else "REAL"),
            ("Network", net_state),
            ("Round trip time", round_trip),
            ("Receive age", receive_age),
            ("Peer", peer),
            (f"Speed % (Cap {cap_pct}%):", f"{out_pct}%"),
            ("Uptime", f"{runtime_s}s"),
        ]

        return _scrollable_panel("status", "Status", "bright_blue", rows)

    def autonomous_panel() -> Panel:
        cam_ok = bool(cam and cam.available)
        net_ok = bool(link and link.stats.connected)
        rows = [
            ("Mode", on_off(state.autonomous)),
            ("Net", on_off(net_ok)),
            ("Camera", on_off(cam_ok)),
            ("Last server obs", last_obs or "—"),
            ("Obs age", age_s(last_obs_ts)),
            ("Decision src", last_mode),
            ("Manual input", f"thr={state.manual_throttle:+.1f} steer={state.manual_steer:+.1f}"),
            ("Targets", f"L={last_targets[0]:+.2f}  R={last_targets[1]:+.2f}"),
            ("Msg", message),
        ]
        base = "green" if state.autonomous else "red"
        return _scrollable_panel("auto", "Autonomous", base, rows)

    def voice_panel() -> Panel:
        enabled = bool(features.voice)
        available = bool(vc and vc.available)
        on = bool(enabled and available)

        last_text: Optional[str] = None
        last_cmd: Optional[ConsoleCmd] = None
        last_ts: Optional[float] = None
        if vc and hasattr(vc, "last_event"):
            try:
                last_text, last_cmd, last_ts = vc.last_event()  # type: ignore[attr-defined]
            except Exception:
                pass

        parsed = f"{last_cmd[0]}" if last_cmd else "—"
        applied = f"{last_voice_cmd[0]}" if last_voice_cmd else "—"
        rows = [
            ("Enabled", on_off(enabled)),
            ("Available", on_off(available)),
            ("Last words", last_text or "—"),
            ("Parsed cmd", parsed),
            ("Applied cmd", applied),
            ("Age", age_s(last_ts or last_voice_ts)),
        ]
        return _scrollable_panel("voice", "Voice", "green" if on else "red", rows)

    def ir_panel() -> Panel:
        enabled = bool(features.ir)
        available = bool(irc and irc.available)
        on = bool(enabled and available)

        applied = f"{last_ir_cmd[0]}" if last_ir_cmd else "—"
        rows = [
            ("Enabled", on_off(enabled)),
            ("Available", on_off(available)),
            ("Last code", last_ir_code or "—"),
            ("Applied cmd", applied),
            ("Age", age_s(last_ir_ts)),
        ]
        return _scrollable_panel("ir", "IR", "green" if on else "red", rows)

    def camera_panel() -> Panel:
        enabled = bool(features.camera)
        available = bool(cam and cam.available)
        on = bool(enabled and available)

        now = time.time()
        last_age = "—" if not last_frame_tx else f"{int(now - last_frame_tx)}s"
        rows = [
            ("Enabled", on_off(enabled)),
            ("Available", on_off(available)),
            ("Frames sent", str(frames_sent)),
            ("Last frame", last_age),
        ]
        return _scrollable_panel("camera", "Camera", "green" if on else "red", rows)

    def top_panel() -> Panel:
        return Panel(
            "[bold]W/A/S/D[/]=drive   [bold]Space[/]=stop   [bold]O[/]=toggle autonomous   "
            "[bold][[/]/[bold]][/]=speed cap   [bold]←/→[/]=select panel   [bold]↑/↓[/]=scroll   [bold]Q[/]=quit",
            border_style="bright_cyan",
        )

    def bottom_panel(msg: str) -> Panel:
        return Panel(msg, border_style="dim")

    message = "Ready."
    message_until: float = 0.0

    def set_message(msg: str, *, duration_s: float = 3.0) -> None:
        nonlocal message, message_until
        message = str(msg)
        message_until = (time.time() + float(duration_s)) if duration_s and duration_s > 0 else 0.0

    def recalc_layout_sizes() -> None:
        nonlocal main_h, motors_panel_h, status_panel_h, voice_panel_h, camera_panel_h, ir_panel_h

        try:
            term_h_now = int(getattr(console, "size").height)
        except Exception:
            term_h_now = 40

        main_h = max(1, term_h_now - TOP_SIZE - BOTTOM_SIZE)

        motors_panel_h = max(10, min(18, main_h // 2))
        motors_panel_h = min(motors_panel_h, max(8, main_h - 6))

        right_heights = allocate_round_robin_heights(main_h, right_panel_order, min_panel_h=5)
        status_panel_h = right_heights["status"]
        voice_panel_h = right_heights["voice"]
        camera_panel_h = right_heights["camera"]
        ir_panel_h = right_heights["ir"]

        layout["left"]["motors"].size = motors_panel_h
        layout["right"]["status"].size = status_panel_h
        layout["debug_right"]["voice"].size = voice_panel_h
        layout["debug_right"]["camera"].size = camera_panel_h
        layout["debug_right"]["ir"].size = ir_panel_h

    try:
        with Live(layout, console=console, refresh_per_second=20, screen=True):
            while True:
                now = time.time()
                recalc_layout_sizes()

                # Network RX
                if link is not None:
                    while True:
                        msg = link.poll()
                        if msg is None:
                            break
                        mtype = msg.get("type")
                        if mtype == "command":
                            st = msg.get("state")
                            if isinstance(st, dict):
                                remote_state = st
                                last_remote_cmd_ts = now
                        elif mtype == "raw_motors":
                            mm = msg.get("motors")
                            if isinstance(mm, dict):
                                raw_motors = mm
                                last_raw_motors_ts = now
                        elif mtype == "server_obs":
                            obs = msg.get("obs")
                            if isinstance(obs, str):
                                last_obs = obs
                                last_obs_ts = now

                # Remote control is considered active if we're connected and have a recent command.
                remote_active = bool(
                    link
                    and link.stats.connected
                    and last_remote_cmd_ts is not None
                    and (now - float(last_remote_cmd_ts)) <= 0.6
                )

                # Edge events
                while True:
                    ev = keys.poll_event()
                    if ev is None:
                        break
                    kind, k = ev
                    if kind == "down" and k == "o":
                        if remote_active:
                            set_message("Server is controlling; use server TUI")
                            continue
                        # In the new design, autonomous requires a server link.
                        if link is None or not link.stats.connected:
                            set_message("Autonomous requires server link")
                        else:
                            state.autonomous = not state.autonomous
                            if state.autonomous:
                                state.manual_throttle = 0.0
                                state.manual_steer = 0.0
                            set_message(f"Autonomous: {'ON' if state.autonomous else 'OFF'}")
                    elif kind == "down" and k == "[":
                        if remote_active:
                            set_message("Server is controlling; use server TUI")
                            continue
                        state.max_speed_setting = max(0.0, state.max_speed_setting - 0.05)
                        state.speed_setting = min(state.speed_setting, state.max_speed_setting)
                    elif kind == "down" and k == "]":
                        if remote_active:
                            set_message("Server is controlling; use server TUI")
                            continue
                        state.max_speed_setting = min(1.0, state.max_speed_setting + 0.05)
                        state.speed_setting = min(state.max_speed_setting, state.speed_setting + 0.05)
                    elif kind == "down" and k in ("left", "right"):
                        idx = focus_order.index(focused)
                        if k == "left":
                            focused = focus_order[(idx - 1) % len(focus_order)]
                        else:
                            focused = focus_order[(idx + 1) % len(focus_order)]
                    elif kind == "down" and k in ("up", "down"):
                        # Scroll focused debug panel.
                        delta = -1 if k == "up" else 1
                        scroll[focused] = max(0, scroll.get(focused, 0) + delta)

                pressed = keys.snapshot()
                if "q" in pressed:
                    break

                if "space" in pressed:
                    drive.emergency_stop()
                    state.manual_throttle = 0.0
                    state.manual_steer = 0.0
                    state.autonomous = False
                    set_message("Emergency stop.")

                if remote_active:
                    try:
                        state.max_speed_setting = max(0.0, min(1.0, float(remote_state.get("max_speed_setting", state.max_speed_setting))))
                        state.speed_setting = max(0.0, min(state.max_speed_setting, float(remote_state.get("speed_setting", state.speed_setting))))
                        state.autonomous = bool(remote_state.get("autonomous", state.autonomous))
                        state.manual_throttle = max(-1.0, min(1.0, float(remote_state.get("manual_throttle", 0.0))))
                        state.manual_steer = max(-1.0, min(1.0, float(remote_state.get("manual_steer", 0.0))))
                        if bool(remote_state.get("emergency_stop", False)):
                            drive.emergency_stop()
                            state.autonomous = False
                            state.manual_throttle = 0.0
                            state.manual_steer = 0.0
                            set_message("Remote emergency stop")
                    except Exception:
                        pass

                # Voice
                if vc:
                    try:
                        while True:
                            cmd = voice_q.get_nowait()
                            _apply_external_command(state, cmd)
                            message = f"Voice: {cmd[0]}"
                            last_voice_cmd = cmd
                            last_voice_ts = time.time()
                    except queue.Empty:
                        pass

                # IR
                if irc and irc.available:
                    for code in irc.poll_codes():
                        cmd = irc.parse_code(code)
                        if cmd:
                            ir_q.put(cmd)
                            last_ir_code = code
                            last_ir_ts = time.time()

                if irc:
                    try:
                        while True:
                            cmd = ir_q.get_nowait()
                            _apply_external_command(state, cmd)
                            message = f"IR: {cmd[0]}"
                            last_ir_cmd = cmd
                            last_ir_ts = time.time()
                    except queue.Empty:
                        pass

                # Raw motor override (server-side motor test / diagnostics)
                raw_active = bool(
                    link
                    and link.stats.connected
                    and last_raw_motors_ts is not None
                    and (now - float(last_raw_motors_ts)) <= 0.4
                    and isinstance(raw_motors, dict)
                )

                if raw_active:
                    if not raw_prev:
                        # Reset drive state so we don't jump when raw mode ends.
                        try:
                            drive.emergency_stop()
                        except Exception:
                            pass
                    for name, sp in raw_motors.items():
                        if name in motor_map:
                            try:
                                motor_map[name].set(float(sp), coast_on_stop=cfg.coast_on_stop)
                            except Exception:
                                pass
                    target_left, target_right = 0.0, 0.0
                    last_mode = "raw"
                else:
                    if raw_prev:
                        try:
                            drive.emergency_stop()
                        except Exception:
                            pass
                    # Targets
                    if state.autonomous:
                        if remote_active:
                            # Autonomous decisions are expected to come from the server as throttle/steer.
                            target_left, target_right = mix_throttle_steer(
                                state.manual_throttle,
                                state.manual_steer,
                                speed_setting=state.speed_setting,
                                cfg=cfg,
                                full_speed=False,
                            )
                            last_mode = "server"
                        else:
                            target_left, target_right = 0.0, 0.0
                            last_mode = "autonomous"
                            set_message("Autonomous: waiting for server")
                    else:
                        if not remote_active:
                            throttle = (1.0 if "w" in pressed else 0.0) + (-1.0 if "s" in pressed else 0.0)
                            steer = (-1.0 if "a" in pressed else 0.0) + (1.0 if "d" in pressed else 0.0)

                            if any(k in pressed for k in ("w", "a", "s", "d")):
                                state.manual_throttle = max(-1.0, min(1.0, throttle))
                                state.manual_steer = max(-1.0, min(1.0, steer))
                            else:
                                state.manual_throttle = 0.0
                                state.manual_steer = 0.0
                            last_mode = "manual"
                        else:
                            last_mode = "remote"

                        target_left, target_right = mix_throttle_steer(
                            state.manual_throttle,
                            state.manual_steer,
                            speed_setting=state.speed_setting,
                            cfg=cfg,
                            full_speed=False,
                        )

                raw_prev = raw_active

                last_targets = (float(target_left), float(target_right))

                now = time.time()
                dt = max(0.0, now - last_t)
                last_t = now

                if not raw_active:
                    drive.update(target_left=target_left, target_right=target_right, dt_s=dt)

                # Auto-reset transient messages.
                if message_until and now >= message_until:
                    message = "Ready."
                    message_until = 0.0

                # Camera streaming (optional)
                if link is not None and link.stats.connected and cam and cam.available and features.camera:
                    if (now - last_frame_tx) >= 0.5:
                        jpeg = cam.read_jpeg()
                        if jpeg:
                            try:
                                b64 = base64.b64encode(jpeg).decode("ascii")
                                link.send({"type": "frame", "ts": now, "jpeg_b64": b64})
                                last_frame_tx = now
                                frames_sent += 1
                            except Exception:
                                pass

                # Telemetry TX (optional)
                if link is not None and (now - last_telemetry_tx) >= 0.10:
                    try:
                        motors_tlm: dict[str, Any] = {}
                        for n, m in motor_map.items():
                            motors_tlm[n] = {"speed": float(m.last_speed), "invert": bool(m.pins.invert)}

                        voice_info: dict[str, Any] = {
                            "enabled": bool(features.voice),
                            "available": bool(vc and vc.available),
                        }
                        if vc and hasattr(vc, "last_event"):
                            try:
                                last_text, last_cmd, last_ts = vc.last_event()  # type: ignore[attr-defined]
                                voice_info.update(
                                    {
                                        "last_words": last_text or "—",
                                        "parsed": (last_cmd[0] if last_cmd else "—"),
                                        "applied": (last_voice_cmd[0] if last_voice_cmd else "—"),
                                        "age": age_s(last_ts or last_voice_ts),
                                    }
                                )
                            except Exception:
                                pass

                        ir_info: dict[str, Any] = {
                            "enabled": bool(features.ir),
                            "available": bool(irc and irc.available),
                            "last_code": last_ir_code or "—",
                            "applied": (last_ir_cmd[0] if last_ir_cmd else "—"),
                            "age": age_s(last_ir_ts),
                        }

                        cam_info: dict[str, Any] = {
                            "enabled": bool(features.camera),
                            "available": bool(cam and cam.available),
                            "frames_sent": int(frames_sent),
                            "last_frame_age": ("—" if not last_frame_tx else f"{int(now - last_frame_tx)}s"),
                        }

                        link.send(
                            {
                                "type": "telemetry",
                                "ts": now,
                                "motors": motors_tlm,
                                "state": {
                                    "speed_setting": float(state.speed_setting),
                                    "max_speed_setting": float(state.max_speed_setting),
                                    "autonomous": bool(state.autonomous),
                                    "manual_throttle": float(state.manual_throttle),
                                    "manual_steer": float(state.manual_steer),
                                },
                                "last_cmd": {
                                    "source": str(last_mode),
                                    "manual": f"thr={state.manual_throttle:+.1f} steer={state.manual_steer:+.1f}",
                                    "targets": f"L={last_targets[0]:+.2f}  R={last_targets[1]:+.2f}",
                                },
                                "message": str(message),
                                "voice": voice_info,
                                "camera": cam_info,
                                "ir": ir_info,
                            }
                        )
                        last_telemetry_tx = now
                    except Exception:
                        pass

                # Render
                layout["top"].update(top_panel())
                layout["motors"].update(Panel(_motor_table(motor_map), title="Motors", border_style="bright_green"))
                layout["status"].update(status_panel())
                layout["auto"].update(autonomous_panel())
                layout["voice"].update(voice_panel())
                layout["camera"].update(camera_panel())
                layout["ir"].update(ir_panel())
                layout["bottom"].update(bottom_panel(message))

                time.sleep(0.02)

        return 0

    finally:
        try:
            keys.stop()
        except Exception:
            pass
        _drain_stdin()
        try:
            if link is not None:
                link.stop()
        except Exception:
            pass
        try:
            if vc:
                vc.stop()
        except Exception:
            pass
        try:
            if irc:
                irc.close()
        except Exception:
            pass
        try:
            if cam:
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
            ("dry", "Run without GPIO (safe on Windows)") ,
            ("real", "Run on Raspberry Pi GPIO") ,
        ],
    )
    dry_run = mode == "dry"

    _header(console)
    action = _select_one(
        console,
        "Operation",
        [
            ("live", "Drive the robot with a live dashboard") ,
            ("test", "Run motor test with progress bars") ,
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
            ("voice", "Voice", "Voice control (SpeechRecognition + PyAudio)"),
            ("camera", "Camera", "Obstacle detection (OpenCV)"),
            ("ir", "IR", "IR remote (LIRC)"),
        ],
        default_checked=set(),
    )

    features = FeatureFlags(
        motor=True,
        voice="voice" in picked,
        camera="camera" in picked,
        ir="ir" in picked,
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
    raise SystemExit(
        "Do not run tui.py directly.\n\n"
        "Run one of these instead:\n"
        "  python tui_client.py\n"
        "  python server/tui_server.py\n"
    )
