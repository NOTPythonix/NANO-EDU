from __future__ import annotations

import sys
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from movement import Motor
import robot_config

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except Exception as exc:
    raise SystemExit(
        "This TUI requires the 'rich' package.\n\n"
        "Install it with:\n"
        "  pip install rich\n"
    ) from exc


def _motor_label(short: str) -> str:
    return {
        "lf": "Left Front",
        "lr": "Left Rear",
        "rf": "Right Front",
        "rr": "Right Rear",
    }.get(short, short)


def _motor_pins(name: str, pins: Optional[Dict[str, int]] = None) -> str:
    source = pins or dict(robot_config.MOTORS.get(name, {}))
    if not source:
        return "—"
    return f"IN1 {source.get('in1', '—')} / IN2 {source.get('in2', '—')} / EN {source.get('en', '—')}"


def _marquee(text: str, width: int, offset: float, gap: int = 4) -> str:
    s = str(text)
    width = max(1, int(width))
    if len(s) <= width:
        return s.ljust(width)

    spacer = " " * max(1, int(gap))
    cycle = s + spacer
    pos = int(offset) % len(cycle)
    window = cycle[pos : pos + width]
    if len(window) < width:
        window += cycle[: width - len(window)]
    return window


def _ellipsize(s: str, max_len: int) -> str:
    s = str(s)
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def ui_on_off(v: bool) -> str:
    return "[green]ON[/]" if bool(v) else "[red]OFF[/]"


def ui_on_off_error(state: str) -> str:
    s = str(state or "").strip().lower()
    if s == "on":
        return "[green]ON[/]"
    if s == "error":
        return "[red]ERROR[/]"
    return "[red]OFF[/]"


def build_info_grid(rows: list[tuple[str, str]]) -> Table:
    grid = Table.grid(expand=True, pad_edge=False, padding=(0, 1))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1, justify="right")
    for k, v in rows:
        grid.add_row(str(k), str(v))
    return grid


def _analysis_time_label(now_s: float, ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    try:
        stamp = time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:
        stamp = "—"
    age_s = int(max(0.0, float(now_s) - float(ts)))
    return f"{stamp} ({age_s}s ago)"


def _age_seconds_label(analysis_state: dict[str, Any], *, age_key: str, ts_key: str, now_s: float) -> str:
    age_val = analysis_state.get(age_key)
    if age_val is not None:
        try:
            return f"{float(age_val):.1f}s"
        except Exception:
            pass
    ts_val = analysis_state.get(ts_key)
    if ts_val is not None:
        try:
            return f"{max(0.0, float(now_s) - float(ts_val)):.1f}s"
        except Exception:
            pass
    return "—"


def _detection_rows(analysis_state: dict[str, Any]) -> list[dict[str, Any]]:
    detections = None
    if isinstance(analysis_state, dict):
        detections = analysis_state.get("all_detections") or analysis_state.get("top_detections")
    rows: list[dict[str, Any]] = list(detections) if isinstance(detections, list) else []
    if not rows and isinstance(analysis_state, dict):
        label = str(analysis_state.get("label", "—") or "—")
        conf = float(analysis_state.get("confidence", 0.0) or 0.0)
        if label not in ("—", "none", "unknown"):
            rows = [{"label": label, "confidence": conf, "x1": "—", "y1": "—", "x2": "—", "y2": "—"}]
    return rows


def build_detection_table(analysis_state: dict[str, Any], *, offset: int = 0, window: int = 24) -> tuple[Table, bool, bool]:
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("#", width=3, justify="right")
    table.add_column("Label", style="bold cyan", overflow="ellipsis")
    table.add_column("Conf", width=7, justify="right")
    table.add_column("Box", overflow="ellipsis")

    rows = _detection_rows(analysis_state)

    if not rows:
        table.add_row("—", "No detections", "—", "—")
        return table, False, False

    window = max(1, int(window))
    off = max(0, int(offset))
    max_off = max(0, len(rows) - window)
    off = min(off, max_off)
    view = rows[off : off + window]
    more_up = off > 0
    more_down = (off + window) < len(rows)

    for idx, det in enumerate(view, start=off + 1):
        label = str(det.get("label", "—"))
        conf = float(det.get("confidence", 0.0) or 0.0)
        box_txt = f"{det.get('x1', '—')},{det.get('y1', '—')} -> {det.get('x2', '—')},{det.get('y2', '—')}"
        table.add_row(str(idx), label, f"{conf:.2f}", box_txt)
    return table, more_up, more_down


def _detection_stats_rows(now_s: float, analysis_state: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Model error", str(analysis_state.get("model_error", analysis_state.get("error", "—")) or "—")),
        ("Model path", str(analysis_state.get("model_path", "—") or "—")),
        ("Prompt classes", str(analysis_state.get("prompt_class_count", "—"))),
        ("FPS", f"{float(analysis_state.get('analysis_fps', 0.0) or 0.0):.2f}"),
        ("Detections", str(analysis_state.get("detections", "—"))),
        ("Raw detections", str(analysis_state.get("raw_detections", "—"))),
        ("Analysis age", f"{analysis_state.get('age_s', '—')}s" if analysis_state.get("age_s") is not None else "—"),
        ("Last update", _analysis_time_label(now_s, analysis_state.get("updated_ts") if isinstance(analysis_state, dict) else None)),

        ("Phone detected", ui_on_off(bool(analysis_state.get("phone_detected", False)))),
        (
            "Phone detect age",
            _age_seconds_label(
                analysis_state,
                age_key="phone_detected_age_s",
                ts_key="phone_detected_ts",
                now_s=now_s,
            ),
        ),
        ("Phone alert age", _age_seconds_label(analysis_state, age_key="phone_alert_age_s", ts_key="phone_alert_triggered_ts", now_s=now_s)),

        ("Badge detected", ui_on_off(bool(analysis_state.get("badge_detected", False)))),
        (
            "Badge detect age",
            _age_seconds_label(
                analysis_state,
                age_key="badge_detected_age_s",
                ts_key="badge_detected_ts",
                now_s=now_s,
            ),
        ),
        ("Badge alert age", _age_seconds_label(analysis_state, age_key="badge_alert_age_s", ts_key="badge_alert_triggered_ts", now_s=now_s)),

        ("Wordmark detected", ui_on_off(bool(analysis_state.get("wordmark_detected", False)))),
        (
            "Wordmark detect age",
            _age_seconds_label(
                analysis_state,
                age_key="wordmark_detected_age_s",
                ts_key="wordmark_detected_ts",
                now_s=now_s,
            ),
        ),
        ("Wordmark alert age", _age_seconds_label(analysis_state, age_key="uniform_alert_age_s", ts_key="uniform_alert_triggered_ts", now_s=now_s)),

        ("Person medium", ui_on_off(bool(analysis_state.get("person_medium_close", False)))),
        ("Person close", ui_on_off(bool(analysis_state.get("person_close", False)))),

        ("Missing ID badge", ui_on_off(bool(analysis_state.get("missing_id_badge", False)))),
        ("Badge missing for", f"{float(analysis_state.get('missing_badge_for_s', 0.0) or 0.0):.1f}s"),

        ("Missing wordmark", ui_on_off(bool(analysis_state.get("missing_uniform_wordmark", False)))),
        ("Wordmark missing for", f"{float(analysis_state.get('missing_wordmark_for_s', 0.0) or 0.0):.1f}s"),
    ]


def build_detection_stats_panel(
    now_s: float,
    analysis_state: dict[str, Any],
    *,
    offset: int = 0,
    window: int = 16,
    focused: bool = False,
) -> tuple[Panel, bool, bool]:
    rows = _detection_stats_rows(now_s, analysis_state)
    window = max(1, int(window))
    off = max(0, int(offset))
    max_off = max(0, len(rows) - window)
    off = min(off, max_off)
    view = rows[off : off + window]
    more_up = off > 0
    more_down = (off + window) < len(rows)

    subtitle = ""
    if focused:
        up_ind = "↑" if more_up else " "
        dn_ind = "↓" if more_down else " "
        subtitle = f"[dim]←/→ select  ↑/↓ scroll  {up_ind}{dn_ind}[/]"

    panel = Panel(build_info_grid(view), title="Model Stats", subtitle=subtitle, border_style=("bright_yellow" if focused else "bright_magenta"))
    return panel, more_up, more_down


def build_detection_page_layout(
    now_s: float,
    analysis_state: dict[str, Any],
    message_text: str,
    *,
    page_title: str = "Detection",
    focused_panel: str = "detections",
    scroll_offsets: Optional[dict[str, int]] = None,
    window_rows: Optional[dict[str, int]] = None,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="main"),
        Layout(name="bottom", size=3),
    )
    layout["main"].split_row(Layout(name="detections", ratio=7), Layout(name="stats", ratio=5))
    layout["top"].update(
        Panel(
            "[bold]W/A/S/D[/]=drive   [bold]Space[/]=stop   [bold]O[/]=toggle autonomous   "
            "[bold][[/]/[bold]][/]=speed cap   [bold]←/→[/]=select panel   [bold]↑/↓[/]=scroll   [bold]Q[/]=next page   [bold]Esc[/]=quit",
            border_style="bright_cyan",
        )
    )

    offsets = dict(scroll_offsets or {})
    windows = dict(window_rows or {})
    det_off = max(0, int(offsets.get("detections", 0)))
    stat_off = max(0, int(offsets.get("stats", 0)))
    det_window = max(1, int(windows.get("detections", 24)))
    stat_window = max(1, int(windows.get("stats", 16)))

    det_table, det_more_up, det_more_down = build_detection_table(analysis_state, offset=det_off, window=det_window)
    det_subtitle = ""
    if focused_panel == "detections":
        up_ind = "↑" if det_more_up else " "
        dn_ind = "↓" if det_more_down else " "
        det_subtitle = f"[dim]←/→ select  ↑/↓ scroll  {up_ind}{dn_ind}[/]"

    stats_panel, _, _ = build_detection_stats_panel(
        now_s,
        analysis_state,
        offset=stat_off,
        window=stat_window,
        focused=(focused_panel == "stats"),
    )

    layout["detections"].update(
        Panel(
            det_table,
            title="Detected Objects",
            subtitle=det_subtitle,
            border_style=("bright_yellow" if focused_panel == "detections" else "bright_green"),
        )
    )
    layout["stats"].update(stats_panel)
    layout["bottom"].update(Panel(Text(str(message_text)), border_style="dim"))
    return layout


def _header(console: Console) -> None:
    console.clear()
    console.print(Align.center("[bold bright_cyan]HSEF Robot Control[/]  [dim]TUI[/]"))


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
    max_speed_setting: float = 0.65
    autonomous: bool = False
    manual_throttle: float = 0.0
    manual_steer: float = 0.0
    voice_active: bool = False
    voice_throttle: float = 0.0
    voice_steer: float = 0.0


@dataclass
class DetectionPageUiState:
    focused: str = "detections"
    scroll: Dict[str, int] = field(default_factory=lambda: {"detections": 0, "stats": 0})


def detection_page_window_rows(main_h: int) -> dict[str, int]:
    # Detection and stats panels are side-by-side, so they share the same height.
    panel_h = max(3, int(main_h))
    return {
        "detections": max(1, panel_h - 4),
        "stats": max(1, panel_h - 2),
    }


def handle_detection_page_nav(ui_state: DetectionPageUiState, key: str) -> bool:
    k = str(key or "").lower().strip()
    if k in ("left", "right"):
        order = ["detections", "stats"]
        current = ui_state.focused if ui_state.focused in order else "detections"
        idx = order.index(current)
        ui_state.focused = order[(idx - 1) % len(order)] if k == "left" else order[(idx + 1) % len(order)]
        return True
    if k in ("up", "down"):
        delta = -1 if k == "up" else 1
        cur = ui_state.focused if ui_state.focused in ("detections", "stats") else "detections"
        ui_state.scroll[cur] = max(0, int(ui_state.scroll.get(cur, 0)) + delta)
        return True
    return False


class KeyboardKeys:
    def __init__(self):
        try:
            import keyboard as keyboard_lib  # type: ignore
        except Exception as e:
            raise RuntimeError("keyboard is required for live mode. Install with: pip install keyboard") from e

        self._kb = keyboard_lib
        self._pressed: Set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        self._events: "queue.Queue[Tuple[str, str]]" = queue.Queue()

    def start(self) -> None:
        def normalize_name(name: Any) -> Optional[str]:
            text = str(name or "").strip().lower()
            if not text:
                return None
            if text in {"up", "down", "left", "right", "enter", "esc", "space"}:
                return text
            if len(text) == 1:
                return text
            return None

        def on_event(event) -> None:
            try:
                event_type = str(getattr(event, "event_type", "")).lower()
                if event_type not in {"down", "up"}:
                    return
                key_name = normalize_name(getattr(event, "name", None))
                if key_name is None:
                    return
                with self._lock:
                    if event_type == "down":
                        self._pressed.add(key_name)
                        self._events.put(("down", key_name))
                    else:
                        self._pressed.discard(key_name)
                        self._events.put(("up", key_name))
            except Exception:
                return

        try:
            self._listener = self._kb.hook(on_event, suppress=False)
        except TypeError:
            self._listener = self._kb.hook(on_event)
        except Exception:
            self._listener = self._kb.hook(on_event)

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._kb.unhook(self._listener)
            except Exception:
                pass
            self._listener = None

    def snapshot(self) -> Set[str]:
        with self._lock:
            return set(self._pressed)

    def poll_event(self) -> Optional[Tuple[str, str]]:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None


@dataclass
class PageNavigationState:
    page_index: int = 0
    q_down: bool = False
    page_count: int = 2

    def on_key_down(self, key: str) -> Optional[str]:
        k = str(key or "").strip().lower()
        if k == "q":
            if not self.q_down and self.page_count > 0:
                self.page_index = (self.page_index + 1) % self.page_count
            self.q_down = True
            return "page"
        if k == "esc":
            return "quit"
        return None

    def on_key_up(self, key: str) -> None:
        if str(key or "").strip().lower() == "q":
            self.q_down = False


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
    keys = KeyboardKeys()
    keys.start()

    selected = 0
    fd = None
    old_attrs = None
    try:
        console.clear()
        try:
            fd = sys.stdin.fileno()
            old_attrs = _silence_terminal(fd)
        except Exception:
            fd = None
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
        try:
            if fd is not None:
                _restore_terminal(fd, old_attrs)
        except Exception:
            pass
        _drain_stdin()


def _select_checklist(
    console: Console,
    title: str,
    options: list[tuple[str, str, str]],
    *,
    default_checked: Optional[set[str]] = None,
) -> set[str]:
    def render(title: str, options: list[tuple[str, str, str]], selected: int, checked: set[str]) -> Panel:
        t = Table(box=box.SIMPLE, show_header=False, expand=True)
        t.add_column("sel", width=2)
        t.add_column("box", width=3)
        t.add_column("Option", style="bold")
        t.add_column("Description", style="dim")

        for i, (key, label, desc) in enumerate(options):
            is_sel = i == selected
            marker = "➤" if is_sel else " "
            box_txt = Text("[x]" if key in checked else "[ ]")
            opt_style = "reverse bold" if is_sel else ""
            shown_label = f"[{opt_style}]{label}[/]" if opt_style else label
            t.add_row(marker, box_txt, shown_label, desc)

        help_line = "[dim]↑/↓ move, Space toggle, Enter confirm, Q/Esc cancel[/]"
        return Panel(t, title=title, border_style="bright_blue", subtitle=help_line)

    keys = KeyboardKeys()
    keys.start()

    checked: set[str] = set(default_checked or set())
    selected = 0
    fd = None
    old_attrs = None

    try:
        console.clear()
        try:
            fd = sys.stdin.fileno()
            old_attrs = _silence_terminal(fd)
        except Exception:
            fd = None
        with Live(render(title, options, selected, checked), console=console, refresh_per_second=30, screen=True) as live:
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
                live.update(render(title, options, selected, checked))
    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        try:
            keys.stop()
        except Exception:
            pass
        try:
            if fd is not None:
                _restore_terminal(fd, old_attrs)
        except Exception:
            pass
        _drain_stdin()


def _drain_stdin() -> None:
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


def _silence_terminal(fd: int):
    if sys.platform.startswith("win"):
        return None
    try:
        import termios  # type: ignore
    except Exception:
        return None
    try:
        attrs = termios.tcgetattr(fd)
        new = list(attrs)
        new[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSANOW, new)
        return attrs
    except Exception:
        return None


def _restore_terminal(fd: int, old_attrs) -> None:
    if sys.platform.startswith("win") or old_attrs is None:
        return
    try:
        import termios  # type: ignore

        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
    except Exception:
        pass

    try:
        import select

        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            sys.stdin.read(1)
    except Exception:
        pass


class ErrorMessageQueue:
    def __init__(self, *, display_s: float = 3.0):
        self._display_s = float(display_s)
        self._seq = 0
        self._pending: list[tuple[int, int, str]] = []
        self._last_by_source: Dict[str, str] = {}

    def feed(self, source: str, err: Optional[str], *, now_s: float, prefix: str, priority: int, label: Optional[str] = None) -> None:
        src = str(source)
        txt = str(err or "").strip()
        if not txt or txt == "—":
            return
        shown = f"{str(prefix)} {str(label or src)}: {txt}".strip()
        prev = self._last_by_source.get(src, "")
        if txt != prev:
            self._pending.append((int(priority), self._seq, shown))
            self._seq += 1
            self._last_by_source[src] = txt

    def next_message(self, *, now_s: float, current: str, current_until: float, ready: str = "Ready.") -> tuple[str, float]:
        if current_until and float(now_s) < float(current_until):
            return current, current_until
        if self._pending:
            best_i = 0
            best = self._pending[0]
            for i, item in enumerate(self._pending[1:], start=1):
                if item[0] > best[0] or (item[0] == best[0] and item[1] < best[1]):
                    best_i = i
                    best = item
            self._pending.pop(best_i)
            return best[2], float(now_s) + self._display_s
        return str(ready), 0.0


def _build_motor_table(rows: list[dict[str, Any]], *, now_s: Optional[float] = None) -> Table:
    t = Table(box=box.SIMPLE, expand=True, pad_edge=False, padding=(0, 1))
    t.add_column("Motor", style="bold cyan", no_wrap=True, overflow="ellipsis", width=15)
    t.add_column("Pins", style="dim", no_wrap=True, overflow="ellipsis", width=10)
    t.add_column("Cmd", justify="right", no_wrap=True, width=5)
    t.add_column("Inv", justify="center", no_wrap=True, width=3)
    t.add_column("Dir", justify="center", no_wrap=True, width=4)
    t.add_column("Power", justify="left", no_wrap=True, width=12)

    def dir_of(s: float) -> str:
        if abs(s) < 1e-3:
            return "STOP"
        return "FWD" if s > 0 else "REV"

    def bar(s: float) -> str:
        mag = max(0.0, min(1.0, abs(s)))
        blocks = int(round(mag * 10))
        fill = "█" * blocks
        empty = "░" * (10 - blocks)
        color = "green" if s > 0 else ("red" if s < 0 else "white")
        return f"[{color}]{fill}{empty}[/]"

    marquee_offset = (float(now_s) if now_s is not None else time.time()) * 6.0
    for i, row in enumerate(rows):
        s = float(row.get("speed", 0.0))
        d = f"{dir_of(s):<4}"[:4]
        t.add_row(
            _marquee(str(row.get("name") or ""), 15, marquee_offset + (i * 2)),
            _marquee(str(row.get("pins") or "—"), 10, marquee_offset + (i * 3)),
            f"{s:+.2f}"[:5],
            "Y" if bool(row.get("invert", False)) else "N",
            d,
            bar(s),
        )
    return t


def build_motor_table_from_motor_map(motor_map: Dict[str, Motor], *, now_s: Optional[float] = None) -> Table:
    preferred = ["lf", "lr", "rf", "rr"]
    ordered = [n for n in preferred if n in motor_map] + [n for n in sorted(motor_map.keys()) if n not in preferred]
    rows: list[dict[str, Any]] = []
    for name in ordered:
        m = motor_map[name]
        rows.append({"name": f"{_motor_label(name)} ({name})", "pins": _motor_pins(name, {"in1": m.pins.in1, "in2": m.pins.in2, "en": m.pins.en}), "speed": m.last_speed, "invert": m.pins.invert})
    return _build_motor_table(rows, now_s=now_s)


def build_motor_table_from_telemetry(tlm: dict[str, Any], *, now_s: Optional[float] = None) -> Table:
    motors: dict[str, Any] = {}
    try:
        motors = dict(tlm.get("motors") or {})
    except Exception:
        motors = {}
    preferred = ["lf", "lr", "rf", "rr"]
    ordered = [n for n in preferred if n in motors] + [n for n in sorted(motors.keys()) if n not in preferred]
    rows: list[dict[str, Any]] = []
    for name in ordered:
        m = motors.get(name) or {}
        pins = robot_config.MOTORS.get(name) or {}
        rows.append({"name": f"{_motor_label(name)} ({name})", "pins": f"IN1 {pins.get('in1', '—')} / IN2 {pins.get('in2', '—')} / EN {pins.get('en', '—')}", "speed": float(m.get("speed", 0.0)) if isinstance(m, dict) else 0.0, "invert": bool(m.get("invert", False)) if isinstance(m, dict) else False})
    return _build_motor_table(rows, now_s=now_s)


def main() -> int:
    raise SystemExit(
        "Do not run tui.py directly.\n\n"
        "Run one of these instead:\n"
        "  python tui_client.py\n"
        "  python server/tui_server.py\n"
    )


if __name__ == "__main__":
    raise SystemExit(
        "Do not run tui.py directly.\n\n"
        "Run one of these instead:\n"
        "  python tui_client.py\n"
        "  python server/tui_server.py\n"
    )
