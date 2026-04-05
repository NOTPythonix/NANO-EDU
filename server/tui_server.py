from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from server.net_server import JsonLineRobotServer
from server.rtsp_web import RtspWebUi
from tui import (
    DetectionPageUiState,
    ErrorMessageQueue,
    KeyboardKeys,
    _motor_label,
    _select_checklist,
    _select_one,
    build_detection_page_layout,
    build_info_grid,
    build_motor_table_from_telemetry,
    detection_page_window_rows,
    handle_detection_page_nav,
    ui_on_off,
)
from ui_layout import allocate_round_robin_heights
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text


@dataclass
class ServerCmdState:
    speed_setting: float = 0.65
    max_speed_setting: float = 1.0
    autonomous: bool = False
    manual_throttle: float = 0.0
    manual_steer: float = 0.0
    emergency_stop: bool = False


def _header(console: Console) -> None:
    console.clear()
    console.print(Align.center("[bold bright_cyan]HSEF Robot Control[/]  [dim]TUI[/]"))


def _age_s(now: float, ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return f"{int(max(0.0, now - ts))}s"


def _fmt_when(now: float, ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    try:
        stamp = time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:
        stamp = "—"
    return f"{stamp} ({_age_s(now, ts)} ago)"


def wait_for_robot(console: Console, srv: JsonLineRobotServer) -> None:
    keys = KeyboardKeys()
    keys.start()

    def panel() -> Panel:
        err = srv.stats.last_error
        line1 = f"Listening: {srv.stats.bound or '—'}"
        line2 = f"Robot: {'CONNECTED' if srv.stats.connected else 'WAITING'}"
        line3 = f"Peer: {srv.stats.client_peer or '—'}"
        line4 = f"Error: {err or '—'}"
        body = "\n".join([line1, line2, line3, line4, "", "[dim]Press Q to quit.[/]"])
        style = "bright_green" if srv.stats.connected else "bright_yellow"
        return Panel(body, title="Network Check", border_style=style)

    try:
        _header(console)
        with Live(panel(), console=console, refresh_per_second=10, screen=True) as live:
            while True:
                pressed = keys.snapshot()
                if "q" in pressed:
                    raise SystemExit(130)
                if srv.stats.connected:
                    return
                live.update(panel())
                time.sleep(0.05)
    finally:
        try:
            keys.stop()
        except Exception:
            pass


def run_dashboard(console: Console, srv: JsonLineRobotServer, *, web_ui: Optional[RtspWebUi] = None) -> int:
    keys = KeyboardKeys()
    keys.start()

    cmd = ServerCmdState()
    start_t = time.time()

    TOP_SIZE = 3
    BOTTOM_SIZE = 3

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=TOP_SIZE),
        Layout(name="main"),
        Layout(name="bottom", size=BOTTOM_SIZE),
    )

    layout["main"].split_row(Layout(name="left", ratio=6), Layout(name="right", ratio=5))

    right_panel_order = ["status", "voice", "camera", "ir"]

    message = "Ready."
    message_until = 0.0
    error_messages = ErrorMessageQueue(display_s=3.0)

    try:
        term_h = int(getattr(console, "size").height)
    except Exception:
        term_h = 40

    main_h = max(1, term_h - TOP_SIZE - BOTTOM_SIZE)

    motors_panel_h = max(10, min(18, main_h // 2))
    motors_panel_h = min(motors_panel_h, max(8, main_h - 6))

    right_heights = allocate_round_robin_heights(main_h, right_panel_order, min_panel_h=5)
    status_panel_h = right_heights["status"]
    voice_panel_h = right_heights["voice"]
    camera_panel_h = right_heights["camera"]
    ir_panel_h = right_heights["ir"]

    layout["left"].split_column(Layout(name="motors", size=motors_panel_h), Layout(name="auto"))
    layout["right"].split_column(Layout(name="status", size=status_panel_h), Layout(name="debug"))
    layout["debug"].split_column(Layout(name="voice", size=voice_panel_h), Layout(name="camera", size=camera_panel_h), Layout(name="ir", size=ir_panel_h))

    focus_order = ["status", "auto", "voice", "camera", "ir"]
    focused = "status"
    scroll: Dict[str, int] = {k: 0 for k in focus_order}
    detection_ui = DetectionPageUiState()
    page_index = 0
    q_down = False

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

        right_heights = allocate_round_robin_heights(main_h, right_panel_order, min_panel_h=5)
        status_panel_h = right_heights["status"]
        voice_panel_h = right_heights["voice"]
        camera_panel_h = right_heights["camera"]
        ir_panel_h = right_heights["ir"]

        motors_panel_h = max(10, min(18, main_h // 2))
        motors_panel_h = min(motors_panel_h, max(8, main_h - 6))

        layout["left"]["motors"].size = motors_panel_h
        layout["right"]["status"].size = status_panel_h
        layout["debug"]["voice"].size = voice_panel_h
        layout["debug"]["camera"].size = camera_panel_h
        layout["debug"]["ir"].size = ir_panel_h

    def _focus_border(base: str, key: str) -> str:
        return "bright_yellow" if key == focused else base

    def _scrollable_panel(key: str, title: str, base_border: str, rows: list[tuple[str, str]]) -> Panel:
        # Keep window size aligned to live panel heights so scroll behavior
        # matches what is visibly available in each section.
        if key == "status":
            panel_h = status_panel_h
        elif key == "auto":
            panel_h = max(3, main_h - motors_panel_h)
        elif key in ("voice", "camera", "ir"):
            if key == "voice":
                panel_h = voice_panel_h
            elif key == "camera":
                panel_h = camera_panel_h
            else:
                panel_h = ir_panel_h
        else:
            panel_h = max(3, main_h // 2)

        window = max(1, panel_h - 2)
        off = max(0, int(scroll.get(key, 0)))
        max_off = max(0, len(rows) - window)
        off = min(off, max_off)
        scroll[key] = off

        more_up = off > 0
        more_down = (off + window) < len(rows)
        view = rows[off : off + window]

        grid = build_info_grid(view)

        subtitle = ""
        if key == focused:
            up_ind = "↑" if more_up else " "
            dn_ind = "↓" if more_down else " "
            subtitle = f"[dim]←/→ select  ↑/↓ scroll  {up_ind}{dn_ind}[/]"

        return Panel(grid, title=title, subtitle=subtitle, border_style=_focus_border(base_border, key))

    def _rows_from_ui(tlm: dict[str, Any], key: str, fallback: list[tuple[str, str]]) -> list[tuple[str, str]]:
        ui = tlm.get("ui_rows") if isinstance(tlm, dict) else None
        sec = ui.get(key) if isinstance(ui, dict) else None
        if isinstance(sec, list):
            out: list[tuple[str, str]] = []
            for item in sec:
                if isinstance(item, dict):
                    out.append((str(item.get("k", "—")), str(item.get("v", "—"))))
            if out:
                return out
        return fallback

    def _page_name() -> str:
        return "Controls" if page_index == 0 else "Detection"

    def top_panel() -> Panel:
        return Panel(
            "[bold]W/A/S/D[/]=drive   [bold]Space[/]=stop   [bold]O[/]=toggle autonomous   "
            "[bold][[/]/[bold]][/]=speed cap   [bold]←/→[/]=select panel   [bold]↑/↓[/]=scroll   [bold]Q[/]=next page   [bold]Esc[/]=quit",
            border_style="bright_cyan",
        )

    def detection_page(now: float, analysis_state: dict[str, Any], message_text: str) -> Layout:
        return build_detection_page_layout(
            now,
            analysis_state,
            message_text,
            page_title="Client",
            focused_panel=detection_ui.focused,
            scroll_offsets=detection_ui.scroll,
            window_rows=detection_page_window_rows(main_h),
        )

    def status_panel(now: float, tlm: dict[str, Any]) -> Panel:
        runtime_s = int(now - start_t)

        # Out % from robot telemetry
        out_pct = 0
        try:
            motors = tlm.get("motors") or {}
            out = max(abs(float((motors.get(k) or {}).get("speed", 0.0))) for k in motors.keys()) if motors else 0.0
            out_pct = int(round(max(0.0, min(1.0, out)) * 100))
        except Exception:
            out_pct = 0

        cap_pct = int(round(cmd.max_speed_setting * 100))
        net_info = tlm.get("network") or {}
        client_ip = str(net_info.get("client_ip", "—")) if isinstance(net_info, dict) else "—"

        net = "[green]CONNECTED[/]" if srv.stats.connected else "[red]DISCONNECTED[/]"
        round_trip = f"{float(srv.stats.rtt_ms):.1f} ms" if srv.stats.rtt_ms is not None else "—"
        receive_age = _age_s(now, srv.stats.last_rx_ts)

        rows = [
            ("Network", net),
            ("Client IP", client_ip),
            ("Round trip time", round_trip),
            ("Receive age", receive_age),
            (f"Speed % (Cap {cap_pct}%):", f"{out_pct}%"),
            ("Uptime", f"{runtime_s}s"),
        ]
        rows = _rows_from_ui(tlm, "status", rows)
        return _scrollable_panel("status", "Status", "bright_blue", rows)

    def auto_panel(now: float, tlm: dict[str, Any]) -> Panel:
        st = tlm.get("state") or {}
        last_cmd = tlm.get("last_cmd") or {}
        rows = [
            ("Mode", ui_on_off(bool(st.get("autonomous", False)))),
            ("Decision src", str(last_cmd.get("source", "—"))),
            ("Manual input", str(last_cmd.get("manual", "—"))),
            ("Targets", str(last_cmd.get("targets", "—"))),
            ("Msg", str(tlm.get("message", "—"))),
        ]
        rows = _rows_from_ui(tlm, "autonomous", [(k, str(v)) for k, v in rows])
        base = "green" if bool(st.get("autonomous", False)) else "red"
        return _scrollable_panel("auto", "Autonomous", base, rows)

    def voice_panel(tlm: dict[str, Any]) -> Panel:
        v = tlm.get("voice") or {}
        rows = [
            ("Enabled", ui_on_off(bool(v.get("enabled", False)))),
            ("Available", ui_on_off(bool(v.get("available", False)))),
            ("Last words", str(v.get("last_words", "—"))),
            ("Parsed cmd", str(v.get("parsed", "—"))),
            ("Applied cmd", str(v.get("applied", "—"))),
            ("Age", str(v.get("age", "—"))),
        ]
        rows = _rows_from_ui(tlm, "voice", [(k, str(vv)) for k, vv in rows])
        on = bool(v.get("enabled", False)) and bool(v.get("available", False))
        return _scrollable_panel("voice", "Voice", "green" if on else "red", rows)

    def camera_panel(now: float, tlm: dict[str, Any]) -> Panel:
        c = tlm.get("camera") or {}
        enabled = bool(c.get("enabled", False))
        available = bool(c.get("available", False))
        on = enabled and available
        rows = [
            ("Enabled", ui_on_off(enabled)),
            ("Available", ui_on_off(available)),
            ("Camera err", str(c.get("camera_error", "—"))),
            ("Stream", str(c.get("rtsp_status", "—"))),
            ("Stream URL", str(c.get("rtsp_url", "—"))),
            ("Stream err", str(c.get("rtsp_error", "—"))),
            ("Frames sent", str(c.get("frames_sent", "—"))),
            ("Last frame", str(c.get("last_frame_age", "—"))),
        ]
        rows = _rows_from_ui(tlm, "camera", [(k, str(v)) for k, v in rows])
        return _scrollable_panel("camera", "Camera", "green" if on else "red", rows)

    def ir_panel(tlm: dict[str, Any]) -> Panel:
        i = tlm.get("ir") or {}
        rows = [
            ("Enabled", ui_on_off(bool(i.get("enabled", False)))),
            ("Available", ui_on_off(bool(i.get("available", False)))),
            ("Last code", str(i.get("last_code", "—"))),
            ("Applied cmd", str(i.get("applied", "—"))),
            ("Age", str(i.get("age", "—"))),
        ]
        rows = _rows_from_ui(tlm, "ir", [(k, str(v)) for k, v in rows])
        on = bool(i.get("enabled", False)) and bool(i.get("available", False))
        return _scrollable_panel("ir", "IR", "green" if on else "dim", rows)

    seq = 0
    last_obs: Optional[str] = None
    last_obs_tx = 0.0
    last_analysis_tx = 0.0

    try:
        with Live(layout, console=console, refresh_per_second=20, screen=True) as live:
            while True:
                now = time.time()

                while True:
                    inbound = srv.session.poll()
                    if inbound is None:
                        break
                    if not isinstance(inbound, dict):
                        continue
                    if str(inbound.get("type", "")) != "client_request":
                        continue
                    req = str(inbound.get("request", "")).strip().lower()
                    if req == "autonomous_on":
                        cmd.autonomous = True
                        cmd.manual_throttle = 0.0
                        cmd.manual_steer = 0.0
                        set_message("Client voice requested autonomous ON")
                    elif req == "autonomous_off":
                        cmd.autonomous = False
                        cmd.manual_throttle = 0.0
                        cmd.manual_steer = 0.0
                        set_message("Client voice requested autonomous OFF")

                while True:
                    ev = keys.poll_event()
                    if ev is None:
                        break
                    kind, k = ev
                    if kind == "down":
                        if k == "q":
                            if not q_down:
                                page_index = (page_index + 1) % 2
                                set_message(f"Page switched to {page_index + 1}/2", duration_s=1.5)
                            q_down = True
                        elif k == "esc":
                            raise SystemExit(130)
                        elif k == "o":
                            cmd.autonomous = not cmd.autonomous
                            cmd.manual_throttle = 0.0
                            cmd.manual_steer = 0.0
                            set_message(f"Autonomous: {'ON' if cmd.autonomous else 'OFF'}")
                        elif k == "[":
                            cmd.max_speed_setting = max(0.0, cmd.max_speed_setting - 0.05)
                            cmd.speed_setting = min(cmd.speed_setting, cmd.max_speed_setting)
                        elif k == "]":
                            cmd.max_speed_setting = min(1.0, cmd.max_speed_setting + 0.05)
                            cmd.speed_setting = min(cmd.max_speed_setting, cmd.speed_setting + 0.05)
                        elif k in ("left", "right"):
                            if page_index == 1:
                                handle_detection_page_nav(detection_ui, k)
                            else:
                                idx = focus_order.index(focused)
                                if k == "left":
                                    focused = focus_order[(idx - 1) % len(focus_order)]
                                else:
                                    focused = focus_order[(idx + 1) % len(focus_order)]
                        elif k in ("up", "down"):
                            if page_index == 1:
                                handle_detection_page_nav(detection_ui, k)
                            else:
                                delta = -1 if k == "up" else 1
                                scroll[focused] = max(0, scroll.get(focused, 0) + delta)
                    elif k == "q":
                        q_down = False

                pressed = keys.snapshot()
                if "esc" in pressed:
                    break

                cmd.emergency_stop = False
                if "space" in pressed:
                    cmd.emergency_stop = True
                    cmd.autonomous = False
                    cmd.manual_throttle = 0.0
                    cmd.manual_steer = 0.0
                    set_message("Emergency stop.")

                analysis_state = web_ui.get_latest_analysis() if web_ui else {}
                detection_result: dict[str, Any] = analysis_state if isinstance(analysis_state, dict) else {}
                if detection_result:
                    obstacle = str(detection_result.get("obstacle", "unknown"))
                    label = str(detection_result.get("label", "—") or "—")
                    conf = float(detection_result.get("confidence", 0.0) or 0.0)
                    err_text = str(detection_result.get("error", "") or detection_result.get("model_error", "")).strip().lower()
                    obs_low = obstacle.strip().lower()
                    loading_state = (
                        "model-unavailable" in err_text
                        or "ultralytics unavailable" in err_text
                        or (obs_low == "unknown" and ("model" in err_text and "unavailable" in err_text))
                        or (obs_low == "unknown" and int(detection_result.get("raw_detections", 0) or 0) == 0 and not err_text)
                    )

                    if loading_state:
                        last_obs = "model loading"
                    elif obstacle == "clear":
                        last_obs = "clear"
                    elif label and label not in ("—", "none"):
                        last_obs = f"{obstacle} ({label} {conf:.2f})"
                    else:
                        last_obs = obstacle
                else:
                    last_obs = "model loading" if web_ui else "—"

                if not cmd.autonomous:
                    throttle = (1.0 if "w" in pressed else 0.0) + (-1.0 if "s" in pressed else 0.0)
                    steer = (-1.0 if "a" in pressed else 0.0) + (1.0 if "d" in pressed else 0.0)
                    cmd.manual_throttle = max(-1.0, min(1.0, throttle))
                    cmd.manual_steer = max(-1.0, min(1.0, steer))
                else:
                    # Server-side autonomous: use the latest detection result to drive.
                    if detection_result:
                        cmd.manual_throttle = max(-1.0, min(1.0, float(detection_result.get("throttle", 0.0))))
                        cmd.manual_steer = max(-1.0, min(1.0, float(detection_result.get("steer", 0.0))))
                    else:
                        cmd.manual_throttle = 0.0
                        cmd.manual_steer = 0.0

                seq += 1
                srv.session.send(
                    {
                        "type": "command",
                        "seq": seq,
                        "ts": now,
                        "state": {
                            "speed_setting": cmd.speed_setting,
                            "max_speed_setting": cmd.max_speed_setting,
                            "autonomous": cmd.autonomous,
                            "manual_throttle": cmd.manual_throttle,
                            "manual_steer": cmd.manual_steer,
                            "emergency_stop": cmd.emergency_stop,
                        },
                    }
                )

                # Push obstacle display to client even while manual driving, so the
                # camera feed still reports detections.
                if (now - last_obs_tx) >= 0.5:
                    srv.session.send({"type": "server_obs", "ts": now, "obs": last_obs or "—"})
                    last_obs_tx = now

                if (now - last_analysis_tx) >= 0.5:
                    srv.session.send({"type": "analysis", "ts": now, "analysis": analysis_state if isinstance(analysis_state, dict) else {}})
                    last_analysis_tx = now

                tlm = srv.session.latest_telemetry or {}

                srv_net_err = str(srv.stats.last_error or "") if not srv.stats.connected else ""
                srv_web_err = str(web_ui.last_error or "") if web_ui else ""

                net_info = tlm.get("network") or {}
                cam_info = tlm.get("camera") or {}
                voice_info = tlm.get("voice") or {}
                ir_info = tlm.get("ir") or {}

                cli_net_err = str(net_info.get("error", "") if isinstance(net_info, dict) else "")
                cli_cam_err = str(cam_info.get("camera_error", "") if isinstance(cam_info, dict) else "")
                cli_rtsp_err = str(cam_info.get("rtsp_error", "") if isinstance(cam_info, dict) else "")
                cli_voice_err = str(voice_info.get("error", "") if isinstance(voice_info, dict) else "")
                cli_ir_err = str(ir_info.get("error", "") if isinstance(ir_info, dict) else "")
                cli_inference_err = str((analysis_state.get("error", "") or analysis_state.get("model_error", "") or "") if isinstance(analysis_state, dict) else "")

                # Server-side errors first, then client-side relayed errors.
                error_messages.feed("server.network", srv_net_err, now_s=now, prefix="[s]", priority=2, label="Network")
                error_messages.feed("server.web", srv_web_err, now_s=now, prefix="[s]", priority=2, label="Web UI")
                error_messages.feed("client.network", cli_net_err, now_s=now, prefix="[c]", priority=1, label="Network")
                error_messages.feed("client.camera", cli_cam_err, now_s=now, prefix="[c]", priority=1, label="Camera")
                error_messages.feed("client.stream", cli_rtsp_err, now_s=now, prefix="[c]", priority=1, label="Stream")
                error_messages.feed("client.inference", cli_inference_err, now_s=now, prefix="[c]", priority=1, label="Inference")
                error_messages.feed("client.voice", cli_voice_err, now_s=now, prefix="[c]", priority=1, label="Voice")
                error_messages.feed("client.ir", cli_ir_err, now_s=now, prefix="[c]", priority=1, label="IR")

                message, message_until = error_messages.next_message(now_s=now, current=message, current_until=message_until, ready="Ready.")

                if page_index == 0:
                    layout["top"].update(top_panel())
                    layout["motors"].update(Panel(build_motor_table_from_telemetry(tlm, now_s=now), title="Motors", border_style="bright_green"))
                    layout["status"].update(status_panel(now, tlm))
                    layout["auto"].update(auto_panel(now, tlm))
                    layout["voice"].update(voice_panel(tlm))
                    layout["camera"].update(camera_panel(now, tlm))
                    layout["ir"].update(ir_panel(tlm))
                    layout["bottom"].update(Panel(Text(str(message)), border_style="dim"))
                else:
                    live_page = detection_page(now, analysis_state if isinstance(analysis_state, dict) else {}, message)
                live.update(layout if page_index == 0 else live_page)

                time.sleep(0.02)

        return 0

    finally:
        try:
            keys.stop()
        except Exception:
            pass


def run_remote_motor_test(console: Console, srv: JsonLineRobotServer, *, peak: float, cycles_per_motor: int) -> int:
    """Remote motor test: server drives individual motors by sending raw_motors to the client."""

    keys = KeyboardKeys()
    keys.start()

    peak = max(0.0, min(1.0, float(peak)))
    cycles_per_motor = max(1, int(cycles_per_motor))
    on_time_s = 1.0
    off_time_s = 0.5
    step_s = 0.02

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

    # Motor order: prefer standard names.
    preferred = ["lf", "lr", "rf", "rr"]
    motors = preferred

    motor_state: Dict[str, Dict[str, Any]] = {m: {"phase": "idle", "cmd": 0.0} for m in motors}

    def set_state(name: str, phase: str, cmd: float) -> None:
        motor_state[name]["phase"] = f"{phase:<12}"[:12]
        motor_state[name]["cmd"] = float(cmd)

    def send_raw(name: str, speed: float) -> None:
        # Only drive one motor at a time.
        mm = {m: 0.0 for m in motors}
        if name in mm:
            mm[name] = float(speed)
        srv.session.send({"type": "raw_motors", "ts": time.time(), "motors": mm})

    def render_table() -> Table:
        t = Table(box=box.SIMPLE, expand=True)
        t.add_column("Motor", style="bold cyan", width=18)
        t.add_column("Phase", style="magenta", width=12)
        t.add_column("Dir", justify="center", width=4)
        t.add_column("Power", justify="left", width=12)
        t.add_column("%", justify="right", width=4)
        for m in motors:
            cmd = float(motor_state[m]["cmd"])
            p = max(0.0, min(1.0, abs(cmd)))
            t.add_row(f"{_motor_label(m)} ({m})", str(motor_state[m]["phase"]), f"{dir_of(cmd):<4}"[:4], bar_fill(p, sign=cmd), f"{(p*100.0):>3.0f}")
        return t

    def panel() -> Panel:
        subtitle = "[dim]Remote motor test. Press Q to stop.[/]"
        return Panel(render_table(), title="Motor Test (Remote)", border_style="bright_green", subtitle=subtitle)

    try:
        console.clear()
        console.print(
            Panel(
                "[bold]Remote Motor Test[/]\n"
                "Server will command one motor at a time on the robot client.\n"
                f"[dim]Peak: {int(round(peak*100))}%   Cycles/motor: {cycles_per_motor}[/]",
                border_style="bright_blue",
            )
        )

        with Live(panel(), console=console, refresh_per_second=20, screen=True) as live:
            def update_ui() -> None:
                live.update(panel())

            def ramp_run(name: str, signed_peak: float, total_s: float) -> None:
                total_s = max(0.0, float(total_s))
                up_s = total_s / 2.0
                down_s = total_s - up_s

                t0 = time.time()
                while True:
                    pressed = keys.snapshot()
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    t = time.time() - t0
                    if t >= up_s:
                        break
                    frac = 0.0 if up_s <= 1e-9 else (t / up_s)
                    sp = signed_peak * frac
                    send_raw(name, sp)
                    set_state(name, "ramp up", sp)
                    update_ui()
                    time.sleep(step_s)

                send_raw(name, signed_peak)
                set_state(name, "hold", signed_peak)
                update_ui()

                t0 = time.time()
                while True:
                    pressed = keys.snapshot()
                    if "q" in pressed:
                        raise KeyboardInterrupt
                    t = time.time() - t0
                    if t >= down_s:
                        break
                    frac = 0.0 if down_s <= 1e-9 else (1.0 - (t / down_s))
                    sp = signed_peak * frac
                    send_raw(name, sp)
                    set_state(name, "ramp down", sp)
                    update_ui()
                    time.sleep(step_s)

                send_raw(name, 0.0)
                set_state(name, "idle", 0.0)
                update_ui()

            for m in motors:
                set_state(m, "idle", 0.0)
            update_ui()

            for m in motors:
                for _ in range(cycles_per_motor):
                    ramp_run(m, +peak, on_time_s)
                    time.sleep(off_time_s)
                    ramp_run(m, -peak, on_time_s)
                    time.sleep(off_time_s)

        # ensure motors are released
        srv.session.send({"type": "raw_motors", "ts": time.time(), "motors": {m: 0.0 for m in motors}})
        console.print(Panel("[bold bright_green]Done.[/]", border_style="bright_green"))
        return 0

    except KeyboardInterrupt:
        try:
            srv.session.send({"type": "raw_motors", "ts": time.time(), "motors": {m: 0.0 for m in motors}})
        except Exception:
            pass
        console.print(Panel("[bold yellow]Interrupted. Motors stopped.[/]", border_style="yellow"))
        return 130

    finally:
        try:
            keys.stop()
        except Exception:
            pass


def main() -> int:
    console = Console()
    _header(console)

    bind_host = os.environ.get("ROBOT_BIND_HOST", "0.0.0.0")
    bind_port = int(os.environ.get("ROBOT_BIND_PORT", "8765"))

    console.print(Panel("Server will listen for the robot client connection.\nThis step is REQUIRED on the server.", border_style="bright_blue"))

    try:
        bind_host = console.input(f"Bind host [{bind_host}]: ").strip() or bind_host
    except Exception:
        pass
    bind_port = IntPrompt.ask("Bind port", default=bind_port)

    srv = JsonLineRobotServer(host=bind_host, port=bind_port)
    srv.start()

    # The bind happens on a background thread; give it a moment to report
    # either success or a real exception before showing a failure panel.
    t0 = time.time()
    while not srv.stats.listening and not srv.stats.last_error and (time.time() - t0) < 1.0:
        time.sleep(0.05)

    _header(console)
    if not srv.stats.listening:
        if not srv.stats.last_error:
            srv.stats.last_error = "bind startup timed out"
        console.print(Panel(f"Failed to bind/listen: {srv.stats.last_error}", border_style="red"))
        return 2

    # Hard gate: do not proceed until the robot connects.
    wait_for_robot(console, srv)

    web_host = os.environ.get("ROBOT_WEB_HOST", bind_host)
    web_port = int(os.environ.get("ROBOT_WEB_PORT", "8088"))

    def _current_rtsp_url() -> str:
        tlm = srv.session.latest_telemetry or {}
        if not isinstance(tlm, dict):
            return ""

        net_info = tlm.get("network")
        if isinstance(net_info, dict):
            url = str(net_info.get("rtsp_url", "")).strip()
            low = url.lower()
            if url and low not in ("—", "-", "none", "off", "null") and (low.startswith("rtsp://") or low.startswith("http://") or low.startswith("https://")):
                return url

        cam_info = tlm.get("camera")
        if isinstance(cam_info, dict):
            url = str(cam_info.get("rtsp_url", "")).strip()
            low = url.lower()
            if url and low not in ("—", "-", "none", "off", "null") and (low.startswith("rtsp://") or low.startswith("http://") or low.startswith("https://")):
                return url

        return ""

    _header(console)
    console.print(Panel("Choose optional server features.", border_style="bright_blue"))
    picked_server = _select_checklist(
        console,
        "Optional Features",
        [
            ("alerts", "Detection Alert Emails", "Phone/Badge/Wordmark email alerts"),
        ],
        default_checked={"alerts"},
    )
    alert_emails_enabled = "alerts" in picked_server

    web_ui = RtspWebUi(
        host=web_host,
        port=web_port,
        get_rtsp_url=_current_rtsp_url,
        alert_emails_enabled=alert_emails_enabled,
    )
    try:
        web_ui.start()
    except Exception as e:
        _header(console)
        console.print(Panel(f"Failed to start web UI: {e}", border_style="red"))
        return 3

    display_host = web_host
    if display_host in ("0.0.0.0", "::", ""):
        display_host = "127.0.0.1"
    web_url = f"http://{display_host}:{web_port}/"

    # Give operator a short window to open the viewer before choosing operation.
    # Keep this static so terminal link detection (ctrl+click) remains stable.
    rtsp_now = _current_rtsp_url() or "waiting for client stream URL..."
    _header(console)
    console.print(
        Panel(
            "Ctrl+Click to open camera web UI before selecting mode.\n"
            f"[underline bright_cyan]{web_url}[/]\n"
            f"Client Stream: [dim]{rtsp_now}[/]\n"
            "Continuing in 3 seconds...",
            title="Camera Viewer",
            border_style="bright_blue",
        )
    )
    time.sleep(3.0)

    try:
        _header(console)
        action = _select_one(
            console,
            "Operation",
            [
                ("live", "Drive the robot with a live dashboard"),
                ("test", "Run motor test with progress bars (remote)"),
            ],
        )

        if action == "test":
            console.print(Panel("Set session peak power for motor test.", border_style="bright_blue"))
            peak_pct = IntPrompt.ask("Peak power %", default=65)
            peak = max(0.0, min(1.0, float(peak_pct) / 100.0))
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
            return run_remote_motor_test(console, srv, peak=peak, cycles_per_motor=int(cycles_choice))

        _header(console)
        console.print(Panel("Robot connected. Entering dashboard...\n[dim]Press Q to switch pages. Esc quits.[/]", border_style="bright_green"))
        time.sleep(0.6)
        return run_dashboard(console, srv, web_ui=web_ui)
    finally:
        try:
            web_ui.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
