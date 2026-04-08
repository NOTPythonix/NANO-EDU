from __future__ import annotations

import base64
import os
import queue
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import IntPrompt
    from rich.table import Table
    from rich.text import Text
except Exception as exc:
    raise SystemExit(
        "This TUI requires the 'rich' package.\n\n"
        "Install it with:\n"
        "  pip install rich\n"
    ) from exc

from tui import (
    DetectionPageUiState,
    ErrorMessageQueue,
    FeatureFlags,
    KeyboardKeys,
    NetworkConfig,
    PageNavigationState,
    RuntimeState,
    _drain_stdin,
    _header,
    _motor_label,
    _select_checklist,
    _select_one,
    _silence_terminal,
    _restore_terminal,
    build_detection_page_layout,
    build_info_grid,
    build_motor_table_from_motor_map,
    detection_page_window_rows,
    handle_detection_page_nav,
    ui_on_off,
    ui_on_off_error,
)
from camera_control import CameraConfig, CameraController
from movement import DriveConfig, Motor, SkidSteerDrive, build_motors, load_motor_pins, mix_throttle_steer
from robot_net import JsonLineLink
from rtsp_stream import RtspStreamConfig, RtspStreamPublisher
from ui_layout import allocate_round_robin_heights
from voice_control import VoiceConfig, VoiceController


def _drain_pending_console_input() -> None:
    # Prevent a leftover Enter from the previous keyboard-driven menu
    # from being consumed by the next prompt.
    if not sys.platform.startswith("win"):
        return
    try:
        import msvcrt  # type: ignore

        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        return


def _detect_local_ip(preferred_peer: Optional[str] = None) -> str:
    candidates = []
    peer = str(preferred_peer or "").strip()
    if peer and peer not in ("0.0.0.0", "::", "localhost", "127.0.0.1"):
        candidates.append(peer)
    candidates.append("8.8.8.8")

    for target in candidates:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect((target, 1))
                local_ip = sock.getsockname()[0]
            finally:
                sock.close()
            if local_ip and local_ip not in ("0.0.0.0", "127.0.0.1"):
                return str(local_ip)
        except Exception:
            continue

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip and host_ip not in ("0.0.0.0", "127.0.0.1"):
            return str(host_ip)
    except Exception:
        pass

    return "127.0.0.1"


def _apply_external_command(state, cmd: Tuple[str, Optional[int]]) -> None:
    name, val = cmd
    if name == "forward":
        state.autonomous = False
        state.voice_active = True
        state.voice_throttle = 1.0
        state.voice_steer = 0.0
        state.manual_throttle = 1.0
        state.manual_steer = 0.0
        return
    if name == "backward":
        state.autonomous = False
        state.voice_active = True
        state.voice_throttle = -1.0
        state.voice_steer = 0.0
        state.manual_throttle = -1.0
        state.manual_steer = 0.0
        return
    if name == "left":
        state.autonomous = False
        state.voice_active = True
        state.voice_steer = -1.0
        state.manual_steer = -1.0
        return
    if name == "right":
        state.autonomous = False
        state.voice_active = True
        state.voice_steer = 1.0
        state.manual_steer = 1.0
        return
    if name == "stop":
        state.autonomous = False
        state.voice_active = True
        state.voice_throttle = 0.0
        state.voice_steer = 0.0
        state.manual_throttle = 0.0
        state.manual_steer = 0.0
        return
    if name == "autonomous_on":
        state.autonomous = True
        state.manual_throttle = 0.0
        state.manual_steer = 0.0
        state.voice_active = False
        state.voice_throttle = 0.0
        state.voice_steer = 0.0
        return
    if name == "autonomous_off":
        state.autonomous = False
        return
    if name == "speed" and isinstance(val, int):
        state.speed_setting = max(0.0, min(state.max_speed_setting, float(val) / 100.0))


def _ellipsize(s: str, max_len: int) -> str:
    s = str(s)
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _pi_stats_label() -> str:
    # Linux/Raspberry Pi thermal zone in millidegrees Celsius.
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
            raw = f.read().strip()
        temp_c = float(raw) / 1000.0
    except Exception:
        temp_c = None

    # CPU frequency in kHz on Raspberry Pi/Linux.
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r", encoding="utf-8") as f:
            raw = f.read().strip()
        ghz = float(raw) / 1_000_000.0
    except Exception:
        ghz = None

    if temp_c is None and ghz is None:
        return "[dim]N/A[/]"

    if temp_c is None:
        temp_part = "[dim]Temp N/A[/]"
    else:
        if temp_c >= 75.0:
            color = "red"
        elif temp_c >= 65.0:
            color = "yellow"
        else:
            color = "green"
        temp_part = f"[{color}]{temp_c:.1f} C[/]"

    ghz_part = f"[cyan]{ghz:.3f} GHz[/]" if ghz is not None else "[dim]GHz N/A[/]"
    return f"{temp_part}  {ghz_part}"


def run_motor_test_tui(*, dry_run: bool, peak: float, cycles_per_motor: int) -> int:
    console = Console()
    _header(console)

    motors_cfg, _left_names, _right_names = load_motor_pins()
    motor_map, _stop_all, cleanup = build_motors(motors_cfg, dry_run=dry_run, dry_run_output=False)

    preferred = ["lf", "lr", "rf", "rr"]
    motors: list[Motor] = [motor_map[n] for n in preferred if n in motor_map]
    for name, motor in motor_map.items():
        if name not in preferred:
            motors.append(motor)

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

    def dir_of(speed: float) -> str:
        if abs(speed) < 1e-3:
            return "STOP"
        return "FWD" if speed > 0 else "REV"

    motor_state: Dict[str, Dict[str, object]] = {}
    invert_map: Dict[str, bool] = {}
    for motor in motors:
        motor_state[motor.pins.name] = {"phase": "idle", "cmd": 0.0}
        invert_map[motor.pins.name] = bool(motor.pins.invert)

    def set_state(motor: Motor, phase: str) -> None:
        motor_state[motor.pins.name]["phase"] = f"{phase:<12}"[:12]
        motor_state[motor.pins.name]["cmd"] = float(motor.last_speed)

    def render_table() -> Table:
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Motor", style="bold cyan", width=18)
        table.add_column("Phase", style="magenta", width=12)
        table.add_column("Inv", justify="center", width=3)
        table.add_column("Dir", justify="center", width=4)
        table.add_column("Power", justify="left", width=12)
        table.add_column("%", justify="right", width=4)

        for motor in motors:
            st = motor_state[motor.pins.name]
            cmd = float(st["cmd"])
            mag = max(0.0, min(1.0, abs(cmd)))
            table.add_row(
                f"{_motor_label(motor.pins.name)} ({motor.pins.name})",
                str(st["phase"]),
                "Y" if invert_map.get(motor.pins.name, False) else "N",
                dir_of(cmd),
                bar_fill(mag, sign=cmd),
                f"{(mag * 100.0):>3.0f}",
            )
        return table

    def ramp_run(motor: Motor, signed_peak: float, total_s: float, update_ui) -> None:
        total_s = max(0.0, float(total_s))
        up_s = total_s / 2.0
        down_s = total_s - up_s

        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= up_s:
                break
            frac = 0.0 if up_s <= 1e-9 else (elapsed / up_s)
            motor.set(signed_peak * frac)
            set_state(motor, "ramp up")
            update_ui()
            time.sleep(step_s)

        motor.set(signed_peak)
        set_state(motor, "hold")
        update_ui()

        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= down_s:
                break
            frac = 0.0 if down_s <= 1e-9 else (1.0 - (elapsed / down_s))
            motor.set(signed_peak * frac)
            set_state(motor, "ramp down")
            update_ui()
            time.sleep(step_s)

        motor.stop()
        set_state(motor, "idle")
        update_ui()

    console.print(
        Panel(
            "[bold]Motor Test[/]\n"
            "Each motor runs forward then reverse with a smooth ramp up/down.\n"
            f"[dim]Peak: {int(round(peak * 100))}%   Cycles/motor: {cycles_per_motor}[/]\n\n"
            "[dim]Press Ctrl+C to abort safely.[/]",
            border_style="bright_green",
        )
    )

    fd = None
    old_attrs = None
    try:
        def panel() -> Panel:
            return Panel(render_table(), title="Motor Test", border_style="bright_green")

        try:
            fd = sys.stdin.fileno()
            old_attrs = _silence_terminal(fd)
        except Exception:
            fd = None

        with Live(panel(), console=console, refresh_per_second=20) as live:
            def update_ui() -> None:
                live.update(panel())

            for motor in motors:
                set_state(motor, "idle")
                update_ui()
                for _ in range(cycles_per_motor):
                    ramp_run(motor, +peak, on_time_s, update_ui)
                    time.sleep(off_time_s)
                    ramp_run(motor, -peak, on_time_s, update_ui)
                    time.sleep(off_time_s)
        console.print(Panel("[bold bright_green]Done.[/]", border_style="bright_green"))
        return 0
    except KeyboardInterrupt:
        console.print(Panel("[bold yellow]Interrupted. Motors stopped.[/]", border_style="yellow"))
        return 130
    finally:
        try:
            if fd is not None:
                _restore_terminal(fd, old_attrs)
        except Exception:
            pass
        try:
            cleanup()
        except Exception:
            pass
        _drain_stdin()


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

    keys = KeyboardKeys()
    keys.start()

    state = RuntimeState(speed_setting=max(0.0, min(1.0, float(peak))), max_speed_setting=max(0.0, min(1.0, float(peak))))
    voice_q: "queue.Queue[Tuple[str, Optional[int]]]" = queue.Queue()
    ir_q: "queue.Queue[Tuple[str, Optional[int]]]" = queue.Queue()

    vc = VoiceController(voice_q, VoiceConfig(mic_device_index=mic_index)) if features.voice else None
    if vc and vc.available:
        vc.start()

    irc = None
    cam: Optional[CameraController] = None

    link: Optional[JsonLineLink] = None
    if net and net.enabled and net.host:
        link = JsonLineLink(host=net.host, port=net.port, role="client", name="robot")
        link.start()

    client_ip = _detect_local_ip(net.host if net and net.host else None)

    rtsp_pub: Optional[RtspStreamPublisher] = None
    rtsp_public_url = "—"
    cam_init_q: "queue.Queue[tuple[CameraController, Optional[RtspStreamPublisher], str]]" = queue.Queue(maxsize=1)
    cam_init_started = False

    def _start_camera_stack_async() -> None:
        nonlocal cam_init_started
        if cam_init_started or not features.camera:
            return
        cam_init_started = True

        def _worker() -> None:
            cam_w = max(320, int(os.environ.get("ROBOT_CAM_WIDTH", "640")))
            cam_h = max(240, int(os.environ.get("ROBOT_CAM_HEIGHT", "480")))
            cam_q = max(40, min(95, int(os.environ.get("ROBOT_CAM_JPEG_QUALITY", "70"))))
            local_cam = CameraController(CameraConfig(camera_index=0, width=cam_w, height=cam_h, jpeg_quality=cam_q))
            local_rtsp: Optional[RtspStreamPublisher] = None
            local_rtsp_url = "—"
            if local_cam.available:
                stream_port = int(os.environ.get("ROBOT_STREAM_PORT", os.environ.get("ROBOT_RTSP_PORT", "8091")))
                local_rtsp = RtspStreamPublisher(RtspStreamConfig(host="0.0.0.0", mjpeg_port=stream_port))
                if local_rtsp.start():
                    local_rtsp_url = local_rtsp.endpoint_url(client_ip)
            try:
                cam_init_q.put_nowait((local_cam, local_rtsp, local_rtsp_url))
            except queue.Full:
                pass

        threading.Thread(target=_worker, name="CameraInit", daemon=True).start()

    _start_camera_stack_async()

    last_remote_cmd_ts: Optional[float] = None
    remote_state: dict[str, Any] = {}
    last_raw_motors_ts: Optional[float] = None
    raw_motors: dict[str, Any] = {}

    last_frame_tx = 0.0
    last_stream_tx = 0.0
    last_server_frame_tx = 0.0
    last_telemetry_tx = 0.0
    frames_sent = 0

    stream_interval_s = max(0.01, float(os.environ.get("ROBOT_STREAM_INTERVAL_S", "0.02")))
    server_frame_interval_s = max(0.05, float(os.environ.get("ROBOT_SERVER_FRAME_INTERVAL_S", "0.10")))

    start_t = time.time()
    last_t = time.time()

    TOP_SIZE = 3
    BOTTOM_SIZE = 3

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=TOP_SIZE),
        Layout(name="main"),
        Layout(name="bottom", size=BOTTOM_SIZE),
    )

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
    layout["main"].split_row(Layout(name="left", ratio=motors_ratio), Layout(name="right", ratio=right_ratio))

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
    layout["right"].split_column(Layout(name="status", size=status_panel_h), Layout(name="debug_right"))
    layout["debug_right"].split_column(
        Layout(name="voice", size=voice_panel_h),
        Layout(name="camera", size=camera_panel_h),
        Layout(name="ir", size=ir_panel_h),
    )

    def age_s(ts: Optional[float]) -> str:
        if ts is None:
            return "—"
        return f"{int(time.time() - ts)}s"

    focus_order = ["status", "auto", "voice", "camera", "ir"]
    focused = "status"
    scroll: Dict[str, int] = {k: 0 for k in focus_order}
    detection_ui = DetectionPageUiState()
    page_nav = PageNavigationState(page_count=2)

    def _focus_border(base: str, key: str) -> str:
        return "bright_yellow" if key == focused else base

    def _scrollable_panel(key: str, title: str, base_border: str, rows: list[tuple[str, str]]) -> Panel:
        try:
            term_h_now = int(getattr(console, "size").height)
        except Exception:
            term_h_now = 40
        main_h_now = max(1, term_h_now - TOP_SIZE - BOTTOM_SIZE)

        if key == "status":
            panel_h = status_panel_h
        elif key == "auto":
            panel_h = max(3, main_h_now - motors_panel_h)
        elif key in ("voice", "camera", "ir"):
            panel_h = int(voice_panel_h if key == "voice" else (camera_panel_h if key == "camera" else ir_panel_h))
        else:
            panel_h = max(3, main_h_now // 2)

        window = max(1, panel_h - 2)
        off = max(0, int(scroll.get(key, 0)))
        max_off = max(0, len(rows) - window)
        off = min(off, max_off)
        scroll[key] = off

        more_up = off > 0
        more_down = (off + window) < len(rows)
        view = rows[off : off + window]
        grid = build_info_grid(view)

        if key == focused:
            up_ind = "↑" if more_up else " "
            dn_ind = "↓" if more_down else " "
            subtitle = f"[dim]←/→ select  ↑/↓ scroll  {up_ind}{dn_ind}[/]"
        else:
            subtitle = ""

        return Panel(grid, title=title, subtitle=subtitle, border_style=_focus_border(base_border, key))

    def detection_page(now: float, state_data: dict[str, Any], message_text: str) -> Layout:
        return build_detection_page_layout(
            now,
            state_data,
            message_text,
            page_title="Client",
            focused_panel=detection_ui.focused,
            scroll_offsets=detection_ui.scroll,
            window_rows=detection_page_window_rows(main_h),
        )

    last_voice_cmd: Optional[Tuple[str, Optional[int]]] = None
    last_voice_ts: Optional[float] = None
    last_ir_code: Optional[str] = None
    last_ir_cmd: Optional[Tuple[str, Optional[int]]] = None
    last_ir_ts: Optional[float] = None
    last_obs: Optional[str] = None
    last_obs_ts: Optional[float] = None
    analysis_state: dict[str, Any] = {}
    last_targets = (0.0, 0.0)
    last_mode = "manual"

    def status_panel() -> Panel:
        runtime_s = int(time.time() - start_t)
        cap_pct = int(round(state.max_speed_setting * 100))
        pi_stats = _pi_stats_label()
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
            ("Pi Stats", pi_stats),
            ("Run", "DRY" if dry_run else "REAL"),
            ("Network", net_state),
            ("Client IP", client_ip),
            ("Round trip time", round_trip),
            ("Receive age", receive_age),
            ("Peer", peer),
            (f"Speed % (Cap {cap_pct}%):", f"{out_pct}%"),
            ("Uptime", f"{runtime_s}s"),
        ]
        return _scrollable_panel("status", "Status", "bright_blue", rows)

    def autonomous_panel(message_text: str) -> Panel:
        rows = [
            ("Mode", ui_on_off(state.autonomous)),
            ("Net", ui_on_off(bool(link and link.stats.connected))),
            ("Camera", ui_on_off(bool(cam and cam.available))),
            ("Last server obs", last_obs or "—"),
            ("Obs age", age_s(last_obs_ts)),
            ("Decision src", last_mode),
            ("Manual input", f"thr={state.manual_throttle:+.1f} steer={state.manual_steer:+.1f}"),
            ("Targets", f"L={last_targets[0]:+.2f}  R={last_targets[1]:+.2f}"),
            ("Msg", message_text),
        ]
        return _scrollable_panel("auto", "Autonomous", "green" if state.autonomous else "red", rows)

    def voice_panel() -> Panel:
        enabled = bool(features.voice)
        available = bool(vc and vc.available)
        voice_err = str(getattr(vc, "last_error", "") or "") if vc else ""
        last_text = None
        last_cmd = None
        last_ts = None
        if vc and hasattr(vc, "last_event"):
            try:
                last_text, last_cmd, last_ts = vc.last_event()
            except Exception:
                pass
        rows = [
            ("Enabled", ui_on_off(enabled)),
            ("Available", ui_on_off(available)),
            ("Voice err", _ellipsize(voice_err or "—", 30)),
            ("Last words", last_text or "—"),
            ("Parsed cmd", f"{last_cmd[0]}" if last_cmd else "—"),
            ("Applied cmd", f"{last_voice_cmd[0]}" if last_voice_cmd else "—"),
            ("Age", age_s(last_ts or last_voice_ts)),
        ]
        return _scrollable_panel("voice", "Voice", "green" if (enabled and available) else "red", rows)

    def ir_panel() -> Panel:
        rows = [
            ("Enabled", ui_on_off(bool(features.ir))),
            ("Available", ui_on_off(bool(irc and irc.available))),
            ("IR err", _ellipsize(str(getattr(irc, "last_error", "") or "—") if irc else "—", 30)),
            ("Last code", last_ir_code or "—"),
            ("Applied cmd", f"{last_ir_cmd[0]}" if last_ir_cmd else "—"),
            ("Age", age_s(last_ir_ts)),
        ]
        return _scrollable_panel("ir", "IR", "green" if (features.ir and irc and irc.available) else "dim", rows)

    def camera_panel() -> Panel:
        enabled = bool(features.camera)
        available = bool(cam and cam.available)
        cam_err = str(getattr(cam, "last_error", "") or "") if cam else ("initializing..." if enabled else "")
        rtsp_err = str(rtsp_pub.last_error) if rtsp_pub else ""
        has_stream = bool(rtsp_pub and rtsp_pub.available)
        rtsp_state = ui_on_off_error("on" if (has_stream and not rtsp_err) else ("error" if rtsp_err else "off"))
        now = time.time()
        rows = [
            ("Enabled", ui_on_off(enabled)),
            ("Available", ui_on_off(available)),
            ("Camera err", _ellipsize(cam_err or "—", 30)),
            ("Stream", rtsp_state),
            ("Stream URL", _ellipsize(rtsp_public_url, 30)),
            ("Stream err", _ellipsize(rtsp_err or "—", 30)),
            ("Frames sent", str(frames_sent)),
            ("Last frame", "—" if not last_frame_tx else f"{int(now - last_frame_tx)}s"),
        ]
        return _scrollable_panel("camera", "Camera", "green" if (enabled and available) else "red", rows)

    def top_panel() -> Panel:
        return Panel(
            "[bold]W/A/S/D[/]=drive   [bold]Space[/]=stop   [bold]O[/]=toggle autonomous   "
            "[bold][[/]/[bold]][/]=speed cap   [bold]←/→[/]=select panel   [bold]↑/↓[/]=scroll   [bold]Q[/]=next page   [bold]Esc[/]=quit",
            border_style="bright_cyan",
        )

    message = "Ready."
    message_until = 0.0
    error_messages = ErrorMessageQueue(display_s=3.0)

    def set_message(msg: str, *, duration_s: float = 3.0) -> None:
        nonlocal message, message_until
        message = str(msg)
        message_until = (time.time() + float(duration_s)) if duration_s and duration_s > 0 else 0.0

    def trigger_emergency_stop(msg: str = "Emergency stop.") -> None:
        drive.emergency_stop()
        state.manual_throttle = 0.0
        state.manual_steer = 0.0
        state.autonomous = False
        set_message(msg)

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

    fd = None
    old_attrs = None
    try:
        try:
            fd = sys.stdin.fileno()
            old_attrs = _silence_terminal(fd)
        except Exception:
            fd = None
        with Live(layout, console=console, refresh_per_second=20, screen=True) as live:
            while True:
                now = time.time()
                recalc_layout_sizes()

                if features.camera and cam is None:
                    try:
                        cam, rtsp_pub, rtsp_public_url = cam_init_q.get_nowait()
                    except queue.Empty:
                        pass

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
                        elif mtype == "analysis":
                            data = msg.get("analysis")
                            if isinstance(data, dict):
                                analysis_state = data

                remote_active = bool(link and link.stats.connected and last_remote_cmd_ts is not None and (now - float(last_remote_cmd_ts)) <= 0.6)

                while True:
                    ev = keys.poll_event()
                    if ev is None:
                        break
                    kind, k = ev
                    if kind == "down":
                        nav_action = page_nav.on_key_down(k)
                        if nav_action == "quit":
                            raise SystemExit(130)
                        if nav_action == "page":
                            set_message(f"Page switched to {page_nav.page_index + 1}/2", duration_s=1.5)
                            continue
                    elif kind == "up":
                        page_nav.on_key_up(k)

                    if kind == "down" and k == "o":
                        if remote_active:
                            set_message("Server is controlling; use server TUI")
                            continue
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
                        if page_nav.page_index == 1:
                            handle_detection_page_nav(detection_ui, k)
                        else:
                            idx = focus_order.index(focused)
                            focused = focus_order[(idx - 1) % len(focus_order)] if k == "left" else focus_order[(idx + 1) % len(focus_order)]
                    elif kind == "down" and k in ("up", "down"):
                        if page_nav.page_index == 1:
                            handle_detection_page_nav(detection_ui, k)
                        else:
                            delta = -1 if k == "up" else 1
                            scroll[focused] = max(0, scroll.get(focused, 0) + delta)

                pressed = keys.snapshot()
                if "esc" in pressed:
                    break

                if "space" in pressed:
                    trigger_emergency_stop("Emergency stop.")

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

                raw_active = bool(link and link.stats.connected and last_raw_motors_ts is not None and (now - float(last_raw_motors_ts)) <= 0.4 and isinstance(raw_motors, dict))

                if raw_active:
                    for name, sp in raw_motors.items():
                        if name in motor_map:
                            try:
                                motor_map[name].set(float(sp), coast_on_stop=cfg.coast_on_stop)
                            except Exception:
                                pass
                    target_left, target_right = 0.0, 0.0
                    last_mode = "raw"
                else:
                    # Only apply local keyboard fallback when server is not actively driving.
                    if not remote_active:
                        throttle = (1.0 if "w" in pressed else 0.0) + (-1.0 if "s" in pressed else 0.0)
                        steer = (-1.0 if "a" in pressed else 0.0) + (1.0 if "d" in pressed else 0.0)
                        if any(k in pressed for k in ("w", "a", "s", "d")):
                            state.manual_throttle = max(-1.0, min(1.0, throttle))
                            state.manual_steer = max(-1.0, min(1.0, steer))
                        elif not state.voice_active:
                            state.manual_throttle = 0.0
                            state.manual_steer = 0.0
                        last_mode = "manual"
                    else:
                        last_mode = "remote"

                    if vc:
                        try:
                            while True:
                                cmd = voice_q.get_nowait()
                                if remote_active and link is not None and cmd[0] in ("autonomous_on", "autonomous_off"):
                                    try:
                                        link.send({"type": "client_request", "request": cmd[0], "ts": time.time()})
                                        message = f"Voice request sent: {cmd[0]}"
                                    except Exception:
                                        _apply_external_command(state, cmd)
                                elif cmd[0] == "stop":
                                    _apply_external_command(state, cmd)
                                    trigger_emergency_stop("Emergency stop.")
                                else:
                                    _apply_external_command(state, cmd)
                                last_voice_cmd = cmd
                                last_voice_ts = time.time()
                                if not (remote_active and cmd[0] in ("autonomous_on", "autonomous_off")):
                                    message = f"Voice: {cmd[0]}"
                        except queue.Empty:
                            pass

                    if irc:
                        try:
                            while True:
                                cmd = ir_q.get_nowait()
                                _apply_external_command(state, cmd)
                                last_ir_cmd = cmd
                                last_ir_ts = time.time()
                                message = f"IR: {cmd[0]}"
                        except queue.Empty:
                            pass

                    remote_thr = float(remote_state.get("manual_throttle", 0.0)) if isinstance(remote_state, dict) else 0.0
                    remote_str = float(remote_state.get("manual_steer", 0.0)) if isinstance(remote_state, dict) else 0.0
                    remote_auto = bool(remote_state.get("autonomous", False)) if isinstance(remote_state, dict) else False
                    server_is_actively_commanding = bool(remote_active and (remote_auto or abs(remote_thr) > 1e-3 or abs(remote_str) > 1e-3))

                    if not server_is_actively_commanding and not any(k in pressed for k in ("w", "a", "s", "d")) and state.voice_active:
                        state.manual_throttle = float(state.voice_throttle)
                        state.manual_steer = float(state.voice_steer)

                    target_left, target_right = mix_throttle_steer(
                        state.manual_throttle,
                        state.manual_steer,
                        speed_setting=state.speed_setting,
                        cfg=cfg,
                        full_speed=False,
                    )
                    if state.autonomous and remote_active:
                        last_mode = "server"

                last_targets = (float(target_left), float(target_right))

                now = time.time()
                dt = max(0.0, now - last_t)
                last_t = now
                if not raw_active:
                    drive.update(target_left=target_left, target_right=target_right, dt_s=dt)

                net_err = str(link.stats.last_error or "") if (link and not link.stats.connected) else ""
                error_messages.feed("client.network", net_err, now_s=now, prefix="[c]", priority=0, label="Network")
                error_messages.feed("client.camera", (getattr(cam, "last_error", "") if cam else ""), now_s=now, prefix="[c]", priority=0, label="Camera")
                error_messages.feed("client.stream", (str(rtsp_pub.last_error) if rtsp_pub else ""), now_s=now, prefix="[c]", priority=0, label="Stream")
                error_messages.feed("client.voice", (getattr(vc, "last_error", "") if vc else ""), now_s=now, prefix="[c]", priority=0, label="Voice")
                message, message_until = error_messages.next_message(now_s=now, current=message, current_until=message_until, ready="Ready.")

                if cam and cam.available and features.camera:
                    send_stream = bool(rtsp_pub and rtsp_pub.available and (now - last_stream_tx) >= stream_interval_s)
                    send_server = bool(link is not None and link.stats.connected and (now - last_server_frame_tx) >= server_frame_interval_s)
                    if send_stream or send_server:
                        jpeg = cam.read_jpeg()
                        if jpeg:
                            if send_stream and rtsp_pub and rtsp_pub.available:
                                rtsp_pub.push_jpeg(jpeg)
                                last_stream_tx = now
                            if send_server and link is not None:
                                try:
                                    b64 = base64.b64encode(jpeg).decode("ascii")
                                    link.send({"type": "frame", "ts": now, "jpeg_b64": b64})
                                    last_server_frame_tx = now
                                except Exception:
                                    pass
                            last_frame_tx = now
                            frames_sent += 1

                if link is not None and link.stats.connected and (now - last_telemetry_tx) >= 0.10:
                    try:
                        pi_stats_now = _pi_stats_label()
                        net_state = "[green]CONNECTED[/]" if link.stats.connected else "[red]DISCONNECTED[/]"
                        round_trip = f"{float(link.stats.rtt_ms):.1f} ms" if link.stats.rtt_ms is not None else "—"
                        receive_age = f"{int(now - link.stats.last_rx_ts)}s" if link.stats.last_rx_ts is not None else "—"
                        peer = _ellipsize(link.stats.peer or (f"{net.host}:{net.port}" if net else "—"), 28)

                        runtime_s = int(now - start_t)
                        cap_pct = int(round(state.max_speed_setting * 100))
                        try:
                            out_now = max(abs(m.last_speed) for m in motor_map.values()) if motor_map else 0.0
                        except Exception:
                            out_now = 0.0
                        out_pct = int(round(max(0.0, min(1.0, out_now)) * 100))

                        cam_enabled = bool(features.camera)
                        cam_available = bool(cam and cam.available)
                        cam_err_now = str(getattr(cam, "last_error", "") or "") if cam else ("initializing..." if cam_enabled else "")
                        rtsp_err_now = str(rtsp_pub.last_error or "") if rtsp_pub else ""
                        has_stream_now = bool(rtsp_pub and rtsp_pub.available)
                        rtsp_state_now = ui_on_off_error(
                            "on"
                            if (has_stream_now and not rtsp_err_now)
                            else ("error" if rtsp_err_now else "off")
                        )

                        motors_tlm: dict[str, Any] = {}
                        for n, m in motor_map.items():
                            motors_tlm[n] = {"speed": float(m.last_speed), "invert": bool(m.pins.invert)}

                        voice_info: dict[str, Any] = {
                            "enabled": bool(features.voice),
                            "available": bool(vc and vc.available),
                            "error": (str(getattr(vc, "last_error", "") or "—") if vc else "—"),
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
                            "error": (str(getattr(irc, "last_error", "") or "—") if irc else "—"),
                            "last_code": last_ir_code or "—",
                            "applied": (last_ir_cmd[0] if last_ir_cmd else "—"),
                            "age": age_s(last_ir_ts),
                        }

                        cam_info: dict[str, Any] = {
                            "enabled": cam_enabled,
                            "available": cam_available,
                            "camera_error": (str(getattr(cam, "last_error", "") or "—") if cam else "—"),
                            "rtsp_status": (
                                "on"
                                if (rtsp_pub and rtsp_pub.available and not rtsp_pub.last_error)
                                else ("error" if (rtsp_pub and rtsp_pub.last_error) else "off")
                            ),
                            "rtsp_error": (str(rtsp_pub.last_error) if rtsp_pub and rtsp_pub.last_error else "—"),
                            "rtsp_url": str(rtsp_public_url or "—"),
                            "frames_sent": int(frames_sent),
                            "last_frame_age": ("—" if not last_frame_tx else f"{int(now - last_frame_tx)}s"),
                        }

                        link.send(
                            {
                                "type": "telemetry",
                                "ts": now,
                                "network": {
                                    "client_ip": str(client_ip or "—"),
                                    "rtsp_url": str(rtsp_public_url or "—"),
                                    "error": str(net_err or "—"),
                                },
                                "ui_rows": {
                                    "status": [
                                        {"k": "Pi Stats", "v": pi_stats_now},
                                        {"k": "Run", "v": ("DRY" if dry_run else "REAL")},
                                        {"k": "Network", "v": net_state},
                                        {"k": "Client IP", "v": str(client_ip or "—")},
                                        {"k": "Round trip time", "v": round_trip},
                                        {"k": "Receive age", "v": receive_age},
                                        {"k": "Peer", "v": peer},
                                        {"k": f"Speed % (Cap {cap_pct}%):", "v": f"{out_pct}%"},
                                        {"k": "Uptime", "v": f"{runtime_s}s"},
                                    ],
                                    "autonomous": [
                                        {"k": "Mode", "v": ui_on_off(state.autonomous)},
                                        {"k": "Net", "v": ui_on_off(bool(link and link.stats.connected))},
                                        {"k": "Camera", "v": ui_on_off(bool(cam and cam.available))},
                                        {"k": "Last server obs", "v": str(last_obs or "—")},
                                        {"k": "Obs age", "v": age_s(last_obs_ts)},
                                        {"k": "Decision src", "v": str(last_mode)},
                                        {"k": "Manual input", "v": f"thr={state.manual_throttle:+.1f} steer={state.manual_steer:+.1f}"},
                                        {"k": "Targets", "v": f"L={last_targets[0]:+.2f}  R={last_targets[1]:+.2f}"},
                                        {"k": "Msg", "v": str(message)},
                                    ],
                                    "voice": [
                                        {"k": "Enabled", "v": ui_on_off(bool(features.voice))},
                                        {"k": "Available", "v": ui_on_off(bool(vc and vc.available))},
                                        {"k": "Voice err", "v": _ellipsize(str(voice_info.get("error", "—")), 30)},
                                        {"k": "Last words", "v": str(voice_info.get("last_words", "—"))},
                                        {"k": "Parsed cmd", "v": str(voice_info.get("parsed", "—"))},
                                        {"k": "Applied cmd", "v": str(voice_info.get("applied", "—"))},
                                        {"k": "Age", "v": str(voice_info.get("age", "—"))},
                                    ],
                                    "camera": [
                                        {"k": "Enabled", "v": ui_on_off(cam_enabled)},
                                        {"k": "Available", "v": ui_on_off(cam_available)},
                                        {"k": "Camera err", "v": _ellipsize(str(cam_err_now or "—"), 30)},
                                        {"k": "Stream", "v": rtsp_state_now},
                                        {"k": "Stream URL", "v": _ellipsize(str(rtsp_public_url or "—"), 30)},
                                        {"k": "Stream err", "v": _ellipsize(str(rtsp_err_now or "—"), 30)},
                                        {"k": "Frames sent", "v": str(frames_sent)},
                                        {"k": "Last frame", "v": ("—" if not last_frame_tx else f"{int(now - last_frame_tx)}s")},
                                    ],
                                    "ir": [
                                        {"k": "Enabled", "v": ui_on_off(bool(features.ir))},
                                        {"k": "Available", "v": ui_on_off(bool(irc and irc.available))},
                                        {"k": "IR err", "v": _ellipsize(str(ir_info.get("error", "—")), 30)},
                                        {"k": "Last code", "v": str(last_ir_code or "—")},
                                        {"k": "Applied cmd", "v": str(last_ir_cmd[0] if last_ir_cmd else "—")},
                                        {"k": "Age", "v": age_s(last_ir_ts)},
                                    ],
                                },
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

                if page_nav.page_index == 0:
                    layout["top"].update(top_panel())
                    layout["motors"].update(Panel(build_motor_table_from_motor_map(motor_map, now_s=time.time()), title="Motors", border_style="bright_green"))
                    layout["status"].update(status_panel())
                    layout["auto"].update(autonomous_panel(message))
                    layout["voice"].update(voice_panel())
                    layout["camera"].update(camera_panel())
                    layout["ir"].update(ir_panel())
                    layout["bottom"].update(Panel(Text(str(message)), border_style="dim"))
                    live.update(layout)
                else:
                    live.update(detection_page(time.time(), analysis_state, message))

                time.sleep(0.02)

        return 0
    finally:
        try:
            if fd is not None:
                _restore_terminal(fd, old_attrs)
        except Exception:
            pass
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
            if cam:
                cam.close()
        except Exception:
            pass
        try:
            if rtsp_pub:
                rtsp_pub.stop()
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
        _drain_pending_console_input()
        host = default_host
        try:
            t0 = time.time()
            entered = console.input(f"Server host [{default_host}]: ").strip()
            if not entered and (time.time() - t0) < 0.20:
                # If Enter was injected immediately, ask again once.
                entered = console.input(f"Server host [{default_host}]: ").strip()
            host = entered or default_host
        except Exception:
            host = default_host
        _drain_pending_console_input()
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
    console.print(Panel("Entering dashboard...\n[dim]Press Q to switch pages. Esc quits.[/]", border_style="bright_green"))
    time.sleep(0.6)

    return run_live_dashboard_tui(dry_run=dry_run, features=features, mic_index=mic_index, peak=peak, net=net_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
