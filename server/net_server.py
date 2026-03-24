from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ServerStats:
    listening: bool = False
    bound: str = ""
    connected: bool = False
    client_peer: str = ""
    last_rx_ts: Optional[float] = None
    last_tx_ts: Optional[float] = None
    last_error: str = ""
    rtt_ms: Optional[float] = None


class RobotSession:
    def __init__(self):
        self._lock = threading.Lock()
        self._in_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._out_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.latest_telemetry: dict[str, Any] = {}
        self.latest_hello: dict[str, Any] = {}
        self.latest_frame: dict[str, Any] = {}
        self.last_frame_ts: Optional[float] = None

    def poll(self) -> Optional[dict[str, Any]]:
        try:
            return self._in_q.get_nowait()
        except queue.Empty:
            return None

    def send(self, msg: dict[str, Any]) -> None:
        self._out_q.put(msg)


class JsonLineRobotServer:
    """Accepts exactly one robot connection at a time."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 8765):
        self._host = str(host)
        self._port = int(port)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        self.stats = ServerStats()
        self.session = RobotSession()

        self._pending_pings: dict[int, float] = {}
        self._last_ping_id = 0
        self._ping_interval_s = 0.2

    def start(self) -> None:
        self._stop.clear()
        t = threading.Thread(target=self._run, name="JsonLineRobotServer", daemon=True)
        t.start()
        self._thread = t

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass

    def _set_err(self, e: str) -> None:
        self.stats.last_error = str(e)

    def _touch_rx(self) -> None:
        self.stats.last_rx_ts = time.time()

    def _touch_tx(self) -> None:
        self.stats.last_tx_ts = time.time()

    def _run(self) -> None:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
            srv.listen(1)
            srv.settimeout(1.0)
            self._sock = srv
            self.stats.listening = True
            self.stats.bound = f"{self._host}:{self._port}"
        except Exception as e:
            self._set_err(str(e))
            self.stats.listening = False
            return

        while not self._stop.is_set():
            try:
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue

                self.stats.connected = True
                self.stats.client_peer = f"{addr[0]}:{addr[1]}"
                self.stats.rtt_ms = None

                rx_t = threading.Thread(target=self._rx_loop, args=(conn,), daemon=True)
                tx_t = threading.Thread(target=self._tx_loop, args=(conn,), daemon=True)
                rx_t.start()
                tx_t.start()

                # ping loop
                while not self._stop.is_set() and self.stats.connected:
                    self._last_ping_id += 1
                    ping_id = self._last_ping_id
                    self._pending_pings[ping_id] = time.time()
                    self.session.send({"type": "ping", "id": ping_id, "ts": time.time()})
                    time.sleep(self._ping_interval_s)

                try:
                    conn.close()
                except Exception:
                    pass
                self.stats.connected = False
                self.stats.client_peer = ""

            except Exception as e:
                self._set_err(str(e))
                self.stats.connected = False

    def _rx_loop(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = b""
        while not self._stop.is_set():
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    self._touch_rx()
                    t = msg.get("type")
                    if t == "pong":
                        ping_id = msg.get("id")
                        try:
                            ping_id_i = int(ping_id)
                        except Exception:
                            ping_id_i = -1
                        t0 = self._pending_pings.pop(ping_id_i, None)
                        if t0 is not None:
                            self.stats.rtt_ms = (time.time() - t0) * 1000.0
                        continue
                    if t == "ping":
                        try:
                            self.session.send({"type": "pong", "id": int(msg.get("id")), "ts": time.time()})
                        except Exception:
                            pass
                        continue
                    if t == "hello":
                        self.session.latest_hello = msg
                    if t == "telemetry":
                        self.session.latest_telemetry = msg
                    if t == "frame":
                        self.session.latest_frame = msg
                        self.session.last_frame_ts = time.time()
                    self.session._in_q.put(msg)
            except socket.timeout:
                continue
            except Exception as e:
                self._set_err(str(e))
                break

        self.stats.connected = False

    def _tx_loop(self, conn: socket.socket) -> None:
        while not self._stop.is_set() and self.stats.connected:
            try:
                msg = self.session._out_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                raw = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
                conn.sendall(raw)
                self._touch_tx()
            except Exception as e:
                self._set_err(str(e))
                self.stats.connected = False
                break
