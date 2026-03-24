from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LinkStats:
    connected: bool = False
    peer: str = ""
    last_rx_ts: Optional[float] = None
    last_tx_ts: Optional[float] = None
    last_error: str = ""
    rtt_ms: Optional[float] = None


class JsonLineLink:
    """Simple TCP link with newline-delimited JSON messages.

    Designed to be used from a non-async TUI loop.
    """

    def __init__(self, *, host: str, port: int, role: str, name: str, connect_timeout_s: float = 3.0):
        self._host = str(host)
        self._port = int(port)
        self._role = str(role)
        self._name = str(name)
        self._connect_timeout_s = float(connect_timeout_s)

        self._sock: Optional[socket.socket] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._out_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._in_q: "queue.Queue[dict[str, Any]]" = queue.Queue()

        self.stats = LinkStats()
        self._lock = threading.Lock()

        self._last_ping_id = 0
        self._pending_pings: dict[int, float] = {}
        self._ping_interval_s = 0.2

    def start(self) -> None:
        self._stop.clear()
        t = threading.Thread(target=self._run, name=f"JsonLineLink({self._role})", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._close_socket()
        except Exception:
            pass

    def send(self, msg: dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            raise TypeError("msg must be a dict")
        self._out_q.put(msg)

    def poll(self) -> Optional[dict[str, Any]]:
        try:
            return self._in_q.get_nowait()
        except queue.Empty:
            return None

    def _set_error(self, err: str) -> None:
        with self._lock:
            self.stats.last_error = str(err)

    def _set_connected(self, connected: bool, peer: str = "") -> None:
        with self._lock:
            self.stats.connected = bool(connected)
            self.stats.peer = str(peer)
            if not connected:
                self.stats.rtt_ms = None

    def _touch_rx(self) -> None:
        with self._lock:
            self.stats.last_rx_ts = time.time()

    def _touch_tx(self) -> None:
        with self._lock:
            self.stats.last_tx_ts = time.time()

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self._connect_timeout_s)
        s.connect((self._host, self._port))
        s.settimeout(1.0)
        return s

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._set_error("")
                sock = self._connect()
                self._sock = sock
                peer = ""
                try:
                    peer = f"{sock.getpeername()[0]}:{sock.getpeername()[1]}"
                except Exception:
                    peer = f"{self._host}:{self._port}"
                self._set_connected(True, peer)

                self._rx_thread = threading.Thread(target=self._rx_loop, name=f"JsonLineLink-rx({self._role})", daemon=True)
                self._tx_thread = threading.Thread(target=self._tx_loop, name=f"JsonLineLink-tx({self._role})", daemon=True)
                self._rx_thread.start()
                self._tx_thread.start()

                # Hello
                self.send({"type": "hello", "role": self._role, "name": self._name, "ts": time.time()})

                # Ping loop (this thread)
                while not self._stop.is_set() and self.stats.connected:
                    self._last_ping_id += 1
                    ping_id = self._last_ping_id
                    self._pending_pings[ping_id] = time.time()
                    self.send({"type": "ping", "id": ping_id, "ts": time.time()})
                    time.sleep(self._ping_interval_s)

                # connection dropped
                self._close_socket()
                self._set_connected(False)

            except Exception as e:
                self._set_error(str(e))
                self._set_connected(False)
                try:
                    self._close_socket()
                except Exception:
                    pass
                time.sleep(backoff)
                backoff = min(5.0, backoff * 1.5)
                continue

            backoff = 0.5

    def _rx_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        buf = b""
        while not self._stop.is_set():
            try:
                data = sock.recv(4096)
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
                    if isinstance(msg, dict):
                        self._handle_rx(msg)
            except socket.timeout:
                continue
            except Exception as e:
                self._set_error(str(e))
                break

        self._set_connected(False)

    def _handle_rx(self, msg: dict[str, Any]) -> None:
        self._touch_rx()
        t = msg.get("type")
        if t == "ping":
            # Reply
            ping_id = msg.get("id")
            try:
                self.send({"type": "pong", "id": int(ping_id), "ts": time.time()})
            except Exception:
                pass
            return
        if t == "pong":
            ping_id = msg.get("id")
            try:
                ping_id_i = int(ping_id)
            except Exception:
                ping_id_i = -1
            t0 = self._pending_pings.pop(ping_id_i, None)
            if t0 is not None:
                rtt = (time.time() - t0) * 1000.0
                with self._lock:
                    self.stats.rtt_ms = float(rtt)
            return

        self._in_q.put(msg)

    def _tx_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        while not self._stop.is_set():
            try:
                msg = self._out_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                raw = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
                sock.sendall(raw)
                self._touch_tx()
            except Exception as e:
                self._set_error(str(e))
                self._set_connected(False)
                break
