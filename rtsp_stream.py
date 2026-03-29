from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


@dataclass
class RtspStreamConfig:
    host: str = "0.0.0.0"
    mjpeg_port: int = 8091


class RtspStreamPublisher:
    """Publishes JPEG frames over a low-latency MJPEG HTTP endpoint."""

    def __init__(self, cfg: RtspStreamConfig):
        self._cfg = cfg
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame_event = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_id = 0
        self._http: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self.last_error: str = ""

    @property
    def available(self) -> bool:
        return self._http is not None

    @property
    def mode(self) -> str:
        return "mjpeg"

    def endpoint_url(self, public_host: str) -> str:
        return f"http://{public_host}:{int(self._cfg.mjpeg_port)}/mjpeg"

    def start(self) -> bool:
        if self.available:
            return True

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path in ("/", ""):
                    body = (
                        b"<!doctype html><html><head><meta charset='utf-8'>"
                        b"<meta http-equiv='Cache-Control' content='no-store'></head>"
                        b"<body style='margin:0;background:#000'><img src='/mjpeg' style='width:100%%;height:auto' /></body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != "/mjpeg":
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                with outer._lock:
                    last_seen_frame = outer._latest_frame_id

                while not outer._stop.is_set():
                    with outer._lock:
                        outer._frame_event.wait_for(
                            lambda: outer._stop.is_set() or outer._latest_frame_id != last_seen_frame,
                            timeout=1.0,
                        )
                        if outer._stop.is_set():
                            return

                        jpeg = outer._latest_jpeg
                        current_frame = outer._latest_frame_id
                        if jpeg is None or current_frame == last_seen_frame:
                            continue
                        last_seen_frame = current_frame

                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except Exception:
                        return

        try:
            http = ThreadingHTTPServer((str(self._cfg.host), int(self._cfg.mjpeg_port)), Handler)
        except Exception as e:
            self.last_error = str(e)
            self._http = None
            return False

        self._stop.clear()
        self._http = http
        self._http_thread = threading.Thread(target=http.serve_forever, name="MjpegStreamHttp", daemon=True)
        self._http_thread.start()
        self.last_error = ""
        return True

    def push_jpeg(self, jpeg: bytes) -> None:
        if not jpeg:
            return

        with self._lock:
            self._latest_jpeg = bytes(jpeg)
            self._latest_frame_id += 1
            self._frame_event.notify_all()

    def stop(self) -> None:
        self._stop.set()

        with self._lock:
            self._frame_event.notify_all()

        http = self._http
        self._http = None
        if http is not None:
            try:
                http.shutdown()
            except Exception:
                pass
            try:
                http.server_close()
            except Exception:
                pass
