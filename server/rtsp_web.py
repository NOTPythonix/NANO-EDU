from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


class RtspWebUi:
    """Server-side web UI that proxies client stream URLs into browser-viewable MJPEG."""

    def __init__(self, *, host: str, port: int, get_rtsp_url: Callable[[], str]):
        self._host = str(host)
        self._port = int(port)
        self._get_rtsp_url = get_rtsp_url

        self._stop = threading.Event()
        self._http: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._cap_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_ts: Optional[float] = None
        self._last_error: str = ""
        self._active_rtsp_url: str = ""

    @property
    def bound_url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    @property
    def last_error(self) -> str:
        with self._lock:
            return str(self._last_error)

    def start(self) -> None:
        self._stop.clear()

        self._cap_thread = threading.Thread(target=self._capture_loop, name="RtspWebUiCapture", daemon=True)
        self._cap_thread.start()

        handler_cls = self._make_handler_class()
        self._http = ThreadingHTTPServer((self._host, self._port), handler_cls)
        self._http_thread = threading.Thread(target=self._http.serve_forever, name="RtspWebUiHttp", daemon=True)
        self._http_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._http is not None:
            try:
                self._http.shutdown()
            except Exception:
                pass
            try:
                self._http.server_close()
            except Exception:
                pass
            self._http = None

    def _set_frame(self, jpeg: Optional[bytes]) -> None:
        with self._lock:
            self._latest_jpeg = jpeg
            self._latest_frame_ts = time.time() if jpeg else None

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._last_error = str(msg)

    def _set_active_url(self, url: str) -> None:
        with self._lock:
            self._active_rtsp_url = str(url)

    def _get_state(self) -> dict:
        with self._lock:
            now = time.time()
            age = (now - self._latest_frame_ts) if self._latest_frame_ts else None
            return {
                "rtsp_url": self._active_rtsp_url,
                "last_error": self._last_error,
                "frame_age_s": (None if age is None else round(max(0.0, age), 2)),
                "has_frame": self._latest_jpeg is not None,
            }

    def _get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def _capture_loop(self) -> None:
        cv2 = _try_import_cv2()
        if cv2 is None:
            self._set_error("OpenCV unavailable on server")
            return

        cap = None
        current_url = ""

        try:
            while not self._stop.is_set():
                desired = str(self._get_rtsp_url() or "").strip()
                low = desired.lower()
                if desired and not (low.startswith("rtsp://") or low.startswith("http://") or low.startswith("https://")):
                    desired = ""
                if desired != current_url:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                    current_url = desired
                    self._set_active_url(current_url)
                    self._set_frame(None)

                if not current_url:
                    self._set_error("Waiting for client stream URL")
                    time.sleep(0.2)
                    continue

                if cap is None:
                    try:
                        cap = cv2.VideoCapture(current_url)
                    except Exception:
                        cap = None
                    if cap is None or not cap.isOpened():
                        self._set_error("Failed to open client stream")
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                        time.sleep(0.4)
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                ok, frame = cap.read()
                if not ok or frame is None:
                    self._set_error("Stream frame read failed")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    self._set_frame(None)
                    time.sleep(0.2)
                    continue

                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not ok:
                    self._set_error("JPEG encode failed")
                    time.sleep(0.05)
                    continue

                self._set_error("")
                self._set_frame(bytes(enc.tobytes()))
                time.sleep(0.03)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _make_handler_class(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self._serve_index()
                    return
                if self.path == "/status":
                    self._serve_status()
                    return
                if self.path == "/mjpeg":
                    self._serve_mjpeg()
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

            def _serve_index(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>Robot Stream Viewer</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, Segoe UI, Arial; margin: 20px; background: #0b1220; color: #e6edf3; }
    .card { background: #121c2e; border: 1px solid #23314f; border-radius: 12px; padding: 14px; }
    .muted { color: #9fb0cd; }
    img { width: 100%; max-width: 960px; border-radius: 8px; border: 1px solid #2c3f63; }
    a { color: #8ec5ff; }
  </style>
</head>
<body>
  <h2>Robot Camera Viewer</h2>
  <div class=\"card\">
    <p class=\"muted\">This web UI runs on the server and displays the client stream feed via MJPEG proxy.</p>
    <p>Stream URL: <a id=\"rtsp\" href=\"#\" target=\"_blank\">(waiting)</a></p>
    <p>Status: <span id=\"status\">starting…</span></p>
    <img src=\"/mjpeg\" alt=\"Robot stream\" />
  </div>
  <script>
    async function refresh() {
      try {
        const r = await fetch('/status', { cache: 'no-store' });
        const s = await r.json();
        const text = s.last_error ? ('error: ' + s.last_error) : (s.has_frame ? ('ok, frame age ' + s.frame_age_s + 's') : 'waiting for frames');
        document.getElementById('status').textContent = text;
        const a = document.getElementById('rtsp');
        if (s.rtsp_url) {
          a.textContent = s.rtsp_url;
          a.href = s.rtsp_url;
        }
      } catch (_) {
        document.getElementById('status').textContent = 'status unavailable';
      }
    }
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""
                self.wfile.write(html.encode("utf-8"))

            def _serve_status(self):
                body = json.dumps(outer._get_state()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_mjpeg(self):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while not outer._stop.is_set():
                    jpeg = outer._get_latest_jpeg()
                    if jpeg is None:
                        time.sleep(0.1)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.06)
                    except Exception:
                        return

        return Handler
