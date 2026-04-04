from __future__ import annotations

import os
import json
import smtplib
import threading
import time
from collections import deque
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

from server.inference import analyze_frame_from_bgr, draw_detections, get_model_error, phone_holder_crop_from_detections


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value
    except Exception:
        pass


_load_env_file()


PHONE_ALERT_TO_EMAIL = os.environ.get("ROBOT_ALERT_TO_EMAIL", "")
PHONE_ALERT_SMTP_HOST = os.environ.get("ROBOT_ALERT_SMTP_HOST", "smtp.gmail.com")
PHONE_ALERT_SMTP_PORT = int(os.environ.get("ROBOT_ALERT_SMTP_PORT", "587"))
PHONE_ALERT_SMTP_USERNAME = os.environ.get("ROBOT_ALERT_SMTP_USERNAME", "")
PHONE_ALERT_SMTP_PASSWORD = os.environ.get("ROBOT_ALERT_SMTP_PASSWORD", "")
PHONE_ALERT_FROM_EMAIL = os.environ.get("ROBOT_ALERT_FROM_EMAIL", PHONE_ALERT_SMTP_USERNAME or PHONE_ALERT_TO_EMAIL)
PHONE_ALERT_USE_TLS = str(os.environ.get("ROBOT_ALERT_SMTP_TLS", "1")).strip().lower() not in ("0", "false", "no")
PHONE_ALERT_COOLDOWN_S = max(5.0, float(os.environ.get("ROBOT_PHONE_ALERT_COOLDOWN_S", "60")))


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
        self._phone_alert_last_sent_ts: float = 0.0
        self._latest_frame_bgr = None
        self._latest_frame_seq: int = 0
        self._last_analysed_seq: int = 0
        self._latest_detections = []
        self._latest_analysis: Optional[dict] = None
        self._latest_analysis_ts: Optional[float] = None
        self._analysis_duration_samples: deque[float] = deque(maxlen=5)
        self._analysis_interval_s: float = 1.0 / 30.0
        self._phone_last_seen_ts: Optional[float] = None
        self._phone_last_triggered_ts: Optional[float] = None

    @property
    def bound_url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    @property
    def last_error(self) -> str:
        with self._lock:
            return str(self._last_error)

    def start(self) -> None:
        self._stop.clear()

        self._analysis_thread = threading.Thread(target=self._analysis_loop, name="RtspWebUiAnalysis", daemon=True)
        self._analysis_thread.start()

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

    def _set_latest_frame(self, frame, *, seq: int) -> None:
        with self._lock:
            self._latest_frame_bgr = None if frame is None else frame.copy()
            self._latest_frame_seq = int(seq)

    def _snapshot_latest_frame(self):
        with self._lock:
            if self._latest_frame_bgr is None:
                return None, int(self._latest_frame_seq)
            return self._latest_frame_bgr.copy(), int(self._latest_frame_seq)

    def _set_analysis(self, *, analysis, seq: int, error: str = "", phone_seen: bool = False, phone_triggered: bool = False, phone_triggered_ts: Optional[float] = None) -> None:
        now = time.time()
        if phone_seen:
            self._phone_last_seen_ts = now
        if phone_triggered:
            self._phone_last_triggered_ts = float(phone_triggered_ts or now)

        detections = list(getattr(analysis, "detections", []) or [])
        detections_sorted = sorted(detections, key=lambda det: (-float(getattr(det, "confidence", 0.0) or 0.0), str(getattr(det, "label", ""))))
        all_detections = [
            {
                "label": str(getattr(det, "label", "—")),
                "confidence": round(float(getattr(det, "confidence", 0.0) or 0.0), 3),
                "x1": int(getattr(det, "x1", 0) or 0),
                "y1": int(getattr(det, "y1", 0) or 0),
                "x2": int(getattr(det, "x2", 0) or 0),
                "y2": int(getattr(det, "y2", 0) or 0),
                "cls_index": int(getattr(det, "cls_index", 0) or 0),
            }
            for det in detections_sorted
        ]

        analysis_fps = 1.0 / self._analysis_interval_s if self._analysis_interval_s > 0 else 0.0

        with self._lock:
            self._latest_detections = detections
            self._latest_analysis = {
                "obstacle": str(getattr(analysis, "obstacle", "unknown")),
                "label": str(getattr(analysis, "label", "—")),
                "confidence": float(getattr(analysis, "confidence", 0.0) or 0.0),
                "throttle": float(getattr(analysis, "throttle", 0.0) or 0.0),
                "steer": float(getattr(analysis, "steer", 0.0) or 0.0),
                "error": str(error or getattr(analysis, "error", "") or ""),
                "seq": int(seq),
                "updated_ts": now,
                "detections": len(self._latest_detections),
                "raw_detections": len(detections),
                "analysis_fps": round(analysis_fps, 2),
                "analysis_interval_s": round(self._analysis_interval_s, 3),
                "top_detections": all_detections[:8],
                "all_detections": all_detections,
                "phone_detected": bool(phone_seen),
                "phone_detected_ts": self._phone_last_seen_ts,
                "phone_alert_triggered": bool(phone_triggered),
                "phone_alert_triggered_ts": self._phone_last_triggered_ts,
            }
            self._latest_analysis_ts = now
            self._last_analysed_seq = int(seq)

    def get_latest_analysis(self) -> dict:
        with self._lock:
            data = dict(self._latest_analysis or {})
            if self._latest_analysis_ts is not None:
                data["age_s"] = round(max(0.0, time.time() - self._latest_analysis_ts), 2)
            else:
                data["age_s"] = None
            if self._phone_last_seen_ts is not None:
                data["phone_detected_age_s"] = round(max(0.0, time.time() - self._phone_last_seen_ts), 2)
            else:
                data["phone_detected_age_s"] = None
            if self._phone_last_triggered_ts is not None:
                data["phone_alert_age_s"] = round(max(0.0, time.time() - self._phone_last_triggered_ts), 2)
            else:
                data["phone_alert_age_s"] = None
            data["model_error"] = get_model_error()
            return data

    def _analysis_loop(self) -> None:
        cv2 = _try_import_cv2()
        if cv2 is None:
            self._set_error("OpenCV unavailable on server")
            return

        while not self._stop.is_set():
            loop_started = time.perf_counter()
            frame, seq = self._snapshot_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            analysis = analyze_frame_from_bgr(frame)
            phone_crop = phone_holder_crop_from_detections(frame, getattr(analysis, "detections", []))
            phone_seen = phone_crop is not None
            phone_triggered = False

            if phone_seen and self._should_send_phone_alert(time.time()):
                phone_triggered = True
                self._mark_phone_alert_sent(time.time())
                ok_phone, phone_enc = cv2.imencode(".jpg", phone_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_phone and phone_enc is not None:
                    phone_jpeg = bytes(phone_enc.tobytes())

                    def _send_async(payload: bytes) -> None:
                        self._send_phone_alert_email(
                            payload,
                            subject="Phone detected on robot camera",
                            body="A phone was detected in the robot camera feed. An image is attached.",
                        )

                    threading.Thread(target=_send_async, args=(phone_jpeg,), name="PhoneAlertEmail", daemon=True).start()

            self._set_analysis(
                analysis=analysis,
                seq=seq,
                phone_seen=phone_seen,
                phone_triggered=phone_triggered,
                phone_triggered_ts=time.time() if phone_triggered else None,
            )
            analysis_duration = max(0.0, time.perf_counter() - loop_started)
            self._analysis_duration_samples.append(analysis_duration)
            avg_duration = sum(self._analysis_duration_samples) / len(self._analysis_duration_samples)
            target_interval = max(1.0 / 30.0, avg_duration)
            self._analysis_interval_s = target_interval

            sleep_for = max(0.0, target_interval - analysis_duration)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _should_send_phone_alert(self, now: float) -> bool:
        with self._lock:
            return (now - self._phone_alert_last_sent_ts) >= PHONE_ALERT_COOLDOWN_S

    def _mark_phone_alert_sent(self, now: float) -> None:
        with self._lock:
            self._phone_alert_last_sent_ts = float(now)

    def _send_phone_alert_email(self, image_bytes: bytes, *, subject: str, body: str) -> None:
        if not PHONE_ALERT_TO_EMAIL or PHONE_ALERT_TO_EMAIL.startswith("replace-this-with-a-recipient"):
            return
        if not PHONE_ALERT_SMTP_USERNAME or not PHONE_ALERT_SMTP_PASSWORD:
            self._set_error("phone alert skipped: SMTP credentials not configured")
            return

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = PHONE_ALERT_FROM_EMAIL
        msg["To"] = PHONE_ALERT_TO_EMAIL
        msg.set_content(body)
        msg.add_attachment(image_bytes, maintype="image", subtype="jpeg", filename="phone-alert.jpg")

        try:
            if PHONE_ALERT_SMTP_PORT == 465:
                smtp = smtplib.SMTP_SSL(PHONE_ALERT_SMTP_HOST, PHONE_ALERT_SMTP_PORT, timeout=15)
            else:
                smtp = smtplib.SMTP(PHONE_ALERT_SMTP_HOST, PHONE_ALERT_SMTP_PORT, timeout=15)
            with smtp:
                smtp.ehlo()
                if PHONE_ALERT_USE_TLS and PHONE_ALERT_SMTP_PORT != 465:
                    smtp.starttls()
                    smtp.ehlo()
                smtp.login(PHONE_ALERT_SMTP_USERNAME, PHONE_ALERT_SMTP_PASSWORD)
                smtp.send_message(msg)
        except Exception as exc:
            self._set_error(f"phone alert email failed: {exc}")

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

                self._set_latest_frame(frame, seq=self._latest_frame_seq + 1)

                with self._lock:
                    detections = list(self._latest_detections)
                annotated = draw_detections(frame, detections)

                ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
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
                if self.path == "/analysis":
                    self._serve_analysis()
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
                const statusResp = await fetch('/status', { cache: 'no-store' });
                const s = await statusResp.json();
                const text = s.last_error ? ('error: ' + s.last_error) : (s.has_frame ? ('ok, frame age ' + s.frame_age_s + 's') : 'waiting for frames');
                document.getElementById('status').textContent = text;
                const rtspLink = document.getElementById('rtsp');
                if (s.rtsp_url) {
                    rtspLink.textContent = s.rtsp_url;
                    rtspLink.href = s.rtsp_url;
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

            def _serve_analysis(self):
                body = json.dumps(outer.get_latest_analysis()).encode("utf-8")
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
