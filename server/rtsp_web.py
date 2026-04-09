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

from server.inference import (
    analyze_frame_from_bgr,
    detect_uniform_compliance_from_detections,
    draw_detections,
    get_model_error,
    get_model_path,
    get_prompt_classes,
    phone_holder_crop_from_detections,
)


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
BADGE_ALERT_COOLDOWN_S = max(5.0, float(os.environ.get("ROBOT_BADGE_ALERT_COOLDOWN_S", "90")))
UNIFORM_ALERT_COOLDOWN_S = max(5.0, float(os.environ.get("ROBOT_UNIFORM_ALERT_COOLDOWN_S", "90")))
MISSING_COMPLIANCE_HOLD_S = 5.0


def _label_seen(detections, hints: tuple[str, ...]) -> bool:
    for det in detections or []:
        label = str(getattr(det, "label", "") or "").strip().lower()
        if not label:
            continue
        for hint in hints:
            if hint in label:
                return True
    return False


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


class RtspWebUi:
    """Server-side web UI that proxies client stream URLs into browser-viewable MJPEG."""

    def __init__(self, *, host: str, port: int, get_rtsp_url: Callable[[], str], alert_emails_enabled: bool = True):
        self._host = str(host)
        self._port = int(port)
        self._get_rtsp_url = get_rtsp_url
        self._alert_emails_enabled = bool(alert_emails_enabled)

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
        self._badge_alert_last_sent_ts: float = 0.0
        self._uniform_alert_last_sent_ts: float = 0.0
        self._latest_frame_bgr = None
        self._latest_frame_seq: int = 0
        self._last_analysed_seq: int = 0
        self._latest_detections = []
        self._latest_analysis: Optional[dict] = None
        self._latest_analysis_ts: Optional[float] = None
        self._analysis_duration_samples: deque[float] = deque(maxlen=5)
        self._analysis_target_fps: float = max(5.0, min(60.0, float(os.environ.get("ROBOT_ANALYSIS_TARGET_FPS", "30"))))
        self._analysis_min_interval_s: float = 1.0 / self._analysis_target_fps
        self._analysis_interval_s: float = self._analysis_min_interval_s
        self._analysis_warmup_remaining: int = max(0, int(os.environ.get("ROBOT_ANALYSIS_WARMUP_FRAMES", "20")))
        self._analysis_ema_alpha: float = max(0.05, min(0.90, float(os.environ.get("ROBOT_ANALYSIS_ADAPT_ALPHA", "0.35"))))
        self._analysis_ema_duration: Optional[float] = None
        self._phone_last_seen_ts: Optional[float] = None
        self._badge_last_seen_ts: Optional[float] = None
        self._wordmark_last_seen_ts: Optional[float] = None
        self._phone_last_triggered_ts: Optional[float] = None
        self._badge_last_triggered_ts: Optional[float] = None
        self._uniform_last_triggered_ts: Optional[float] = None
        self._badge_missing_since_ts: Optional[float] = None
        self._uniform_missing_since_ts: Optional[float] = None

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

    def _set_analysis(
        self,
        *,
        analysis,
        seq: int,
        error: str = "",
        phone_seen: bool = False,
        phone_triggered: bool = False,
        phone_triggered_ts: Optional[float] = None,
        badge_alert_triggered: bool = False,
        badge_alert_triggered_ts: Optional[float] = None,
        uniform_alert_triggered: bool = False,
        uniform_alert_triggered_ts: Optional[float] = None,
        badge_seen: bool = False,
        wordmark_seen: bool = False,
        has_medium_person: bool = False,
        has_close_person: bool = False,
        missing_badge: bool = False,
        missing_wordmark: bool = False,
        badge_missing_for_s: float = 0.0,
        uniform_missing_for_s: float = 0.0,
        badge_hold_met: bool = False,
        uniform_hold_met: bool = False,
    ) -> None:
        now = time.time()
        if phone_seen:
            self._phone_last_seen_ts = now
        if badge_seen:
            self._badge_last_seen_ts = now
        if wordmark_seen:
            self._wordmark_last_seen_ts = now
        if phone_triggered:
            self._phone_last_triggered_ts = float(phone_triggered_ts or now)
        if badge_alert_triggered:
            self._badge_last_triggered_ts = float(badge_alert_triggered_ts or now)
        if uniform_alert_triggered:
            self._uniform_last_triggered_ts = float(uniform_alert_triggered_ts or now)

        detections = list(getattr(analysis, "detections", []) or [])
        detections_sorted = sorted(detections, key=lambda det: (-float(getattr(det, "confidence", 0.0) or 0.0), str(getattr(det, "label", ""))))
        all_detections = [
            {
                "label": str(getattr(det, "label", "ΓÇö")),
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
                "label": str(getattr(analysis, "label", "ΓÇö")),
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
                "badge_detected": bool(badge_seen),
                "badge_detected_ts": self._badge_last_seen_ts,
                "wordmark_detected": bool(wordmark_seen),
                "wordmark_detected_ts": self._wordmark_last_seen_ts,
                "person_medium_close": bool(has_medium_person),
                "person_close": bool(has_close_person),
                "missing_id_badge": bool(missing_badge),
                "missing_uniform_wordmark": bool(missing_wordmark),
                "missing_badge_for_s": round(max(0.0, float(badge_missing_for_s or 0.0)), 2),
                "missing_wordmark_for_s": round(max(0.0, float(uniform_missing_for_s or 0.0)), 2),
                "badge_hold_met": bool(badge_hold_met),
                "uniform_hold_met": bool(uniform_hold_met),
                "badge_alert_triggered": bool(badge_alert_triggered),
                "badge_alert_triggered_ts": self._badge_last_triggered_ts,
                "uniform_alert_triggered": bool(uniform_alert_triggered),
                "uniform_alert_triggered_ts": self._uniform_last_triggered_ts,
                "model_path": get_model_path(),
                "prompt_class_count": len(get_prompt_classes()),
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
            if self._badge_last_seen_ts is not None:
                data["badge_detected_age_s"] = round(max(0.0, time.time() - self._badge_last_seen_ts), 2)
            else:
                data["badge_detected_age_s"] = None
            if self._wordmark_last_seen_ts is not None:
                data["wordmark_detected_age_s"] = round(max(0.0, time.time() - self._wordmark_last_seen_ts), 2)
            else:
                data["wordmark_detected_age_s"] = None
            if self._phone_last_triggered_ts is not None:
                data["phone_alert_age_s"] = round(max(0.0, time.time() - self._phone_last_triggered_ts), 2)
            else:
                data["phone_alert_age_s"] = None
            if self._badge_last_triggered_ts is not None:
                data["badge_alert_age_s"] = round(max(0.0, time.time() - self._badge_last_triggered_ts), 2)
            else:
                data["badge_alert_age_s"] = None
            if self._uniform_last_triggered_ts is not None:
                data["uniform_alert_age_s"] = round(max(0.0, time.time() - self._uniform_last_triggered_ts), 2)
            else:
                data["uniform_alert_age_s"] = None
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
            detections = list(getattr(analysis, "detections", []) or [])
            phone_crop = phone_holder_crop_from_detections(frame, getattr(analysis, "detections", []))
            compliance = detect_uniform_compliance_from_detections(frame, getattr(analysis, "detections", []))
            phone_seen = phone_crop is not None
            badge_seen = _label_seen(
                detections,
                ("badge", "id badge", "name badge", "lanyard"),
            )
            wordmark_seen = _label_seen(
                detections,
                ("wordmark", "school logo", "school emblem", "harmony logo", "harmony public schools text logo", "logo"),
            )
            phone_triggered = False
            badge_triggered = False
            uniform_triggered = False
            now = time.time()

            missing_badge_now = bool(compliance.get("missing_badge", False)) and bool(compliance.get("has_medium_person", False))
            if missing_badge_now:
                if self._badge_missing_since_ts is None:
                    self._badge_missing_since_ts = now
                badge_missing_for_s = max(0.0, now - self._badge_missing_since_ts)
            else:
                self._badge_missing_since_ts = None
                badge_missing_for_s = 0.0
            badge_hold_met = missing_badge_now and (badge_missing_for_s >= MISSING_COMPLIANCE_HOLD_S)

            missing_wordmark_now = bool(compliance.get("missing_wordmark", False)) and bool(compliance.get("has_close_person", False))
            if missing_wordmark_now:
                if self._uniform_missing_since_ts is None:
                    self._uniform_missing_since_ts = now
                uniform_missing_for_s = max(0.0, now - self._uniform_missing_since_ts)
            else:
                self._uniform_missing_since_ts = None
                uniform_missing_for_s = 0.0
            uniform_hold_met = missing_wordmark_now and (uniform_missing_for_s >= MISSING_COMPLIANCE_HOLD_S)

            if self._alert_emails_enabled and phone_seen and self._should_send_phone_alert(now):
                phone_triggered = True
                self._mark_phone_alert_sent(now)
                ok_phone, phone_enc = cv2.imencode(".jpg", phone_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_phone and phone_enc is not None:
                    phone_jpeg = bytes(phone_enc.tobytes())

                    def _send_async(payload: bytes) -> None:
                        self._send_alert_email(
                            payload,
                            subject="Phone detected on robot camera",
                            body="A phone was detected in the robot camera feed. An image is attached.",
                            attachment_name="phone-alert.jpg",
                        )

                    threading.Thread(target=_send_async, args=(phone_jpeg,), name="PhoneAlertEmail", daemon=True).start()

            if self._alert_emails_enabled and badge_hold_met and self._should_send_badge_alert(now):
                badge_triggered = True
                self._mark_badge_alert_sent(now)
                badge_crop = compliance.get("missing_badge_crop")
                if badge_crop is None:
                    badge_crop = frame
                ok_badge, badge_enc = cv2.imencode(".jpg", badge_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_badge and badge_enc is not None:
                    badge_jpeg = bytes(badge_enc.tobytes())

                    def _send_badge_async(payload: bytes) -> None:
                        self._send_alert_email(
                            payload,
                            subject="ID badge missing (medium-close person)",
                            body="A medium-close person was detected without an ID badge. An image is attached.",
                            attachment_name="missing-badge.jpg",
                        )

                    threading.Thread(target=_send_badge_async, args=(badge_jpeg,), name="BadgeAlertEmail", daemon=True).start()

            if self._alert_emails_enabled and uniform_hold_met and self._should_send_uniform_alert(now):
                uniform_triggered = True
                self._mark_uniform_alert_sent(now)
                uniform_crop = compliance.get("missing_wordmark_crop")
                if uniform_crop is None:
                    uniform_crop = frame
                ok_uni, uni_enc = cv2.imencode(".jpg", uniform_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_uni and uni_enc is not None:
                    uniform_jpeg = bytes(uni_enc.tobytes())

                    def _send_uniform_async(payload: bytes) -> None:
                        self._send_alert_email(
                            payload,
                            subject="Uniform wordmark missing (close person)",
                            body='A close person was detected without the expected "school wordmark on uniform". An image is attached.',
                            attachment_name="missing-wordmark.jpg",
                        )

                    threading.Thread(target=_send_uniform_async, args=(uniform_jpeg,), name="UniformAlertEmail", daemon=True).start()

            self._set_analysis(
                analysis=analysis,
                seq=seq,
                phone_seen=phone_seen,
                phone_triggered=phone_triggered,
                phone_triggered_ts=time.time() if phone_triggered else None,
                badge_alert_triggered=badge_triggered,
                badge_alert_triggered_ts=time.time() if badge_triggered else None,
                uniform_alert_triggered=uniform_triggered,
                uniform_alert_triggered_ts=time.time() if uniform_triggered else None,
                badge_seen=badge_seen,
                wordmark_seen=wordmark_seen,
                has_medium_person=bool(compliance.get("has_medium_person", False)),
                has_close_person=bool(compliance.get("has_close_person", False)),
                missing_badge=bool(compliance.get("missing_badge", False)),
                missing_wordmark=bool(compliance.get("missing_wordmark", False)),
                badge_missing_for_s=badge_missing_for_s,
                uniform_missing_for_s=uniform_missing_for_s,
                badge_hold_met=badge_hold_met,
                uniform_hold_met=uniform_hold_met,
            )
            analysis_duration = max(0.0, time.perf_counter() - loop_started)
            if self._analysis_warmup_remaining > 0:
                # Ignore startup jitters/compilation overhead when learning pacing.
                self._analysis_warmup_remaining -= 1
                target_interval = self._analysis_min_interval_s
                self._analysis_interval_s = target_interval
            else:
                self._analysis_duration_samples.append(analysis_duration)
                if self._analysis_ema_duration is None:
                    self._analysis_ema_duration = analysis_duration
                else:
                    a = self._analysis_ema_alpha
                    self._analysis_ema_duration = (1.0 - a) * self._analysis_ema_duration + a * analysis_duration

                # Use EMA for stability, but cap with recent minimum to recover quickly
                # after warm-up spikes.
                ema_duration = float(self._analysis_ema_duration or analysis_duration)
                recent_min = min(self._analysis_duration_samples) if self._analysis_duration_samples else analysis_duration
                learned_duration = min(ema_duration, recent_min * 1.25)
                target_interval = max(self._analysis_min_interval_s, learned_duration)
                self._analysis_interval_s = target_interval

            sleep_for = max(0.0, target_interval - analysis_duration)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _should_send_phone_alert(self, now: float) -> bool:
        with self._lock:
            return (now - self._phone_alert_last_sent_ts) >= PHONE_ALERT_COOLDOWN_S

    def _should_send_badge_alert(self, now: float) -> bool:
        with self._lock:
            return (now - self._badge_alert_last_sent_ts) >= BADGE_ALERT_COOLDOWN_S

    def _should_send_uniform_alert(self, now: float) -> bool:
        with self._lock:
            return (now - self._uniform_alert_last_sent_ts) >= UNIFORM_ALERT_COOLDOWN_S

    def _mark_phone_alert_sent(self, now: float) -> None:
        with self._lock:
            self._phone_alert_last_sent_ts = float(now)

    def _mark_badge_alert_sent(self, now: float) -> None:
        with self._lock:
            self._badge_alert_last_sent_ts = float(now)

    def _mark_uniform_alert_sent(self, now: float) -> None:
        with self._lock:
            self._uniform_alert_last_sent_ts = float(now)

    def _send_alert_email(self, image_bytes: bytes, *, subject: str, body: str, attachment_name: str) -> None:
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
        msg.add_attachment(image_bytes, maintype="image", subtype="jpeg", filename=str(attachment_name or "alert.jpg"))

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
    <title>NANO-EDU Vision Terminal</title>
  <style>
        :root {
            --bg: #0b0f0c;
            --panel: #111612;
            --line: #2a3a2d;
            --text: #d9f7dc;
            --muted: #8bb28f;
            --ok: #7bff8a;
            --warn: #ffe58a;
            --bad: #ff8f8f;
            --accent: #85ffbe;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: Consolas, "Lucida Console", "Courier New", monospace;
            padding: 14px;
        }
        .app {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            gap: 12px;
        }
        .hero,
        .panel,
        .status-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 0;
            box-shadow: none;
        }
        .hero {
            padding: 10px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        .title { margin: 0; font-size: 1.1rem; letter-spacing: 0.04em; }
        .subtitle { margin: 3px 0 0; color: var(--muted); font-size: 0.86rem; }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid var(--line);
            padding: 4px 8px;
            color: var(--muted);
            background: #0e140f;
            font-size: 0.82rem;
        }
        .dot { width: 8px; height: 8px; background: var(--warn); }
        .dot.ok { background: var(--ok); }
        .dot.bad { background: var(--bad); }
        .grid {
            display: grid;
            grid-template-columns: minmax(300px, 1.7fr) minmax(250px, 1fr);
            gap: 12px;
        }
        .panel { padding: 10px; }
        .video-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            color: var(--muted);
            font-size: 0.82rem;
        }
        .btn {
            border: 1px solid var(--line);
            background: #0f1711;
            color: var(--accent);
            font-family: inherit;
            font-size: 0.78rem;
            padding: 5px 8px;
            cursor: pointer;
        }
        .btn:hover { background: #162119; }
        .video-shell {
            border: 1px solid var(--line);
            background: #000;
            width: 100%;
            aspect-ratio: 16/9;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-shell:fullscreen {
            width: 100vw;
            height: 100vh;
            aspect-ratio: auto;
            margin: 0;
            border: none;
            background: #000;
        }
        .video-shell:fullscreen .feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .feed {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: contain;
        }
        .meta {
            margin-top: 8px;
            display: grid;
            gap: 4px;
            font-size: 0.82rem;
            color: var(--muted);
            word-break: break-all;
        }
        .meta a { color: var(--accent); text-decoration: none; }
        .cards {
            display: grid;
            gap: 8px;
            grid-template-columns: 1fr;
        }
        .status-card { padding: 8px 10px; }
        .status-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            font-size: 0.88rem;
        }
        .state {
            border: 1px solid var(--line);
            padding: 2px 6px;
            color: var(--muted);
            background: #0d140f;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            white-space: nowrap;
        }
        .state.ok { color: #05220d; background: var(--ok); border-color: var(--ok); }
        .state.warn { color: #2a1c00; background: var(--warn); border-color: var(--warn); }
        .state.bad { color: #2e0505; background: var(--bad); border-color: var(--bad); }
        .hint { margin-top: 5px; font-size: 0.8rem; color: var(--muted); }
        .stats {
            margin-top: 5px;
            font-size: 0.78rem;
            color: var(--muted);
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        @media (max-width: 940px) {
            .grid { grid-template-columns: 1fr; }
        }
  </style>
</head>
<body>
    <div class=\"app\">
        <section class=\"hero\">
            <div>
                <h1 class=\"title\">NANO-EDU VISION TERMINAL</h1>
                <p class=\"subtitle\">server stream monitor + compliance status</p>
            </div>
            <div class=\"pill\"><span id=\"dot\" class=\"dot\"></span><span id=\"status\">starting...</span></div>
        </section>

        <section class=\"grid\">
            <article class=\"panel\">
                <div class=\"video-head\">
                    <span>camera feed</span>
                    <button id=\"fullscreen-btn\" class=\"btn\" type=\"button\">fullscreen video</button>
                </div>
                <div id=\"video-shell\" class=\"video-shell\">
                    <img class=\"feed\" src=\"/mjpeg\" alt=\"Robot stream\" />
                </div>
                <div class=\"meta\">
                    <div>client stream: <a id=\"rtsp\" href=\"#\" target=\"_blank\">(waiting)</a></div>
                    <div>model: <span id=\"model\">loading...</span></div>
                    <div>analysis fps: <span id=\"fps\">--</span></div>
                </div>
            </article>

            <article class=\"panel cards\">
                <div class=\"status-card\">
                    <div class=\"status-head\"><span>uniform wordmark</span><span id=\"uniform-state\" class=\"state\">...</span></div>
                    <div id=\"uniform-hint\" class=\"hint\">waiting for detections...</div>
                    <div class=\"stats\"><span>person close: <strong id=\"uniform-person\">--</strong></span><span>age: <strong id=\"uniform-age\">--</strong></span></div>
                </div>

                <div class=\"status-card\">
                    <div class=\"status-head\"><span>id badge</span><span id=\"badge-state\" class=\"state\">...</span></div>
                    <div id=\"badge-hint\" class=\"hint\">waiting for detections...</div>
                    <div class=\"stats\"><span>person medium: <strong id=\"badge-person\">--</strong></span><span>age: <strong id=\"badge-age\">--</strong></span></div>
                </div>

                <div class=\"status-card\">
                    <div class=\"status-head\"><span>phone</span><span id=\"phone-state\" class=\"state\">...</span></div>
                    <div id=\"phone-hint\" class=\"hint\">waiting for detections...</div>
                    <div class=\"stats\"><span>person medium: <strong id=\"phone-person\">--</strong></span><span>age: <strong id=\"phone-age\">--</strong></span></div>
                </div>
            </article>
        </section>
  </div>
  <script>
        function ageText(v) {
            if (v === null || v === undefined) return '--';
            const n = Number(v);
            return Number.isFinite(n) ? (n.toFixed(1) + 's') : '--';
        }

        function setState(baseId, kind, text, hint, age, personText) {
            const stateEl = document.getElementById(baseId + '-state');
            stateEl.className = 'state ' + kind;
            stateEl.textContent = text;
            document.getElementById(baseId + '-hint').textContent = hint;
            document.getElementById(baseId + '-age').textContent = age;
            document.getElementById(baseId + '-person').textContent = personText;
        }

        function renderCompliance(a) {
            const hasMedium = !!a.person_medium_close;
            const hasClose = !!a.person_close;
            const phoneDetected = !!a.phone_detected;
            const badgeDetected = !!a.badge_detected;
            const uniformDetected = !!a.wordmark_detected;

            if (uniformDetected) {
                setState('uniform', 'ok', 'detected', 'wordmark detected on close-range person.', ageText(a.wordmark_detected_age_s), hasClose ? 'yes' : 'no');
            } else if (!hasClose) {
                setState('uniform', 'warn', 'too far', 'no close-range person for uniform check yet.', ageText(a.wordmark_detected_age_s), 'no');
            } else {
                setState('uniform', 'bad', 'not detected', 'close-range person seen, no wordmark detected.', ageText(a.wordmark_detected_age_s), 'yes');
            }

            if (badgeDetected) {
                setState('badge', 'ok', 'detected', 'id badge detected.', ageText(a.badge_detected_age_s), hasMedium ? 'yes' : 'no');
            } else if (!hasMedium) {
                setState('badge', 'warn', 'too far', 'no medium-range person for badge check yet.', ageText(a.badge_detected_age_s), 'no');
            } else {
                setState('badge', 'bad', 'not detected', 'medium-range person seen, no badge detected.', ageText(a.badge_detected_age_s), 'yes');
            }

            if (phoneDetected) {
                setState('phone', 'bad', 'detected', 'phone detected in scene.', ageText(a.phone_detected_age_s), hasMedium ? 'yes' : 'no');
            } else if (!hasMedium) {
                setState('phone', 'warn', 'too far', 'no medium-range person for phone check yet.', ageText(a.phone_detected_age_s), 'no');
            } else {
                setState('phone', 'ok', 'not detected', 'no phone detected on medium-range subject.', ageText(a.phone_detected_age_s), 'yes');
            }
        }

        async function refresh() {
            try {
                const [statusResp, analysisResp] = await Promise.all([
                    fetch('/status', { cache: 'no-store' }),
                    fetch('/analysis', { cache: 'no-store' }),
                ]);
                const s = await statusResp.json();
                const a = await analysisResp.json();

                const dot = document.getElementById('dot');
                let text = 'waiting for frames';
                dot.className = 'dot';
                if (s.last_error) {
                    text = 'error: ' + s.last_error;
                    dot.className = 'dot bad';
                } else if (s.has_frame) {
                    text = 'live, frame age ' + s.frame_age_s + 's';
                    dot.className = 'dot ok';
                }
                document.getElementById('status').textContent = text;

                const rtspLink = document.getElementById('rtsp');
                if (s.rtsp_url) {
                    rtspLink.textContent = s.rtsp_url;
                    rtspLink.href = s.rtsp_url;
                }

                document.getElementById('model').textContent = a.model_path || 'loading...';
                document.getElementById('fps').textContent = (a.analysis_fps !== undefined && a.analysis_fps !== null)
                    ? Number(a.analysis_fps).toFixed(2)
                    : '--';
                renderCompliance(a || {});
            } catch (_) {
                document.getElementById('status').textContent = 'status unavailable';
                document.getElementById('dot').className = 'dot bad';
            }
        }

        function toggleVideoFullscreen() {
            const shell = document.getElementById('video-shell');
            if (!document.fullscreenElement) {
                if (shell.requestFullscreen) {
                    shell.requestFullscreen().catch(() => {});
                }
            } else if (document.exitFullscreen) {
                document.exitFullscreen().catch(() => {});
            }
        }

        document.getElementById('fullscreen-btn').addEventListener('click', toggleVideoFullscreen);
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

