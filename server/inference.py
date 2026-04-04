from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence
from urllib.request import urlopen

ObstacleState = Literal["clear", "left", "right", "center", "unknown"]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    cls_index: int


@dataclass
class InferenceResult:
    obstacle: ObstacleState = "unknown"
    throttle: float = 0.0
    steer: float = 0.0
    label: str = "—"
    confidence: float = 0.0


@dataclass
class FrameAnalysis:
    detections: list[Detection]
    obstacle: ObstacleState = "unknown"
    throttle: float = 0.0
    steer: float = 0.0
    label: str = "—"
    confidence: float = 0.0
    error: str = ""


_MODEL_NAME_RAW = str(os.environ.get("ROBOT_YOLO_MODEL", "yolo11m.pt")).strip() or "yolo11m.pt"


def _normalize_model_name(name: str) -> str:
    low = str(name or "").strip().lower()
    aliases = {
        "yolov11n.pt": "yolo11n.pt",
        "yolov11s.pt": "yolo11s.pt",
        "yolov11m.pt": "yolo11m.pt",
        "yolov11l.pt": "yolo11l.pt",
        "yolov11x.pt": "yolo11x.pt",
    }
    return aliases.get(low, str(name or "").strip() or "yolo11m.pt")


_MODEL_NAME = _normalize_model_name(_MODEL_NAME_RAW)
_MODEL = None
_MODEL_ERR: str = ""

_PERSON_CLASS = "person"
_PHONE_CLASS = "cell phone"

_OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "train",
    "truck",
    "bench",
    "chair",
    "dining table",
    "sofa",
    "tv",
    "potted plant",
    "dog",
    "cat",
    "cell phone",
}


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _try_import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception:
        return None


def _resolve_label(cls_index: int, names: Any) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_index), cls_index))
        if isinstance(names, (list, tuple)) and 0 <= int(cls_index) < len(names):
            return str(names[int(cls_index)])
    except Exception:
        pass
    return str(cls_index)


def _download_to(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=20) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"empty download from {url}")
    path.write_bytes(data)


def _load_model():
    global _MODEL, _MODEL_ERR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERR:
        return None

    YOLO = _try_import_yolo()
    if YOLO is None:
        _MODEL_ERR = "ultralytics unavailable"
        return None

    try:
        _MODEL = YOLO(_MODEL_NAME)
        return _MODEL
    except Exception as exc:
        # Fall back to a known-good default in case env var contains an
        # unsupported alias/name. Keep the first error for diagnostics.
        first_err = str(exc)
        try:
            if _MODEL_NAME != "yolo11m.pt":
                _MODEL = YOLO("yolo11m.pt")
                _MODEL_ERR = ""
                return _MODEL
        except Exception as fallback_exc:
            _MODEL_ERR = f"{first_err}; fallback failed: {fallback_exc}"
            return None
        _MODEL_ERR = first_err
        return None


def _analysis_error(message: str) -> FrameAnalysis:
    return FrameAnalysis(detections=[], obstacle="unknown", throttle=0.0, steer=0.0, label=str(message), confidence=0.0, error=str(message))


def analyze_frame_from_bgr(frame) -> FrameAnalysis:
    cv2 = _try_import_cv2()
    if cv2 is None:
        return _analysis_error("opencv-missing")

    model = _load_model()
    if model is None:
        return _analysis_error("model-unavailable" if not _MODEL_ERR else _MODEL_ERR)

    try:
        height, width = frame.shape[:2]
        conf_thr = max(0.36, min(0.9, float(os.environ.get("ROBOT_YOLO_CONF", "0.40"))))
        img_size = max(320, min(1280, int(os.environ.get("ROBOT_YOLO_IMGSZ", "640"))))

        results = model.predict(source=frame, conf=conf_thr, imgsz=img_size, verbose=False)
        if not results:
            return FrameAnalysis(detections=[], obstacle="clear", throttle=0.9, steer=0.0, label="none", confidence=0.0, error="")

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return FrameAnalysis(detections=[], obstacle="clear", throttle=0.9, steer=0.0, label="none", confidence=0.0, error="")

        names = getattr(result, "names", None) or getattr(model, "names", {})
        detections: list[Detection] = []

        best = None
        best_score = -1.0
        min_area_ratio = max(0.0005, min(0.5, float(os.environ.get("ROBOT_OD_MIN_AREA", "0.005"))))

        # Extract arrays from Ultralytics Boxes object in a version-tolerant way.
        try:
            xyxy_arr = getattr(boxes, "xyxy", None)
            conf_arr = getattr(boxes, "conf", None)
            cls_arr = getattr(boxes, "cls", None)
            if xyxy_arr is None or conf_arr is None or cls_arr is None:
                return _analysis_error("dnn-error: boxes missing xyxy/conf/cls")

            xyxy_rows = xyxy_arr.tolist() if hasattr(xyxy_arr, "tolist") else list(xyxy_arr)
            conf_rows = conf_arr.tolist() if hasattr(conf_arr, "tolist") else list(conf_arr)
            cls_rows = cls_arr.tolist() if hasattr(cls_arr, "tolist") else list(cls_arr)
        except Exception as exc:
            return _analysis_error(f"dnn-error: decode boxes failed: {exc}")

        for i in range(min(len(xyxy_rows), len(conf_rows), len(cls_rows))):
            conf = _float_value(conf_rows[i])
            if conf < conf_thr:
                continue

            cls_index = _int_value(cls_rows[i])
            label = _resolve_label(cls_index, names)

            coords = xyxy_rows[i]
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in coords)
            except Exception:
                continue

            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection(
                label=label,
                confidence=float(conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cls_index=cls_index,
            )
            detections.append(det)

            if label not in _OBSTACLE_CLASSES:
                continue

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area_ratio = float(bw * bh) / float(max(1, width * height))
            if area_ratio < min_area_ratio:
                continue

            score = conf * (1.0 + min(1.0, area_ratio * 10.0))
            if score > best_score:
                best_score = score
                best = det

        if best is None:
            return FrameAnalysis(detections=detections, obstacle="clear", throttle=0.9, steer=0.0, label="none", confidence=0.0, error="")

        cx = (best.x1 + best.x2) * 0.5
        nx = cx / float(max(1, width))
        if nx < 0.40:
            obstacle, throttle, steer = "left", 0.0, 1.0
        elif nx > 0.60:
            obstacle, throttle, steer = "right", 0.0, -1.0
        else:
            obstacle, throttle, steer = "center", -0.2, -1.0

        return FrameAnalysis(
            detections=detections,
            obstacle=obstacle,
            throttle=float(throttle),
            steer=float(steer),
            label=str(best.label),
            confidence=float(best.confidence),
            error="",
        )
    except Exception as exc:
        return _analysis_error(f"dnn-error: {exc}")


def get_model_error() -> str:
    return str(_MODEL_ERR)


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)) and value:
            return float(value[0])
        return float(value)
    except Exception:
        return float(default)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if hasattr(value, "item"):
            return int(value.item())
        if isinstance(value, (list, tuple)) and value:
            return int(value[0])
        return int(value)
    except Exception:
        return int(default)


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(min(x1, x2))))
    top = max(0, min(height - 1, int(min(y1, y2))))
    right = max(0, min(width, int(max(x1, x2))))
    bottom = max(0, min(height, int(max(y1, y2))))
    return left, top, right, bottom


def _box_area(det: Detection) -> int:
    return max(0, det.x2 - det.x1) * max(0, det.y2 - det.y1)


def _box_center(det: Detection) -> tuple[float, float]:
    return ((det.x1 + det.x2) * 0.5, (det.y1 + det.y2) * 0.5)


def _point_in_box(x: float, y: float, det: Detection) -> bool:
    return det.x1 <= x <= det.x2 and det.y1 <= y <= det.y2


def _intersection_area(a: Detection, b: Detection) -> int:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def detect_objects_from_bgr(frame) -> list[Detection]:
    return list(analyze_frame_from_bgr(frame).detections)


def draw_detections(frame, detections: Sequence[Detection]):
    cv2 = _try_import_cv2()
    if cv2 is None:
        return frame

    annotated = frame.copy()

    def color_for(label: str) -> tuple[int, int, int]:
        low = str(label).lower()
        if low == _PERSON_CLASS:
            return (255, 160, 0)
        if low == _PHONE_CLASS:
            return (0, 215, 255)
        if low in _OBSTACLE_CLASSES:
            return (0, 0, 255)
        return (0, 200, 0)

    for det in detections:
        color = color_for(det.label)
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = det.x1
        text_y = max(0, det.y1 - 6)
        bg_y1 = max(0, text_y - th - baseline)
        bg_y2 = min(annotated.shape[0], text_y + baseline)
        bg_x2 = min(annotated.shape[1], text_x + tw + 6)
        cv2.rectangle(annotated, (text_x, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (text_x + 3, max(th, text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def phone_holder_crop_from_detections(frame, detections: Sequence[Detection]):
    phone_boxes = [det for det in detections if str(det.label).lower() == _PHONE_CLASS]
    person_boxes = [det for det in detections if str(det.label).lower() == _PERSON_CLASS]

    if not phone_boxes or not person_boxes:
        return None

    phone = max(phone_boxes, key=lambda det: (det.confidence, _box_area(det)))
    phone_cx, phone_cy = _box_center(phone)

    best_person: Optional[Detection] = None
    best_score = -1.0
    for person in person_boxes:
        overlap = float(_intersection_area(person, phone))
        inside = 1.0 if _point_in_box(phone_cx, phone_cy, person) else 0.0
        person_cx, person_cy = _box_center(person)
        distance = ((person_cx - phone_cx) ** 2 + (person_cy - phone_cy) ** 2) ** 0.5
        score = (inside * 10.0) + overlap + (person.confidence * 2.0) - (distance * 0.01)
        if score > best_score:
            best_score = score
            best_person = person

    if best_person is None:
        return None

    frame_h, frame_w = frame.shape[:2]
    left = min(best_person.x1, phone.x1)
    top = min(best_person.y1, phone.y1)
    right = max(best_person.x2, phone.x2)
    bottom = max(best_person.y2, phone.y2)

    pad_x = int(max(12, (right - left) * 0.12))
    pad_y = int(max(12, (bottom - top) * 0.12))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(frame_w, right + pad_x)
    bottom = min(frame_h, bottom + pad_y)

    if right <= left or bottom <= top:
        return None

    return frame[top:bottom, left:right].copy()


def detect_obstacle_from_bgr(frame) -> tuple[ObstacleState, str, float]:
    analysis = analyze_frame_from_bgr(frame)
    if analysis.error and analysis.error not in ("", "none") and not analysis.detections:
        if analysis.error == "model-unavailable" and not _MODEL_ERR:
            return "unknown", "model-unavailable", 0.0
        if analysis.error == "opencv-missing":
            return "unknown", "opencv-missing", 0.0
        return "unknown", str(analysis.error)[:42], 0.0

    return analysis.obstacle, analysis.label, float(analysis.confidence)


def infer_from_jpeg_b64(jpeg_b64: str) -> InferenceResult:
    cv2 = _try_import_cv2()
    if cv2 is None:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="opencv-missing", confidence=0.0)

    try:
        raw = base64.b64decode(jpeg_b64.encode("ascii"), validate=False)
    except Exception:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="decode-error", confidence=0.0)

    try:
        import numpy as np  # type: ignore

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="imdecode-none", confidence=0.0)
    except Exception:
        return InferenceResult(obstacle="unknown", throttle=0.0, steer=0.0, label="numpy-error", confidence=0.0)

    analysis = analyze_frame_from_bgr(frame)

    return InferenceResult(
        obstacle=analysis.obstacle,
        throttle=float(analysis.throttle),
        steer=float(analysis.steer),
        label=str(analysis.label),
        confidence=float(analysis.confidence),
    )
