"""DriverGuard AI backend detection package."""

from .detector import DetectionSession, process_frame, run_detection_loop
from .models import (
    ALERT_THRESHOLD_PERCENT,
    EYE_WEIGHT,
    FACE_WEIGHT,
    EYE_IMG_SIZE,
    FACE_IMG_SIZE,
    load_models,
)

__all__ = [
    "DetectionSession",
    "process_frame",
    "run_detection_loop",
    "load_models",
    "ALERT_THRESHOLD_PERCENT",
    "EYE_WEIGHT",
    "FACE_WEIGHT",
    "EYE_IMG_SIZE",
    "FACE_IMG_SIZE",
]
