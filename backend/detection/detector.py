import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from .models import (
    ALERT_THRESHOLD_PERCENT,
    EYE_WEIGHT,
    FACE_WEIGHT,
    EYE_IMG_SIZE,
    FACE_IMG_SIZE,
    DEVICE,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def detect_faces(gray_frame, face_cascade):
    """Wrap the Haar cascade face detection used by the existing project."""
    return face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )


def preprocess_face(face_roi, target_size=FACE_IMG_SIZE):
    face_resized = cv2.resize(face_roi, target_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_array = face_rgb.astype("float32")
    return np.expand_dims(face_array, axis=0)


def extract_eyes(face_roi):
    h, w = face_roi.shape[:2]
    eye_top = int(h * 0.18)
    eye_bottom = int(h * 0.45)
    left_eye_left = int(w * 0.12)
    left_eye_right = int(w * 0.42)
    right_eye_left = int(w * 0.58)
    right_eye_right = int(w * 0.88)

    try:
        left_eye = face_roi[eye_top:eye_bottom, left_eye_left:left_eye_right]
        right_eye = face_roi[eye_top:eye_bottom, right_eye_left:right_eye_right]
        if left_eye.size == 0 or right_eye.size == 0:
            return None, None
        return left_eye, right_eye
    except Exception:
        return None, None


def preprocess_eye(eye_roi, target_size=EYE_IMG_SIZE):
    eye_rgb = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB)
    eye_resized = cv2.resize(eye_rgb, target_size)
    eye_array = eye_resized.astype("float32") / 255.0
    eye_tensor = torch.from_numpy(eye_array).permute(2, 0, 1)
    return eye_tensor.unsqueeze(0).to(DEVICE)


def predict_face_drowsiness(face_roi, face_model):
    face_input = preprocess_face(face_roi)
    prediction = face_model.predict(face_input, verbose=0)
    return float(prediction[0][0])


def predict_eye_state(eye_roi, eye_model):
    eye_input = preprocess_eye(eye_roi)
    with torch.no_grad():
        outputs = eye_model(eye_input)
        probabilities = torch.softmax(outputs, dim=1)
        closed_prob = probabilities[0][0].item()  # closed class index
        is_closed = closed_prob > 0.5
    return is_closed, closed_prob


def predict_combined_drowsiness(face_roi, face_model, eye_model):
    face_drowsiness = predict_face_drowsiness(face_roi, face_model)
    left_eye, right_eye = extract_eyes(face_roi)

    if left_eye is None or right_eye is None:
        return {
            "face_drowsiness": face_drowsiness,
            "eye_drowsiness": 0.0,
            "combined_drowsiness": face_drowsiness,
            "left_eye_closed": False,
            "right_eye_closed": False,
        }

    left_closed, left_prob = predict_eye_state(left_eye, eye_model)
    right_closed, right_prob = predict_eye_state(right_eye, eye_model)
    eye_drowsiness = (left_prob + right_prob) / 2.0
    combined_drowsiness = (FACE_WEIGHT * face_drowsiness) + (EYE_WEIGHT * eye_drowsiness)

    return {
        "face_drowsiness": face_drowsiness,
        "eye_drowsiness": eye_drowsiness,
        "combined_drowsiness": combined_drowsiness,
        "left_eye_closed": left_closed,
        "right_eye_closed": right_closed,
    }


def draw_detection_results(frame, x, y, w, h, results):
    drowsiness_percent = results["combined_drowsiness"] * 100
    is_drowsy = drowsiness_percent >= ALERT_THRESHOLD_PERCENT
    if is_drowsy:
        box_color = (0, 0, 255)
        text_color = (0, 0, 255)
        status_text = "DROWSY! WAKE UP!"
    else:
        box_color = (0, 255, 0)
        text_color = (0, 255, 0)
        status_text = f"Awake ({100 - drowsiness_percent:.1f}% alert)"

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(frame, status_text, (x, y - 10), FONT, FONT_SCALE, text_color, FONT_THICKNESS)

    info_y = 30
    cv2.putText(frame, f"Combined: {drowsiness_percent:.1f}%", (10, info_y), FONT, 0.6, (255, 255, 255), 1)
    info_y += 25
    cv2.putText(frame, f"Face: {results['face_drowsiness'] * 100:.1f}%", (10, info_y), FONT, 0.6, (255, 255, 255), 1)
    info_y += 25
    cv2.putText(frame, f"Eyes: {results['eye_drowsiness'] * 100:.1f}%", (10, info_y), FONT, 0.6, (255, 255, 255), 1)
    info_y += 25
    eye_status = f"L:{'Closed' if results['left_eye_closed'] else 'Open'} R:{'Closed' if results['right_eye_closed'] else 'Open'}"
    cv2.putText(frame, eye_status, (10, info_y), FONT, 0.6, (255, 255, 255), 1)
    return frame


def process_frame(frame, face_model, eye_model, face_cascade):
    """Process a single OpenCV frame and return the detection result payload."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, face_cascade)

    if len(faces) == 0:
        _, encoded = cv2.imencode('.jpg', frame)
        return {
            "status": "awake",
            "drowsiness_score": 0.0,
            "face_score": 0.0,
            "eye_score": 0.0,
            "left_eye": "OPEN",
            "right_eye": "OPEN",
            "fps": 0.0,
            "alert": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "face": {"x": 0, "y": 0, "width": 0, "height": 0},
            "frame_data": encoded.tobytes(),
            "annotated_frame": frame,
        }

    x, y, w, h = faces[0]
    face_roi = frame[y:y + h, x:x + w]
    results = predict_combined_drowsiness(face_roi, face_model, eye_model)
    drowsiness_score = results["combined_drowsiness"] * 100.0
    face_score = results["face_drowsiness"] * 100.0
    eye_score = results["eye_drowsiness"] * 100.0
    is_drowsy = drowsiness_score >= ALERT_THRESHOLD_PERCENT

    annotated = frame.copy()
    draw_detection_results(annotated, x, y, w, h, results)

    _, encoded = cv2.imencode('.jpg', annotated)
    frame_data = encoded.tobytes()

    payload = {
        "status": "drowsy" if is_drowsy else "awake",
        "drowsiness_score": round(float(drowsiness_score), 2),
        "face_score": round(float(face_score), 2),
        "eye_score": round(float(eye_score), 2),
        "left_eye": "CLOSED" if results["left_eye_closed"] else "OPEN",
        "right_eye": "CLOSED" if results["right_eye_closed"] else "OPEN",
        "fps": 0.0,
        "alert": bool(is_drowsy),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "face": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "frame_data": frame_data,
        "annotated_frame": annotated,
    }
    return payload


class DetectionSession:
    def __init__(self, face_model, eye_model, face_cascade, camera_source=0):
        self.face_model = face_model
        self.eye_model = eye_model
        self.face_cascade = face_cascade
        self.camera_source = camera_source
        self.cap = None
        self.running = False
        self.last_frame_time = time.time()

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError("Camera unavailable")
        self.running = True
        return self.cap

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera unavailable")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read frame from camera")

        current = time.time()
        fps = 1.0 / max(current - self.last_frame_time, 0.01) if self.last_frame_time else 0.0
        self.last_frame_time = current
        return frame, fps


def run_detection_loop(face_model, eye_model, face_cascade, camera_source=0):
    """Convenience wrapper mainly for standalone script use and backend compatibility."""
    session = DetectionSession(face_model, eye_model, face_cascade, camera_source)
    session.start()

    try:
        while session.running:
            frame, fps = session.read_frame()
            payload = process_frame(frame, face_model, eye_model, face_cascade)
            payload["fps"] = round(float(fps), 2)
            yield payload
    finally:
        session.stop()
