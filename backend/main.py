import os

# Suppress verbose TensorFlow and oneDNN logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import asyncio
import base64
import platform
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detection.models import ALERT_THRESHOLD_PERCENT, EYE_IMG_SIZE, FACE_IMG_SIZE, load_models
from detection.detector import process_frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["face_model"], state["eye_model"], state["face_cascade"] = load_models()
        state["models_loaded"] = True
        print("✓ Deep learning models and Haar cascade loaded successfully.")
    except Exception as exc:
        state["models_loaded"] = False
        state["startup_error"] = str(exc)
        print(f"[ERROR] Failed to load models: {exc}", file=sys.stderr)
        traceback.print_exc()
    yield
    camera = state.get("camera")
    if camera is not None:
        camera.release()


app = FastAPI(title="DriverGuard AI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: Dict[str, Any] = {
    "models_loaded": False,
    "camera_ready": False,
    "camera": None,
    "face_model": None,
    "eye_model": None,
    "face_cascade": None,
    "monitoring": False,
    "alert_count": 0,
    "latest_alert_time": None,
    "last_status": "awake",
    "session_started": None,
    "frames_processed": 0,
    "fps_history": [],
    "alert_active": False,
    "muted": False,
}


def open_camera():
    """Open the first available camera backend on the current platform."""
    source = int(os.getenv("CAMERA_SOURCE", "0"))
    backends = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        camera = cv2.VideoCapture(source, backend)
        if camera.isOpened():
            return camera
        camera.release()

    raise RuntimeError(
        f"Unable to open camera index {source}. Check Windows camera permissions, "
        "close other camera apps, or set CAMERA_SOURCE to another index."
    )





@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": bool(state.get("models_loaded")),
        "monitoring": bool(state.get("monitoring")),
        "camera_ready": bool(state.get("camera_ready")),
        "session_started": state.get("session_started"),
        "alert_threshold": ALERT_THRESHOLD_PERCENT,
        "face_input": list(FACE_IMG_SIZE),
        "eye_input": list(EYE_IMG_SIZE),
    }


@app.get("/api/system")
async def system_info():
    return {
        "name": "DriverGuard AI",
        "subtitle": "Real-Time Driver Drowsiness Detection",
        "detection_engine": "OpenCV",
        "face_model": "CNN / TensorFlow",
        "eye_model": "MobileNetV2 / PyTorch",
        "face_input": "64 × 64",
        "eye_input": "224 × 224",
        "fusion": "30% Face + 70% Eyes",
        "alert_threshold": 70.0,
        "device": "CPU" if not __import__('torch').cuda.is_available() else "GPU",
    }


@app.post("/api/start")
async def start_monitoring():
    if not state.get("models_loaded"):
        return JSONResponse(status_code=503, content={"error": "Models are still loading or failed to load"})

    if state.get("camera") is not None and state["camera"].isOpened():
        state["monitoring"] = True
        state["session_started"] = time.time()
        return {"status": "monitoring"}

    try:
        camera = open_camera()
    except RuntimeError as exc:
        state["camera_ready"] = False
        return JSONResponse(status_code=503, content={"error": str(exc)})

    state["camera"] = camera
    state["camera_ready"] = True
    state["monitoring"] = True
    state["session_started"] = time.time()
    return {"status": "monitoring"}


@app.post("/api/stop")
async def stop_monitoring():
    camera = state.get("camera")
    if camera is not None:
        camera.release()
    state["camera"] = None
    state["camera_ready"] = False
    state["monitoring"] = False
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset_session():
    state["alert_count"] = 0
    state["latest_alert_time"] = None
    state["last_status"] = "awake"
    state["fps_history"] = []
    state["frames_processed"] = 0
    state["session_started"] = time.time()
    return {"status": "reset"}


@app.post("/api/mute")
async def mute_alert():
    state["muted"] = not bool(state.get("muted", False))
    return {"muted": state["muted"]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    camera = None

    try:
        while True:
            if state.get("monitoring"):
                camera = state.get("camera")
                if camera is None or not camera.isOpened():
                    try:
                        camera = open_camera()
                    except RuntimeError as exc:
                        await websocket.send_json({"event": "camera_unavailable", "message": str(exc)})
                        await asyncio.sleep(1)
                        continue
                    state["camera"] = camera
                    state["camera_ready"] = True

                ret, frame = camera.read()
                if not ret:
                    await websocket.send_json({"event": "camera_unavailable", "message": "Unable to read camera frame"})
                    await asyncio.sleep(1)
                    continue

                payload = process_frame(frame, state["face_model"], state["eye_model"], state["face_cascade"])
                payload["alert_count"] = state["alert_count"]
                payload["latest_alert_time"] = state["latest_alert_time"]
                payload["session_started"] = state.get("session_started")
                payload["frames_processed"] = state["frames_processed"]
                payload["device"] = "GPU" if __import__('torch').cuda.is_available() else "CPU"

                if payload["alert"] and not state.get("alert_active") and not state.get("muted", False):
                    state["alert_count"] += 1
                    state["latest_alert_time"] = payload["timestamp"]
                state["alert_active"] = bool(payload["alert"])

                state["last_status"] = payload["status"]
                state["frames_processed"] += 1
                state["fps_history"].append(payload["fps"])
                if len(state["fps_history"]) > 60:
                    state["fps_history"] = state["fps_history"][-60:]

                frame_bytes = payload.pop("frame_data", b"")
                payload["frame_data"] = base64.b64encode(frame_bytes).decode("utf-8")
                payload.pop("annotated_frame", None)

                await websocket.send_json(payload)
            else:
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        if camera is not None:
            camera.release()
    except Exception as exc:
        await websocket.send_json({"event": "backend_error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
