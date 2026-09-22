# DriverGuard AI Backend

This backend exposes the existing drowsiness detection logic through FastAPI and a WebSocket stream.

## Important notes

- The original ML inference in `scripts/integrated_detection.py` remains intact.
- The backend loads the TensorFlow CNN and PyTorch MobileNetV2 model once during startup.
- The detection loop reuses the same 30% face / 70% eye fusion and 70% alert threshold.
- Camera access is handled through OpenCV and surfaced over WebSocket for the frontend.

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

Then open the frontend and connect to `ws://localhost:8000/ws`.

## API

- `GET /health`
- `GET /api/system`
- `POST /api/start`
- `POST /api/stop`
- `POST /api/reset`
- `POST /api/mute`
- `WebSocket /ws`

## Notes on streaming

The backend sends JPEG frames only when the browser is upgraded to a streaming-friendly client. This project uses WebSocket JSON payloads for real-time state and the browser reconstructs the camera display from a base64-encoded frame. This avoids duplicating the model pipeline or loading ML models per frame.
