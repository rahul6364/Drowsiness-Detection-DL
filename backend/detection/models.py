import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from torchvision.models import mobilenet_v2

# Reduce TensorFlow noise and avoid GPU JIT issues in CPU-only environments.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FACE_MODEL_PATH = MODELS_DIR / "drowsiness_cnn.keras"
EYE_MODEL_PATH = MODELS_DIR / "eye_model_best.pth"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FACE_IMG_SIZE = (64, 64)
EYE_IMG_SIZE = (224, 224)
FACE_WEIGHT = 0.3
EYE_WEIGHT = 0.7
ALERT_THRESHOLD_PERCENT = 70.0

# Force TensorFlow to use CPU only to keep PyTorch and TensorFlow inference stable.
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    """Load the TensorFlow face model, PyTorch eye model, and Haar cascade once."""
    face_model = load_model(FACE_MODEL_PATH)

    global FACE_IMG_SIZE
    face_input_shape = face_model.input_shape
    if len(face_input_shape) == 4:
        FACE_IMG_SIZE = (face_input_shape[1], face_input_shape[2])

    eye_model = mobilenet_v2(weights=None)
    eye_model.classifier[1] = nn.Linear(eye_model.last_channel, 2)

    checkpoint = torch.load(EYE_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        eye_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        eye_model.load_state_dict(checkpoint)

    eye_model = eye_model.to(DEVICE)
    eye_model.eval()

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise ValueError(f"Failed to load Haar cascade from {HAAR_CASCADE_PATH}")

    return face_model, eye_model, face_cascade
