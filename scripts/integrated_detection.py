#!/usr/bin/env python3
"""
Integrated Drowsiness Detection System

This script implements a real-time drowsiness detection system that combines:
- Face-based drowsiness detection using CNN (TensorFlow/Keras)
- Eye-state detection using MobileNetV2 (PyTorch)

The system uses a weighted fusion approach:
- Face model: 30% weight
- Eye model: 70% weight

Alert is triggered when combined drowsiness score >= 70%

Usage:
    python integrated_detection.py

Press 'q' to quit the detection window.
"""

# ============================================================================
# 1. IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
import os
import sys
import time
from pathlib import Path

# Configure TensorFlow before import to avoid CUDA JIT compilation issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow warnings
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # Disable XLA JIT compilation

# Computer vision and video processing
import cv2
import numpy as np

# TensorFlow/Keras for face model
import tensorflow as tf
from tensorflow.keras.models import load_model

# Force TensorFlow to use CPU only (PyTorch can still use GPU)
tf.config.set_visible_devices([], 'GPU')

# PyTorch for eye model
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


# ============================================================================
# 2. CONFIGURATION PARAMETERS
# ============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

# Model paths
FACE_MODEL_PATH = MODELS_DIR / 'drowsiness_cnn.keras'
EYE_MODEL_PATH = MODELS_DIR / 'eye_model_best.pth'
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Model input sizes
FACE_IMG_SIZE = (64, 64)
EYE_IMG_SIZE = (224, 224)

# Detection parameters
FACE_WEIGHT = 0.3
EYE_WEIGHT = 0.7
ALERT_THRESHOLD_PERCENT = 70.0
EYE_CLOSED_SECONDS = 2.0

# Display settings
WINDOW_NAME = "Integrated Drowsiness Detection"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# PyTorch device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 3. MODEL LOADING
# ============================================================================

def load_models():
    """
    Load all required models for drowsiness detection.

    Returns:
        Tuple of (face_model, eye_model, face_cascade)
    """
    print("=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    # Load face drowsiness model (TensorFlow/Keras)
    print(f"\n[1/3] Loading face drowsiness model...")
    print(f"      Path: {FACE_MODEL_PATH}")
    face_model = load_model(FACE_MODEL_PATH)
    print(f"      ✓ Loaded successfully")
    print(f"      Input shape: {face_model.input_shape}")

    # Auto-detect face model input size
    global FACE_IMG_SIZE
    face_input_shape = face_model.input_shape
    if len(face_input_shape) == 4:
        FACE_IMG_SIZE = (face_input_shape[1], face_input_shape[2])
        print(f"      Auto-detected input size: {FACE_IMG_SIZE}")

    # Load eye state model (PyTorch)
    print(f"\n[2/3] Loading eye state model...")
    print(f"      Path: {EYE_MODEL_PATH}")

    # Create model architecture
    eye_model = mobilenet_v2(pretrained=False)
    eye_model.classifier[1] = nn.Linear(eye_model.last_channel, 2)  # 2 classes

    # Load trained weights (supports both checkpoint and direct state_dict formats)
    checkpoint = torch.load(EYE_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format (includes optimizer, epoch, etc.)
        eye_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"      Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Direct state_dict format
        eye_model.load_state_dict(checkpoint)

    eye_model = eye_model.to(DEVICE)
    eye_model.eval()

    print(f"      ✓ Loaded successfully")
    print(f"      Device: {DEVICE}")
    print(f"      Input size: {EYE_IMG_SIZE}")

    # Load Haar Cascade for face detection
    print(f"\n[3/3] Loading face detector...")
    print(f"      Path: {HAAR_CASCADE_PATH}")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    if face_cascade.empty():
        raise ValueError(f"Failed to load Haar cascade from {HAAR_CASCADE_PATH}")

    print(f"      ✓ Loaded successfully")
    print("\n" + "=" * 60)
    print("ALL MODELS LOADED")
    print("=" * 60 + "\n")

    return face_model, eye_model, face_cascade


# ============================================================================
# 4. FACE DETECTION AND PREPROCESSING
# ============================================================================

def detect_faces(gray_frame, face_cascade):
    """
    Detect faces in grayscale frame using Haar Cascade.

    Args:
        gray_frame: Grayscale image
        face_cascade: Loaded Haar cascade classifier

    Returns:
        List of face rectangles (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def preprocess_face(face_roi, target_size=FACE_IMG_SIZE):
    """
    Preprocess face ROI for drowsiness CNN model.

    Args:
        face_roi: Face region (BGR)
        target_size: Target size tuple (height, width)

    Returns:
        Preprocessed face ready for TensorFlow model
    """
    # Resize to model input size
    face_resized = cv2.resize(face_roi, target_size)

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Convert to float32 (model has built-in Rescaling layer)
    face_array = face_rgb.astype('float32')

    # Add batch dimension
    face_batch = np.expand_dims(face_array, axis=0)

    return face_batch


# ============================================================================
# 5. EYE EXTRACTION AND PREPROCESSING
# ============================================================================

def extract_eyes(face_roi):
    """
    Extract left and right eye regions from face ROI.

    Eye regions are calculated as percentages of face dimensions:
    - Height: 18-45% of face height
    - Left eye: 12-42% of face width
    - Right eye: 58-88% of face width

    Args:
        face_roi: Face region (BGR)

    Returns:
        Tuple of (left_eye, right_eye) or (None, None) if extraction fails
    """
    h, w = face_roi.shape[:2]

    # Define eye regions as percentages
    eye_top = int(h * 0.18)
    eye_bottom = int(h * 0.45)

    left_eye_left = int(w * 0.12)
    left_eye_right = int(w * 0.42)

    right_eye_left = int(w * 0.58)
    right_eye_right = int(w * 0.88)

    # Extract eye regions
    try:
        left_eye = face_roi[eye_top:eye_bottom, left_eye_left:left_eye_right]
        right_eye = face_roi[eye_top:eye_bottom, right_eye_left:right_eye_right]

        # Validate extracted regions
        if left_eye.size == 0 or right_eye.size == 0:
            return None, None

        return left_eye, right_eye
    except Exception as e:
        print(f"Error extracting eyes: {e}")
        return None, None


def preprocess_eye(eye_roi, target_size=EYE_IMG_SIZE):
    """
    Preprocess eye ROI for MobileNetV2 PyTorch model.

    Args:
        eye_roi: Eye region (BGR)
        target_size: Target size tuple (height, width)

    Returns:
        Preprocessed eye tensor ready for PyTorch model
    """
    # Convert BGR to RGB
    eye_rgb = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    eye_resized = cv2.resize(eye_rgb, target_size)

    # Convert to float32 and normalize to [0, 1]
    eye_array = eye_resized.astype('float32') / 255.0

    # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
    eye_tensor = torch.from_numpy(eye_array).permute(2, 0, 1)

    # Add batch dimension
    eye_batch = eye_tensor.unsqueeze(0).to(DEVICE)

    return eye_batch


# ============================================================================
# 6. MODEL INFERENCE
# ============================================================================

def predict_face_drowsiness(face_roi, face_model):
    """
    Predict drowsiness from face using CNN model.

    Args:
        face_roi: Face region (BGR)
        face_model: Loaded TensorFlow face model

    Returns:
        Drowsiness probability (0-1, higher = more drowsy)
    """
    face_input = preprocess_face(face_roi)
    prediction = face_model.predict(face_input, verbose=0)
    drowsiness_prob = prediction[0][0]  # Drowsy class probability
    return float(drowsiness_prob)


def predict_eye_state(eye_roi, eye_model):
    """
    Predict if eye is open or closed using MobileNetV2.

    Args:
        eye_roi: Eye region (BGR)
        eye_model: Loaded PyTorch eye model

    Returns:
        Tuple of (is_closed, closed_probability)
        is_closed: Boolean indicating if eye is closed
        closed_probability: Probability eye is closed (0-1)
    """
    eye_input = preprocess_eye(eye_roi)

    with torch.no_grad():
        outputs = eye_model(eye_input)
        probabilities = torch.softmax(outputs, dim=1)
        closed_prob = probabilities[0][0].item()  # Index 0 is "closed"
        is_closed = closed_prob > 0.5

    return is_closed, closed_prob


def predict_combined_drowsiness(face_roi, face_model, eye_model):
    """
    Predict drowsiness using both face and eye models.

    Args:
        face_roi: Face region (BGR)
        face_model: Loaded TensorFlow face model
        eye_model: Loaded PyTorch eye model

    Returns:
        Dictionary containing:
        - face_drowsiness: Face model score (0-1)
        - eye_drowsiness: Eye model score (0-1)
        - combined_drowsiness: Weighted fusion score (0-1)
        - left_eye_closed: Boolean
        - right_eye_closed: Boolean
    """
    # Get face drowsiness score
    face_drowsiness = predict_face_drowsiness(face_roi, face_model)

    # Extract eyes and get eye states
    left_eye, right_eye = extract_eyes(face_roi)

    if left_eye is None or right_eye is None:
        # Fallback to face-only detection
        return {
            'face_drowsiness': face_drowsiness,
            'eye_drowsiness': 0.0,
            'combined_drowsiness': face_drowsiness,
            'left_eye_closed': False,
            'right_eye_closed': False
        }

    # Predict eye states
    left_closed, left_prob = predict_eye_state(left_eye, eye_model)
    right_closed, right_prob = predict_eye_state(right_eye, eye_model)

    # Calculate eye drowsiness (average of both eyes)
    eye_drowsiness = (left_prob + right_prob) / 2.0

    # Weighted fusion
    combined_drowsiness = (FACE_WEIGHT * face_drowsiness) + (EYE_WEIGHT * eye_drowsiness)

    return {
        'face_drowsiness': face_drowsiness,
        'eye_drowsiness': eye_drowsiness,
        'combined_drowsiness': combined_drowsiness,
        'left_eye_closed': left_closed,
        'right_eye_closed': right_closed
    }


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def draw_detection_results(frame, x, y, w, h, results):
    """
    Draw detection results on frame.

    Args:
        frame: Video frame
        x, y, w, h: Face bounding box
        results: Dictionary from predict_combined_drowsiness()
    """
    drowsiness_percent = results['combined_drowsiness'] * 100
    is_drowsy = drowsiness_percent >= ALERT_THRESHOLD_PERCENT

    # Set colors based on state
    if is_drowsy:
        box_color = (0, 0, 255)  # Red
        text_color = (0, 0, 255)
        status_text = "DROWSY! WAKE UP!"
    else:
        box_color = (0, 255, 0)  # Green
        text_color = (0, 255, 0)
        status_text = f"Awake ({100-drowsiness_percent:.1f}% alert)"

    # Draw face bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

    # Draw status text above face
    cv2.putText(frame, status_text, (x, y-10),
                FONT, FONT_SCALE, text_color, FONT_THICKNESS)

    # Draw detailed info panel
    info_y = 30
    cv2.putText(frame, f"Combined: {drowsiness_percent:.1f}%",
                (10, info_y), FONT, 0.6, (255, 255, 255), 1)

    info_y += 25
    cv2.putText(frame, f"Face: {results['face_drowsiness']*100:.1f}%",
                (10, info_y), FONT, 0.6, (255, 255, 255), 1)

    info_y += 25
    cv2.putText(frame, f"Eyes: {results['eye_drowsiness']*100:.1f}%",
                (10, info_y), FONT, 0.6, (255, 255, 255), 1)

    info_y += 25
    eye_status = f"L:{'Closed' if results['left_eye_closed'] else 'Open'} " \
                 f"R:{'Closed' if results['right_eye_closed'] else 'Open'}"
    cv2.putText(frame, eye_status,
                (10, info_y), FONT, 0.6, (255, 255, 255), 1)


# ============================================================================
# 8. MAIN DETECTION LOOP
# ============================================================================

def run_detection(face_model, eye_model, face_cascade):
    """
    Main detection loop for real-time drowsiness detection.

    Args:
        face_model: Loaded TensorFlow face model
        eye_model: Loaded PyTorch eye model
        face_cascade: Loaded Haar cascade classifier

    Press 'q' to quit.
    """
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(video_source, str):
        print(f"Opening video file: {video_source}")
    else:
        print("Starting webcam...")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        if isinstance(video_source, str):
            print(f"Error: Could not open video file: {video_source}")
        else:
            print("Error: Could not open webcam")
        return

    print("✓ Video source opened successfully")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("\nDetection parameters:")
    print(f"  Face weight: {FACE_WEIGHT * 100:.0f}%")
    print(f"  Eye weight: {EYE_WEIGHT * 100:.0f}%")
    print(f"  Alert threshold: {ALERT_THRESHOLD_PERCENT:.0f}%")
    print("\n" + "=" * 60)
    print("DETECTION STARTED")
    print("=" * 60 + "\n")

    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(video_source, str):
                    # Loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Error: Failed to capture frame")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detect_faces(gray, face_cascade)

            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]

                # Get combined drowsiness prediction
                results = predict_combined_drowsiness(face_roi, face_model, eye_model)

                # Draw results
                draw_detection_results(frame, x, y, w, h, results)

            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 30:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                frame_count = 0

            cv2.putText(frame, f"FPS: {fps:.1f}",
                       (frame.shape[1] - 120, 30),
                       FONT, 0.6, (255, 255, 255), 1)

            # Display frame
            cv2.imshow(WINDOW_NAME, frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("CLEANUP COMPLETE")
        print("=" * 60)


# ============================================================================
# 9. MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the integrated drowsiness detection system.
    """
    print("\n" + "=" * 60)
    print("INTEGRATED DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__} (using CPU)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print()

    # Verify model files exist
    if not FACE_MODEL_PATH.exists():
        print(f"Error: Face model not found at {FACE_MODEL_PATH}")
        sys.exit(1)

    if not EYE_MODEL_PATH.exists():
        print(f"Error: Eye model not found at {EYE_MODEL_PATH}")
        sys.exit(1)

    # Load models
    try:
        face_model, eye_model, face_cascade = load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # Run detection
    try:
        run_detection(face_model, eye_model, face_cascade)
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
