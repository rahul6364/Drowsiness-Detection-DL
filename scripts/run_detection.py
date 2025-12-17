#!/usr/bin/python3
"""
Real-time Driver Drowsiness Detection Script

This script uses a pre-trained CNN model to detect driver drowsiness
via webcam feed.

FIXES APPLIED:
- Convert BGR to RGB before prediction (model trained on RGB)
- Remove double normalization (model has built-in Rescaling layer)

Usage:
    python run_detection.py

Controls:
    - Press 'q' or close the window to quit
"""

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Detection parameters
IMG_SIZE = (64, 64)
DROWSY_THRESHOLD = 0.5  # Probability below this = drowsy
FRAMES_THRESHOLD = 15   # Consecutive frames before alert


def find_model():
    """Find the trained model file."""
    model_paths = [
        MODELS_DIR / "drowsiness_model.keras",
        MODELS_DIR / "drowsiness_cnn.keras",
        MODELS_DIR / "drowsiness_cnn.h5",
    ]
    
    for path in model_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"Model not found. Please train the model first.\n"
        f"Expected locations:\n" +
        "\n".join(f"  - {p}" for p in model_paths)
    )


def load_class_names():
    """Load class names from JSON file or use defaults."""
    class_names_path = MODELS_DIR / "class_names.json"
    
    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            return json.load(f)
    
    # Default: Class 0 = Drowsy, Class 1 = Non Drowsy
    return ["Drowsy", "Non Drowsy"]


def preprocess_face(face_img, img_size=IMG_SIZE):
    """
    Preprocess face image for model prediction.
    
    IMPORTANT: The model includes a Rescaling(1./255) layer internally,
    so we should NOT normalize here. Just resize and convert to RGB.
    """
    # Resize to model's expected input size
    resized = cv2.resize(face_img, img_size)
    
    # Convert BGR (OpenCV) to RGB (TensorFlow training format)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 but DO NOT divide by 255
    # The model's built-in Rescaling layer handles normalization
    normalized = rgb.astype("float32")
    
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)


def main():
    """Main detection loop."""
    # Load model
    print("Loading model...")
    model_path = find_model()
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Load class names
    class_names = load_class_names()
    print(f"Class names: {class_names}")
    
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found. Please check your camera connection.")
        return 1
    
    window_name = "Driver Drowsiness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    drowsy_frame_count = 0
    
    print("\nStarting drowsiness detection...")
    print("Press 'q' or close the window to quit.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(60, 60)
            )
            
            status_text = "Scanning..."
            status_color = (255, 255, 255)  # White
            
            for (x, y, w, h) in faces:
                # Draw face rectangle (green by default)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face and predict
                face_roi = frame[y:y+h, x:x+w]
                preprocessed = preprocess_face(face_roi)
                
                # Get prediction probability
                prob = float(model.predict(preprocessed, verbose=0)[0][0])
                
                # Model interpretation:
                # prob close to 0 = Class 0 (Drowsy)
                # prob close to 1 = Class 1 (Non Drowsy)
                is_drowsy = prob < DROWSY_THRESHOLD
                drowsiness_pct = (1 - prob) * 100  # Higher percentage = more drowsy
                
                if is_drowsy:
                    drowsy_frame_count += 1
                else:
                    drowsy_frame_count = 0
                
                # Set status based on consecutive drowsy frames
                if drowsy_frame_count >= FRAMES_THRESHOLD:
                    status_text = f"DROWSY! WAKE UP! ({drowsiness_pct:.1f}%)"
                    status_color = (0, 0, 255)  # Red
                    # Change rectangle to red for alert
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                else:
                    alertness = 100 - drowsiness_pct
                    status_text = f"Awake ({alertness:.1f}% alert)"
                    status_color = (0, 255, 0)  # Green
                
                # Only process first detected face
                break
            
            if len(faces) == 0:
                status_text = "No face detected"
                status_color = (0, 255, 255)  # Yellow
                drowsy_frame_count = 0
            
            # Display status
            cv2.putText(
                frame, status_text,
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, status_color, 2
            )
            
            cv2.imshow(window_name, frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")
    
    return 0


if __name__ == "__main__":
    exit(main())
