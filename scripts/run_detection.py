#!/usr/bin/python3
"""
Real-time Driver Drowsiness Detection Script

This script uses a pre-trained CNN model to detect driver drowsiness
via webcam feed.

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
        MODELS_DIR / "drowsiness_cnn.h5",
    ]
    
    for path in model_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"Model not found. Please train the model first.\n"
        f"Expected locations:\n"
        f"  - {model_paths[0]}\n"
        f"  - {model_paths[1]}"
    )


def load_class_names():
    """Load class names from JSON file or use defaults."""
    class_names_path = MODELS_DIR / "class_names.json"
    
    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            return json.load(f)
    
    return ["Drowsy", "Non Drowsy"]


def preprocess_face(face_img, img_size=IMG_SIZE):
    """Preprocess face image for model prediction."""
    resized = cv2.resize(face_img, img_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype("float32") / 255.0
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
    cv2.namedWindow(window_name)
    
    drowsy_frame_count = 0
    
    print("\nStarting drowsiness detection...")
    print("Press 'q' or close the window to quit.\n")
    
    try:
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
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
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face and predict
                face_roi = frame[y:y+h, x:x+w]
                preprocessed = preprocess_face(face_roi)
                prob = float(model.predict(preprocessed, verbose=0)[0][0])
                
                # Determine drowsiness
                is_drowsy = prob < DROWSY_THRESHOLD
                drowsiness_pct = (1 - prob) * 100 if class_names[0] == "Drowsy" else prob * 100
                
                if is_drowsy:
                    drowsy_frame_count += 1
                else:
                    drowsy_frame_count = 0
                
                # Set status based on consecutive frames
                if drowsy_frame_count >= FRAMES_THRESHOLD:
                    status_text = f"DROWSY! WAKE UP! ({drowsiness_pct:.1f}%)"
                    status_color = (0, 0, 255)  # Red
                else:
                    status_text = f"Awake ({100 - drowsiness_pct:.1f}%)"
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
            
            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")
    
    return 0


if __name__ == "__main__":
    exit(main())
