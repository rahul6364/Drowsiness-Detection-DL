"""
Installation Verification Script for Driver Drowsiness Detection
Run this script to verify all required packages are installed correctly.

Usage: python verify_installation.py
"""

import sys

print("=" * 60)
print("Driver Drowsiness Detection - Installation Verification")
print("=" * 60)
print()

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Track installation status
all_installed = True

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  → GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"     • {gpu.name}")
    else:
        print(f"  → No GPU detected (CPU mode)")
except ImportError as e:
    print(f"✗ TensorFlow not installed!")
    print(f"  Error: {e}")
    all_installed = False

# Check OpenCV
try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")

    # Check if Haar cascades are available
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"

    import os
    if os.path.exists(face_cascade_path) and os.path.exists(eye_cascade_path):
        print(f"  → Haar Cascades found")
    else:
        print(f"  → Warning: Haar Cascades not found")
except ImportError as e:
    print(f"✗ OpenCV not installed!")
    print(f"  Error: {e}")
    all_installed = False

# Check NumPy
try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy not installed!")
    print(f"  Error: {e}")
    all_installed = False

# Check Matplotlib
try:
    import matplotlib
    print(f"✓ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib not installed!")
    print(f"  Error: {e}")
    all_installed = False

# Check Jupyter
try:
    import jupyter_core
    print(f"✓ Jupyter installed (core version: {jupyter_core.__version__})")
except ImportError:
    try:
        import notebook
        print(f"✓ Jupyter Notebook version: {notebook.__version__}")
    except ImportError as e:
        print(f"✗ Jupyter not installed!")
        print(f"  Error: {e}")
        all_installed = False

# Check IPython Kernel
try:
    import ipykernel
    print(f"✓ IPyKernel version: {ipykernel.__version__}")
except ImportError as e:
    print(f"✗ IPyKernel not installed!")
    print(f"  Error: {e}")
    all_installed = False

print()
print("=" * 60)

if all_installed:
    print("✓ All required packages are installed successfully!")
    print()
    print("Next steps:")
    print("1. Download the dataset from Kaggle")
    print("2. Run: jupyter notebook drowsiness_detection.ipynb")
    print("3. Update the data_dir path in Cell 2")
    print("4. Train the model by running all cells")
else:
    print("✗ Some packages are missing!")
    print()
    print("To install missing packages, run:")
    print("  pip install -r requirements.txt")
    print()
    print("Or install individually:")
    print("  pip install tensorflow opencv-python numpy matplotlib jupyter ipykernel")

print("=" * 60)

# Optional: Check webcam availability
print()
print("Checking webcam availability...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Webcam detected and accessible")
        cap.release()
    else:
        print("✗ Webcam not detected or not accessible")
        print("  Make sure a webcam is connected and you have permissions")
except Exception as e:
    print(f"✗ Error checking webcam: {e}")

print()
