#!/usr/bin/python3
"""
Verify that all required packages are installed correctly.

Usage:
    python verify_installation.py
"""

import sys


def check_package(name, import_name=None):
    """Check if a package is installed and get its version."""
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def main():
    """Run all installation checks."""
    print("=" * 60)
    print("Driver Drowsiness Detection - Installation Verification")
    print("=" * 60)
    print()
    
    # Required packages
    packages = [
        ("tensorflow", "tensorflow", ">=2.10.0"),
        ("opencv-python", "cv2", ">=4.5.0"),
        ("numpy", "numpy", ">=1.19.0"),
        ("matplotlib", "matplotlib", ">=3.3.0"),
        ("seaborn", "seaborn", ">=0.12.0"),
        ("scikit-learn", "sklearn", ">=1.0.0"),
        ("jupyter", "jupyter", ">=1.0.0"),
    ]
    
    all_ok = True
    
    print("Checking required packages:")
    print("-" * 60)
    
    for pkg_name, import_name, min_version in packages:
        installed, version = check_package(pkg_name, import_name)
        
        if installed:
            status = f"✓ {pkg_name}: {version}"
            print(status)
        else:
            status = f"✗ {pkg_name}: NOT INSTALLED (required: {min_version})"
            print(status)
            all_ok = False
    
    print("-" * 60)
    print()
    
    # Check GPU availability
    print("Checking GPU support:")
    print("-" * 60)
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("○ No GPU detected (training will use CPU)")
    except Exception as e:
        print(f"○ Could not check GPU: {e}")
    
    print("-" * 60)
    print()
    
    # Check webcam availability
    print("Checking webcam:")
    print("-" * 60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✓ Webcam available: {width}x{height}")
            else:
                print("○ Webcam detected but could not read frame")
            cap.release()
        else:
            print("○ No webcam detected (required for real-time detection)")
    except Exception as e:
        print(f"○ Could not check webcam: {e}")
    
    print("-" * 60)
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All required packages are installed!")
        print("  You can proceed with training or detection.")
    else:
        print("✗ Some packages are missing.")
        print("  Please install them using: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())