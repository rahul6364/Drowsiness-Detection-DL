# Driver Drowsiness Detection using Deep Learning

A real-time drowsiness detection system that uses Convolutional Neural Networks (CNN) to identify signs of driver fatigue through facial feature analysis. This project aims to enhance road safety by alerting drivers when drowsiness is detected.

## Authors

- **Mohammadreza Hendiani**
- **Aakash Vashist**
- **Negin Ghanei**

## Table of Contents

- [Quick Start Guide](#quick-start-guide)
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

## Quick Start Guide

Get started in 5 minutes:

1. **Install Python 3.8+** for your OS ([Linux](#linux-ubuntudebianfedora) | [Windows](#windows) | [macOS](#macos))

2. **Clone and setup**:
   ```bash
   git clone https://github.com/Man2Dev/Drowsiness-Detection
   cd Drowsiness-Detection
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python scripts/verify_installation.py
   ```

4. **Download dataset** from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

5. **Train model**:
   ```bash
   jupyter notebook drowsiness_detection.ipynb
   # Update data_dir in Cell 2, then run all cells
   ```

6. **Run real-time detection** (using pre-trained model):
   ```bash
   python run_detection.py  # Press 'q' to quit
   ```

For detailed instructions, see the [Installation](#installation) section below.

## Overview

Driver drowsiness is a major cause of road accidents worldwide. This project implements a deep learning-based solution to detect drowsiness in real-time using a webcam feed. The system analyzes eye patterns and facial features to determine if a driver is becoming drowsy and triggers an alert to prevent potential accidents.

## Features

- **Real-time Detection**: Processes live webcam feed to detect drowsiness
- **High Accuracy**: Achieves >99% validation accuracy on the test dataset
- **CNN-based Model**: Custom convolutional neural network architecture optimized for drowsiness detection
- **Data Augmentation**: Implements rotation, flipping, and contrast adjustment for robust training
- **Visual Alerts**: Provides on-screen warnings when drowsiness is detected
- **Haar Cascade Integration**: Uses OpenCV's face and eye detection for preprocessing

## Dataset

This project uses the **Driver Drowsiness Dataset (DDD)** from Kaggle:

**Dataset Link**: [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

### Dataset Details:
- **Total Images**: 41,793
- **Classes**: 2 (Drowsy, Non-Drowsy)
- **Training Set**: 33,435 images (80%)
- **Validation Set**: 8,358 images (20%)
- **Image Size**: 64x64 pixels (resized)

## Model Architecture

The drowsiness detection model is built using TensorFlow/Keras with the following architecture:

### Network Structure:
1. **Input Layer**: 64x64x3 RGB images
2. **Data Augmentation Layer**:
   - Random horizontal flip
   - Random rotation (10%)
   - Random contrast adjustment (10%)
3. **Preprocessing**: Pixel rescaling (normalization to 0-1)
4. **Convolutional Blocks**:
   - Conv2D (32 filters, 3x3) + ReLU + MaxPooling
   - Conv2D (64 filters, 3x3) + ReLU + MaxPooling
   - Conv2D (128 filters, 3x3) + ReLU + MaxPooling
5. **Fully Connected Layers**:
   - Flatten layer
   - Dropout (50%)
   - Dense (128 units, ReLU)
   - Output Dense (1 unit, Sigmoid) - Binary classification

### Model Parameters:
- **Total Parameters**: 1,142,081 (~4.36 MB)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Installation

### System Requirements

- **Python**: 3.7 or higher (recommended: Python 3.8-3.11)
- **RAM**: Minimum 4GB (8GB recommended for training)
- **Webcam**: Required for real-time detection
- **Storage**: At least 5GB for dataset and models

### Required Packages

The following Python packages are required:

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.x (>=2.10.0) | Deep learning framework |
| opencv-python | >=4.5.0 | Computer vision and webcam processing |
| numpy | >=1.19.0 | Numerical computations |
| matplotlib | >=3.3.0 | Data visualization |
| jupyter | >=1.0.0 | Interactive notebook environment |
| ipykernel | >=6.0.0 | Jupyter kernel support |

### Installation Instructions by Operating System

#### Linux (Ubuntu/Debian/Fedora)

**Step 1: Install Python and pip**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora
sudo dnf install python3 python3-pip
```

**Step 2: Install system dependencies for OpenCV**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Fedora
sudo dnf install mesa-libGL glib2 libSM libXext libXrender
```

**Step 3: Clone the repository**
```bash
git clone https://github.com/Man2Dev/Drowsiness-Detection
cd Drowsiness-Detection
```

**Step 4: Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 5: Install Python packages**
```bash
pip install --upgrade pip
pip install tensorflow opencv-python numpy matplotlib jupyter ipykernel
```

**Step 6: Verify webcam access**
```bash
# Check if webcam is available
ls /dev/video*
```

---

#### Windows

**Step 1: Install Python**
- Download Python from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Recommended: Python 3.8-3.11

**Step 2: Open Command Prompt or PowerShell**
```cmd
# Verify Python installation
python --version
pip --version
```

**Step 3: Clone the repository**
```cmd
git clone https://github.com/Man2Dev/Drowsiness-Detection
cd Drowsiness-Detection
```

**Step 4: Create a virtual environment (recommended)**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Step 5: Install Python packages**
```cmd
pip install --upgrade pip
pip install tensorflow opencv-python numpy matplotlib jupyter ipykernel
```

**Step 6: Install Visual C++ Redistributables** (if needed)
- TensorFlow may require [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

**Note for Windows**: If you encounter webcam access issues, ensure your antivirus/firewall allows Python to access the camera.

---

#### macOS

**Step 1: Install Homebrew** (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2: Install Python**
```bash
brew install python@3.11
```

**Step 3: Clone the repository**
```bash
git clone https://github.com/Man2Dev/Drowsiness-Detection
cd Drowsiness-Detection
```

**Step 4: Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 5: Install Python packages**
```bash
pip install --upgrade pip
pip install tensorflow opencv-python numpy matplotlib jupyter ipykernel
```

**Step 6: Grant camera permissions**
- Go to System Preferences → Security & Privacy → Camera
- Allow Terminal (or your IDE) to access the camera

**Note for Apple Silicon (M1/M2/M3 Macs)**:
```bash
# Install TensorFlow optimized for Apple Silicon
pip install tensorflow-macos tensorflow-metal
pip install opencv-python numpy matplotlib jupyter ipykernel
```

---

### Alternative: Using requirements.txt

Create a `requirements.txt` file with the following content:

```
tensorflow>=2.10.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

Then install all packages at once:

**Linux/macOS:**
```bash
pip install -r requirements.txt
```

**Windows:**
```cmd
pip install -r requirements.txt
```

---

### Download the Dataset

1. **Create a Kaggle account** (if you don't have one): [kaggle.com](https://www.kaggle.com/)

2. **Install Kaggle API** (optional, for command-line download):
```bash
pip install kaggle
```

3. **Download the dataset**:
   - **Option A**: Manual download from [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
   - **Option B**: Using Kaggle API:
   ```bash
   # Set up Kaggle API credentials first
   kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd
   unzip driver-drowsiness-dataset-ddd.zip -d ./dataset
   ```

4. **Extract the dataset** to your preferred location

---

### Verify Installation

Run this Python script to verify all packages are installed correctly:

```python
import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed!")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("OpenCV not installed!")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not installed!")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib not installed!")

print("\nAll required packages are installed successfully!")
```

Save this as `verify_installation.py` and run:
```bash
python scripts/verify_installation.py
```

## Usage

### Training the Model

#### Step 1: Launch Jupyter Notebook

**Linux/macOS:**
```bash
# Activate virtual environment if you created one
source venv/bin/activate

# Start Jupyter
jupyter notebook drowsiness_detection.ipynb
```

**Windows:**
```cmd
# Activate virtual environment if you created one
venv\Scripts\activate

# Start Jupyter
jupyter notebook drowsiness_detection.ipynb
```

#### Step 2: Configure Dataset Path

In the notebook, update **Cell 2** with your dataset path:

**Linux/macOS:**
```python
data_dir = "/home/username/datasets/Driver Drowsiness Dataset (DDD)"
```

**Windows:**
```python
data_dir = r"C:\Users\YourUsername\datasets\Driver Drowsiness Dataset (DDD)"
```

**Note**: Use raw string (`r""`) in Windows to handle backslashes correctly.

#### Step 3: Train the Model

Run all cells sequentially (or select "Run All" from the Cell menu):

1. **Cells 1-3**: Import libraries and load the dataset
2. **Cell 4**: Visualize sample images
3. **Cell 5**: Build the CNN architecture
4. **Cell 6**: Train the model (takes ~5-10 minutes on GPU, ~30-60 minutes on CPU)
5. **Cell 7**: View training history plots
6. **Cell 8**: Save the trained model

**Training Options:**
- Modify `epochs = 10` in Cell 6 to train for more/fewer epochs
- Adjust `batch_size = 32` in Cell 2 based on your available RAM

---

### Real-time Drowsiness Detection

#### Prerequisites for Real-time Detection

1. **Working webcam** connected to your computer
2. **Trained model file** (`drowsiness_cnn.h5` or `drowsiness_model.keras`)
3. **Good lighting** for better face detection

#### Running Real-time Detection

**Method 1: Using Jupyter Notebook**

1. Restart the kernel (Kernel → Restart)
2. Run **Cell 12** (imports)
3. Run **Cell 13** (load model)
4. Run **Cell 14** (helper functions)
5. Run **Cell 15** (webcam detection loop)

**Method 2: Standalone Python Script**

Create a file `run_detection.py`:

```python
import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("drowsiness_cnn.h5")
class_names = ['Drowsy', 'Non Drowsy']

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def preprocess_eye(eye_img, img_height=64, img_width=64):
    eye_resized = cv2.resize(eye_img, (img_width, img_height))
    eye_rgb = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)
    eye_array = eye_rgb.astype("float32") / 255.0
    return np.expand_dims(eye_array, axis=0)

def predict_drowsiness(eye_img):
    x = preprocess_eye(eye_img)
    prob = float(model.predict(x, verbose=0)[0][0])
    return prob

# Webcam capture
cap = cv2.VideoCapture(0)
DROWSY_THRESHOLD = 0.60
FRAMES_THRESHOLD = 15
drowsy_frames = 0

print("Starting drowsiness detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    is_drowsy_this_frame = False

    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            prob = predict_drowsiness(eye_img)
            label = f"Drowsy: {prob:.2f}"
            color = (0, 0, 255) if prob >= DROWSY_THRESHOLD else (0, 255, 0)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(roi_color, label, (ex, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if prob >= DROWSY_THRESHOLD:
                is_drowsy_this_frame = True
        break

    if is_drowsy_this_frame:
        drowsy_frames += 1
    else:
        drowsy_frames = 0

    if drowsy_frames >= FRAMES_THRESHOLD:
        cv2.putText(display, "WAKE UP! YOU ARE DROWSY!",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run the script:
```bash
python run_detection.py
```

#### Detection Parameters

You can adjust these parameters in the code:

```python
DROWSY_THRESHOLD = 0.60    # Probability threshold (0.0-1.0)
                           # Higher = more strict detection

FRAMES_THRESHOLD = 15      # Consecutive frames before alert
                           # Higher = fewer false alarms
```

#### Controls

- **Press `q`**: Quit the detection window
- **ESC**: Alternative quit method (may work on some systems)

#### Troubleshooting Real-time Detection

**Webcam not detected:**
- **Linux**: Check `/dev/video*` devices exist and you have permissions
  ```bash
  sudo usermod -a -G video $USER
  ```
- **Windows**: Ensure no other application is using the webcam
- **macOS**: Grant camera permissions in System Preferences

**Poor detection accuracy:**
- Ensure good lighting conditions
- Position face 30-60cm from the camera
- Avoid reflections on glasses
- Ensure the trained model file is loaded correctly

**Slow performance:**
- Reduce `batch_size` in predictions
- Lower webcam resolution
- Use GPU acceleration (if available)

## Results

### Training Performance

The model achieves excellent performance on the validation set:

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|----------------|------------|--------------|----------|
| 1     | 89.14%         | 0.2236     | 99.74%       | 0.0115   |
| 5     | 99.58%         | 0.0126     | 99.90%       | 0.0026   |
| 10    | 99.72%         | 0.0077     | 99.92%       | 0.0024   |

**Final Model Performance**:
- **Validation Accuracy**: 99.92%
- **Validation Loss**: 0.0024

The model shows minimal overfitting and excellent generalization to unseen data.

## Project Structure

```
Drowsiness-Detection/
├── drowsiness_detection.ipynb    # Main Jupyter notebook with complete workflow
├── drowsiness_cnn.h5              # Trained model (HDF5 format)
├── drowsiness_model.keras         # Trained model (Keras native format)
├── requirements.txt               # Python package dependencies
├── verify_installation.py         # Script to verify installation
├── LICENSE                        # Project license
├── README.md                      # This file (complete documentation)
└── .gitignore                     # Git ignore rules
```

### Quick Start Files

After cloning the repository, you'll find these helpful files:

- **[requirements.txt](requirements.txt)**: Install all dependencies with `pip install -r requirements.txt`
- **[verify_installation.py](verify_installation.py)**: Run `python verify_installation.py` to check your setup
- **[drowsiness_detection.ipynb](drowsiness_detection.ipynb)**: Main notebook for training and detection

## Model Files

Two model files are provided:
- **drowsiness_cnn.h5**: Legacy HDF5 format
- **drowsiness_model.keras**: Native Keras format (recommended)

Both contain the same trained weights and can be used interchangeably.

## How It Works

1. **Face Detection**: The system uses OpenCV's Haar Cascade classifier to detect faces in each video frame
2. **Eye Detection**: Within detected faces, the system identifies eye regions
3. **Preprocessing**: Eye images are resized to 64x64 pixels and normalized
4. **Prediction**: The CNN model predicts the probability of drowsiness (0-1)
5. **Alert Logic**: If drowsiness probability exceeds the threshold for consecutive frames, an alert is triggered
6. **Visual Feedback**: Real-time visual indicators show drowsiness levels and warnings

## Future Improvements

- Add audio alerts for better driver notification
- Implement multi-face detection for monitoring multiple drivers
- Add mobile deployment support (TensorFlow Lite)
- Include head pose estimation for additional drowsiness indicators
- Implement LSTM layers for temporal pattern recognition
- Add logging and analytics for drowsiness patterns

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

- Dataset provided by Ismail Nasri on [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- OpenCV for computer vision tools
- TensorFlow/Keras team for the deep learning framework

## References

- Driver Drowsiness Dataset (DDD): https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
- OpenCV Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades
- TensorFlow Documentation: https://www.tensorflow.org/

---

**Note**: This is an educational project for drowsiness detection. For production deployment in safety-critical applications, additional testing, validation, and regulatory compliance would be required.
