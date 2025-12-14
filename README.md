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

4. **Download dataset** from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) and extract to `data/`

5. **Train model**:
   ```bash
   cd notebooks
   jupyter notebook drowsiness_detection.ipynb
   # Run all cells through Section 5
   ```

6. **Run real-time detection** (using pre-trained model):
   ```bash
   python scripts/run_detection.py  # Press 'q' to quit
   ```

For detailed instructions, see the [Installation](#installation) section below.

## Overview

Driver drowsiness is a major cause of road accidents worldwide. This project implements a deep learning-based solution to detect drowsiness in real-time using a webcam feed. The system analyzes facial features to determine if a driver is becoming drowsy and triggers an alert to prevent potential accidents.

## Features

- **Real-time Detection**: Processes live webcam feed to detect drowsiness
- **High Accuracy**: Achieves >99% validation accuracy on the test dataset
- **CNN-based Model**: Custom convolutional neural network architecture optimized for drowsiness detection
- **Data Augmentation**: Implements rotation, flipping, and contrast adjustment for robust training
- **Visual Alerts**: Provides on-screen warnings when drowsiness is detected
- **Haar Cascade Integration**: Uses OpenCV's face detection for preprocessing
- **Model Evaluation**: Includes confusion matrix and classification report
- **Early Stopping**: Prevents overfitting with automatic training termination

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
- **Total Parameters**: ~1.1M (~4.36 MB)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Installation

### System Requirements

- **Python**: 3.8 or higher (recommended: Python 3.11)
- **RAM**: Minimum 4GB (8GB recommended for training)
- **Webcam**: Required for real-time detection
- **Storage**: At least 5GB for dataset and models

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | >=2.10.0 | Deep learning framework |
| opencv-python | >=4.5.0 | Computer vision and webcam processing |
| numpy | >=1.19.0 | Numerical computations |
| matplotlib | >=3.3.0 | Data visualization |
| seaborn | >=0.12.0 | Statistical visualization |
| scikit-learn | >=1.0.0 | ML metrics and evaluation |
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
pip install -r requirements.txt
```

**Step 6: Verify webcam access**
```bash
ls /dev/video*
```

---

#### Windows

**Step 1: Install Python**
- Download Python from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Recommended: Python 3.11

**Step 2: Open Command Prompt or PowerShell**
```cmd
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
pip install -r requirements.txt
```

**Step 6: Install Visual C++ Redistributables** (if needed)
- TensorFlow may require [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

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
pip install -r requirements.txt
```

**Step 6: Grant camera permissions**
- Go to System Preferences → Security & Privacy → Camera
- Allow Terminal (or your IDE) to access the camera

**Note for Apple Silicon (M1/M2/M3 Macs)**:
```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

---

## Usage

### Training the Model

#### Step 1: Download the Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) and extract to `data/`:

```
Drowsiness-Detection/
├── data/
│   └── Driver Drowsiness Dataset (DDD)/
│       ├── Drowsy/
│       └── Non Drowsy/
```

#### Step 2: Run the Training Notebook

```bash
cd notebooks
jupyter notebook drowsiness_detection.ipynb
```

Run cells 1-5 (Setup through Training & Evaluation).

#### Step 3: Training Output

The notebook will:
- Train the CNN model for 10 epochs (with early stopping)
- Display accuracy and loss plots
- Show confusion matrix and classification report
- Save models to `models/` directory

---

### Real-time Drowsiness Detection

#### Prerequisites

1. Working webcam connected to your computer
2. Trained model files in `models/` directory
3. Good lighting for better face detection

#### Running Detection

**Method 1: Using Jupyter Notebook**
- Open `notebooks/drowsiness_detection.ipynb`
- Skip to Section 6 (Real-Time Detection)
- Run all cells in that section

**Method 2: Standalone Script**
```bash
python scripts/run_detection.py
```

#### Detection Parameters

You can adjust these in the detection code:

```python
DROWSY_THRESHOLD = 0.5    # Probability threshold (0.0-1.0)
FRAMES_THRESHOLD = 15     # Consecutive frames before alert
```

#### Controls

- **Press `q`**: Quit the detection window
- **Close window**: Also stops detection

---

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
├── notebooks/
│   └── drowsiness_detection.ipynb    # Main training & detection notebook
├── models/
│   ├── drowsiness_model.keras        # Trained model (Keras format)
│   ├── drowsiness_cnn.h5             # Trained model (HDF5 format)
│   ├── drowsiness_checkpoint.keras   # Best checkpoint during training
│   └── class_names.json              # Class label mapping
├── data/
│   └── Driver Drowsiness Dataset (DDD)/
│       ├── Drowsy/                   # Drowsy face images
│       └── Non Drowsy/               # Non-drowsy face images
├── scripts/
│   ├── run_detection.py              # Standalone detection script
│   └── verify_installation.py        # Installation verification
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Project license
└── README.md                         # This file
```

## Model Files

Two model formats are saved:
- **drowsiness_model.keras**: Native Keras format (recommended)
- **drowsiness_cnn.h5**: Legacy HDF5 format

Both contain the same trained weights and can be used interchangeably.

## How It Works

1. **Face Detection**: OpenCV's Haar Cascade classifier detects faces in each video frame
2. **Preprocessing**: Face images are resized to 64x64 pixels and normalized
3. **Prediction**: The CNN model predicts the probability of drowsiness (0-1)
4. **Alert Logic**: If drowsiness probability exceeds the threshold for consecutive frames, an alert is triggered
5. **Visual Feedback**: Real-time visual indicators show drowsiness levels and warnings

## Future Improvements

- Add audio alerts for better driver notification
- Implement multi-face detection for monitoring multiple passengers
- Add mobile deployment support (TensorFlow Lite)
- Include head pose estimation for additional drowsiness indicators
- Implement LSTM layers for temporal pattern recognition
- Add logging and analytics for drowsiness patterns

## Troubleshooting

### Webcam Issues
- **Linux**: Check `/dev/video*` exists; add user to video group: `sudo usermod -a -G video $USER`
- **Windows**: Ensure no other application is using the webcam
- **macOS**: Grant camera permissions in System Preferences

### Poor Detection Accuracy
- Ensure good lighting conditions
- Position face 30-60cm from camera
- Avoid reflections on glasses
- Verify model file is correctly loaded

### Slow Performance
- Use GPU acceleration if available
- Reduce webcam resolution
- Check system resource usage

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