# Driver Drowsiness Detection using Machine Learning

A real-time drowsiness detection system implementing multiple machine learning approaches following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology. This project compares CNN, Random Forest, and SVM models for driver fatigue detection through facial image analysis.

## Authors

- **Mohammadreza Hendiani**
- **Aakash Vashist**
- **Negin Ghanei**

## Table of Contents

- [Research Question](#research-question)
- [Quick Start Guide](#quick-start-guide)
- [Overview](#overview)
- [CRISP-DM Methodology](#crisp-dm-methodology)
- [Features](#features)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Model Comparison Results](#model-comparison-results)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Research Question

> **Can we accurately classify driver drowsiness states (Drowsy vs. Non-Drowsy) using facial image analysis, and which machine learning approach provides the best balance of accuracy and efficiency for real-time deployment?**

### Answer

Yes, we successfully developed models achieving **>99% validation accuracy**. Our CNN model provides the best balance of accuracy and real-time performance for deployment.

---

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

---

## Overview

Driver drowsiness is a major cause of road accidents worldwide, responsible for approximately 20% of all road crashes. This project implements a deep learning-based solution to detect drowsiness in real-time using a webcam feed. The system analyzes facial features to determine if a driver is becoming drowsy and triggers an alert to prevent potential accidents.

---

## CRISP-DM Methodology

This project follows the six phases of CRISP-DM (Cross-Industry Standard Process for Data Mining):

### Phase 1: Business & Data Understanding
- Defined research question and success criteria
- Identified the Driver Drowsiness Dataset (DDD) from Kaggle
- Established binary classification problem (Drowsy vs. Non-Drowsy)

### Phase 2: Data Preparation
- Loaded and preprocessed 41,793 images
- Applied data augmentation (rotation, flipping, contrast adjustment)
- Split data: 80% training, 20% validation
- Extracted flattened features for traditional ML models

### Phase 3: Exploratory Data Analysis (EDA)
- Analyzed class distribution (balanced dataset)
- Visualized sample images from both classes
- Examined pixel intensity distributions
- Generated statistical summaries

### Phase 4: Modeling
Three different machine learning models were implemented:
- **Model 1**: Convolutional Neural Network (CNN)
- **Model 2**: Random Forest Classifier
- **Model 3**: Support Vector Machine (SVM)

### Phase 5: Evaluation
- Calculated accuracy, precision, recall, F1-score for all models
- Generated confusion matrices
- Created ROC curves with AUC scores
- Performed comprehensive model comparison

### Phase 6: Deployment
- Implemented real-time webcam detection system
- Integrated best model (CNN) for production use
- Created standalone detection script

---

## Features

- **Real-time Detection**: Processes live webcam feed to detect drowsiness
- **Multi-Model Comparison**: Implements and compares 3 different ML approaches (CNN, Random Forest, SVM)
- **High Accuracy**: Achieves >99% validation accuracy on the test dataset
- **CNN-based Model**: Custom convolutional neural network architecture optimized for drowsiness detection
- **Data Augmentation**: Implements rotation, flipping, and contrast adjustment for robust training
- **Visual Alerts**: Provides on-screen warnings when drowsiness is detected
- **Haar Cascade Integration**: Uses OpenCV's face detection for preprocessing
- **Comprehensive Evaluation**: Includes confusion matrices, ROC curves, and classification reports
- **Early Stopping**: Prevents overfitting with automatic training termination

---

## Dataset

This project uses the **Driver Drowsiness Dataset (DDD)** from Kaggle:

**Dataset Link**: [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

### Dataset Details:
| Attribute | Value |
|-----------|-------|
| Total Images | 41,793 |
| Classes | 2 (Drowsy, Non-Drowsy) |
| Training Set | 33,435 images (80%) |
| Validation Set | 8,358 images (20%) |
| Image Size | 64x64 pixels (resized) |

---

## Models Implemented

### Model 1: Convolutional Neural Network (CNN)

**Justification:** CNNs are the gold standard for image classification, automatically learning hierarchical features from raw pixels. They can learn from low-level edges to high-level patterns, making them ideal for facial feature analysis.

**Architecture:**
- Input: 64×64×3 RGB images
- Data Augmentation Layer (flip, rotation, contrast)
- 3 Convolutional blocks (32→64→128 filters)
- MaxPooling after each conv block
- Dropout (50%) for regularization
- Dense layer (128 units)
- Sigmoid output for binary classification

**Parameters:** ~1.1M trainable parameters

### Model 2: Random Forest Classifier

**Justification:** Ensemble method combining multiple decision trees. Robust to overfitting, handles high-dimensional data well, and provides feature importance insights. Serves as a strong baseline for comparison.

**Configuration:**
- 100 decision trees
- Max depth: 20
- Min samples split: 5
- Min samples leaf: 2

### Model 3: Support Vector Machine (SVM)

**Justification:** Effective for binary classification with high-dimensional data. Uses RBF kernel to capture non-linear relationships in the data.

**Configuration:**
- Kernel: RBF (Radial Basis Function)
- C (regularization): 1.0
- Feature scaling: StandardScaler

---

## Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **CNN** | **99.92%** | **99.91%** | **99.93%** | **99.92%** | **0.9999** |
| Random Forest | 97.85% | 97.62% | 98.12% | 97.87% | 0.9951 |
| SVM | 96.50% | 96.23% | 96.81% | 96.52% | 0.9912 |

### Key Findings

1. **Best Model:** CNN achieves the highest performance across all metrics
2. **Recall Priority:** For safety-critical applications, high recall (99.93%) ensures drowsy drivers are not missed
3. **Real-time Capability:** CNN provides efficient inference for live webcam processing
4. **Traditional ML Baseline:** Random Forest provides competitive results without deep learning

---

## Model Architecture

The primary drowsiness detection model (CNN) is built using TensorFlow/Keras with the following architecture:

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

---

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
| pandas | >=1.3.0 | Data manipulation |
| matplotlib | >=3.3.0 | Data visualization |
| seaborn | >=0.12.0 | Statistical visualization |
| scikit-learn | >=1.0.0 | Traditional ML and evaluation metrics |
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

Run all cells to:
1. Load and preprocess the dataset
2. Perform EDA with visualizations
3. Train CNN, Random Forest, and SVM models
4. Evaluate and compare all models
5. Generate confusion matrices and ROC curves

#### Step 3: Training Output

The notebook will:
- Train all three models
- Display accuracy and loss plots
- Show confusion matrices and classification reports
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
- Skip to Section 6 (Deployment)
- Run the real-time detection cells

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

The CNN model achieves excellent performance on the validation set:

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|----------------|------------|--------------|----------|
| 1     | 89.14%         | 0.2236     | 99.74%       | 0.0115   |
| 5     | 99.58%         | 0.0126     | 99.90%       | 0.0026   |
| 10    | 99.72%         | 0.0077     | 99.92%       | 0.0024   |

**Final Model Performance**:
- **Validation Accuracy**: 99.92%
- **Validation Loss**: 0.0024

The model shows minimal overfitting and excellent generalization to unseen data.

---

## Project Structure

```
Drowsiness-Detection/
├── notebooks/
│   └── drowsiness_detection.ipynb    # Main training & detection notebook (CRISP-DM)
├── models/
│   ├── drowsiness_model.keras        # Trained model (Keras format)
│   ├── drowsiness_cnn.h5             # Trained model (HDF5 format)
│   ├── drowsiness_checkpoint.keras   # Best checkpoint during training
│   ├── class_names.json              # Class label mapping
│   ├── model_comparison.png          # Performance comparison chart
│   ├── confusion_matrices.png        # Confusion matrices for all models
│   └── roc_curves.png                # ROC curves comparison
├── data/
│   └── Driver Drowsiness Dataset (DDD)/
│       ├── Drowsy/                   # Drowsy face images
│       └── Non Drowsy/               # Non-drowsy face images
├── scripts/
│   ├── run_detection.py              # Standalone detection script
│   └── verify_installation.py        # Installation verification
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
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

---

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

### Training Issues
- Verify dataset is correctly extracted to `data/` directory
- Check available RAM (8GB+ recommended)
- Use GPU if available for faster training

---

## Future Improvements

- [ ] Add audio alerts for better driver notification
- [ ] Implement multi-face detection for monitoring multiple passengers
- [ ] Add mobile deployment support (TensorFlow Lite)
- [ ] Include head pose estimation for additional drowsiness indicators
- [ ] Implement LSTM layers for temporal pattern recognition
- [ ] Add logging and analytics for drowsiness patterns

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

- Dataset provided by Ismail Nasri on [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- OpenCV for computer vision tools
- TensorFlow/Keras team for the deep learning framework
- scikit-learn for traditional ML implementations

## References

- Driver Drowsiness Dataset (DDD): https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
- CRISP-DM Methodology: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining
- OpenCV Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades
- TensorFlow Documentation: https://www.tensorflow.org/

---

**Note**: This is an educational project for drowsiness detection developed as a capstone project following CRISP-DM methodology. For production deployment in safety-critical applications, additional testing, validation, and regulatory compliance would be required.