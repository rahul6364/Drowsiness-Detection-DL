# Driver Drowsiness Detection using Machine Learning

A real-time drowsiness detection system implementing multiple machine learning approaches following the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology. This project compares **four machine learning models** (CNN, Random Forest, SVM, and MobileNetV2) for driver fatigue detection through facial and eye image analysis.

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
- [Datasets](#datasets)
- [Models Implemented](#models-implemented)
- [Model Comparison Results](#model-comparison-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Research Question

> **Can we accurately classify driver drowsiness states (Drowsy vs. Non-Drowsy) using facial image analysis, and which machine learning approach provides the best balance of accuracy and efficiency for real-time deployment?**

### Answer

Yes, we successfully developed models achieving **>99% validation accuracy**. Our comparison of four different approaches shows that the CNN model provides the best balance of accuracy and real-time performance, while the MobileNetV2 eye-detection model offers a lightweight alternative for embedded deployment.

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

4. **Download datasets** from Kaggle:
   - [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) → extract to `data/`
   - [Open-Closed Eyes Dataset](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset) → extract to `data/archive/`

5. **Train models**:
   ```bash
   cd notebooks
   jupyter notebook drowsiness_detection.ipynb
   # Run all cells through Phase 5
   ```

6. **Run real-time detection** (using pre-trained model):
   ```bash
   python scripts/run_detection.py  # Press 'q' to quit
   ```

---

## Overview

Driver drowsiness is a major cause of road accidents worldwide, responsible for approximately 20% of all road crashes. This project implements a machine learning-based solution to detect drowsiness in real-time using a webcam feed. The system analyzes both full facial images and eye states to determine if a driver is becoming drowsy and triggers an alert to prevent potential accidents.

---

## CRISP-DM Methodology

This project follows the six phases of CRISP-DM (Cross-Industry Standard Process for Data Mining):

### Phase 1: Business & Data Understanding
- Defined research question and success criteria
- Identified two complementary datasets (DDD for faces, Open-Closed Eyes for eye states)
- Established binary classification problem (Drowsy vs. Non-Drowsy)

### Phase 2: Data Preparation
- **DDD Dataset:** Loaded and preprocessed 41,793 face images
- **Eyes Dataset:** Loaded and preprocessed 174,756 eye images
- Applied data augmentation (rotation, flipping, contrast adjustment)
- Split data: 80% training, 20% validation
- Extracted flattened features for traditional ML models

### Phase 3: Exploratory Data Analysis (EDA)
- Analyzed class distribution (balanced dataset)
- Visualized sample images from both classes
- Examined pixel intensity distributions
- Generated statistical summaries

### Phase 4: Modeling
Four different machine learning models were implemented:
- **Model 1:** Convolutional Neural Network (CNN) - TensorFlow/Keras
- **Model 2:** Random Forest Classifier - scikit-learn
- **Model 3:** Support Vector Machine (SVM) - scikit-learn
- **Model 4:** MobileNetV2 Eye State CNN - PyTorch

### Phase 5: Evaluation
- Calculated accuracy, precision, recall, F1-score for all models
- Generated confusion matrices
- Created ROC curves with AUC scores
- Performed comprehensive 4-model comparison

### Phase 6: Eye State Detection Model
- Trained MobileNetV2 on Open-Closed Eyes dataset
- Implemented transfer learning from ImageNet
- Achieved high accuracy on eye state classification

### Phase 7: Integrated Detection System
- Combined face model (30%) + eye model (70%) using weighted fusion
- Implemented automatic eye region extraction from face
- Alert threshold: Combined score >= 70% triggers drowsiness warning

### Phase 8: Deployment
- Implemented real-time webcam detection system
- Integrated both models for robust detection
- Created standalone detection scripts

---

## Features

- **Real-time Detection:** Processes live webcam feed to detect drowsiness
- **Multi-Model Comparison:** Implements and compares **4 different ML approaches** (CNN, Random Forest, SVM, MobileNetV2)
- **Integrated Detection:** Combines face and eye models using weighted fusion (30% face + 70% eye)
- **Dual Detection Approach:** Supports both full-face and eye-specific detection
- **High Accuracy:** Achieves >99% validation accuracy on test datasets
- **Framework Diversity:** Demonstrates proficiency in both TensorFlow and PyTorch
- **Data Augmentation:** Implements rotation, flipping, and contrast adjustment
- **Visual Alerts:** Provides on-screen warnings when drowsiness is detected
- **Haar Cascade Integration:** Uses OpenCV's face detection for preprocessing
- **Comprehensive Evaluation:** Includes confusion matrices, ROC curves, and classification reports

---

## Datasets

This project uses **two complementary datasets** from Kaggle:

### 1. Driver Drowsiness Dataset (DDD)
**Link:** [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)

| Attribute | Value |
|-----------|-------|
| Total Images | 41,793 |
| Classes | 2 (Drowsy, Non-Drowsy) |
| Training Set | 33,435 images (80%) |
| Validation Set | 8,358 images (20%) |
| Image Size | 64×64 pixels (resized) |
| Use | Full-face drowsiness detection |

### 2. Open-Closed Eyes Dataset
**Link:** [Open-Closed Eyes Dataset](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset)

| Attribute | Value |
|-----------|-------|
| Training Images | 139,804 (106,482 open + 33,322 closed) |
| Validation Images | 27,961 |
| Test Images | 6,991 |
| Classes | 2 (Open = Awake, Closed = Drowsy) |
| Image Size | 224×224 pixels (MobileNetV2 standard) |
| Use | Eye-specific state detection |

---

## Models Implemented

### Model 1: Convolutional Neural Network (CNN) - TensorFlow

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

### Model 4: MobileNetV2 Eye State CNN (PyTorch)

**Justification:** Lightweight architecture optimized for mobile and embedded devices. Uses transfer learning from ImageNet for robust feature extraction. Focuses specifically on eye states for fine-grained detection.

**Architecture:**
- Base: MobileNetV2 (ImageNet pre-trained)
- Modified final layer: 1280 → 2 classes
- Framework: PyTorch

**Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- LR Scheduler: ReduceLROnPlateau
- Data Augmentation: HorizontalFlip, Rotation, ColorJitter

**Parameters:** ~2.2M trainable parameters

---

## Model Comparison Results

| Model | Dataset | Framework | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|---------|-----------|----------|-----------|--------|----------|-----|
| **CNN (Face)** | DDD | TensorFlow | **99.92%** | 99.91% | **99.93%** | **99.92%** | **0.9999** |
| Random Forest | DDD | scikit-learn | 97.85% | 97.62% | 98.12% | 97.87% | 0.9951 |
| SVM | DDD | scikit-learn | 96.50% | 96.23% | 96.81% | 96.52% | 0.9912 |
| **MobileNetV2 (Eyes)** | Open-Closed Eyes | PyTorch | **99.56%** | 99.56% | 99.56% | 99.56% | 0.9992 |

### Key Findings

1. **Best Overall Model:** CNN (Face) achieves the highest performance across all metrics
2. **Eye Detection Success:** MobileNetV2 demonstrates that targeted eye detection achieves comparable accuracy
3. **Recall Priority:** For safety-critical applications, high recall (>99%) ensures drowsy drivers are not missed
4. **Framework Diversity:** Both TensorFlow and PyTorch implementations achieve excellent results
5. **Traditional ML Baseline:** Random Forest provides competitive results without deep learning

### Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Production Deployment | CNN (Face) | Highest accuracy, TensorFlow ecosystem |
| Mobile/Embedded | MobileNetV2 (Eyes) | Lightweight, efficient |
| Ensemble Approach | CNN + MobileNetV2 | Combines face + eye signals |
| Resource Constrained | Random Forest | No GPU required, interpretable |

---

## Installation

### System Requirements

- **Python:** 3.8 or higher (recommended: Python 3.11)
- **RAM:** Minimum 4GB (8GB recommended for training)
- **GPU:** Optional but recommended for deep learning models
- **Webcam:** Required for real-time detection
- **Storage:** At least 10GB for datasets and models

### Required Packages

```
tensorflow>=2.10.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.12.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Man2Dev/Drowsiness-Detection
   cd Drowsiness-Detection
   ```

2. Create virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python scripts/verify_installation.py
   ```

---

## Usage

### Training the Models

1. Download datasets from Kaggle and extract:
   ```
   Drowsiness-Detection/
   ├── data/
   │   ├── Driver Drowsiness Dataset (DDD)/
   │   │   ├── Drowsy/
   │   │   └── Non Drowsy/
   │   └── archive/
   │       ├── train/
   │       ├── val/
   │       └── test/
   ```

2. Run the training notebook:
   ```bash
   cd notebooks
   jupyter notebook drowsiness_detection.ipynb
   ```

3. Execute all cells to train all 4 models

### Real-time Detection

```bash
python scripts/run_detection.py
```

Press 'q' or close the window to quit.

---

## Project Structure

```
Drowsiness-Detection/
├── notebooks/
│   └── drowsiness_detection.ipynb    # Main CRISP-DM notebook (4 models)
├── models/
│   ├── drowsiness_cnn.keras          # CNN model (TensorFlow)
│   ├── drowsiness_cnn.h5             # CNN model (HDF5 format)
│   ├── eye_model.pth                 # MobileNetV2 model (PyTorch)
│   ├── class_names.json              # Class label mapping
│   └── *.png                         # Evaluation visualizations
├── data/
│   ├── Driver Drowsiness Dataset (DDD)/
│   │   ├── Drowsy/
│   │   └── Non Drowsy/
│   └── archive/                      # Open-Closed Eyes Dataset
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── run_detection.py              # Standalone detection script
│   ├── integrated_detection.py       # Combined face + eye detection
│   └── verify_installation.py        # Installation verification
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # Project license
└── README.md                         # This file
```

---

## Troubleshooting

### Common Issues

**Webcam not detected:**
- Linux: Check `/dev/video*` exists; add user to video group
- Windows: Ensure no other application is using the webcam
- macOS: Grant camera permissions in System Preferences

**Poor detection accuracy:**
- Ensure good lighting conditions
- Position face 30-60cm from camera
- Avoid reflections on glasses

**PyTorch/TensorFlow conflicts:**
- Use separate virtual environments if needed
- Install PyTorch with CUDA support for GPU acceleration

**Out of memory during training:**
- Reduce batch size in the notebook
- Use GPU if available
- Train models separately

---

## Future Improvements

- [x] Ensemble model combining face and eye predictions (Implemented: 30% face + 70% eye weighted fusion)
- [ ] Add audio alerts for better driver notification
- [ ] Mobile deployment using TensorFlow Lite and ONNX
- [ ] Multi-face detection for passenger monitoring
- [ ] Head pose estimation for additional drowsiness indicators
- [ ] LSTM layers for temporal pattern recognition
- [ ] Model drift monitoring for production systems

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

- Driver Drowsiness Dataset (DDD) by Ismail Nasri on [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- Open-Closed Eyes Dataset by Sehriyar Memmedli on [Kaggle](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset)
- OpenCV for computer vision tools
- TensorFlow/Keras and PyTorch teams for deep learning frameworks
- scikit-learn for traditional ML implementations

## References

- Driver Drowsiness Dataset (DDD): https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
- Open-Closed Eyes Dataset: https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset
- CRISP-DM Methodology: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining
- MobileNetV2 Paper: https://arxiv.org/abs/1801.04381
- OpenCV Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades

---

**Note:** This is an educational project for drowsiness detection developed as a capstone project following CRISP-DM methodology. For production deployment in safety-critical applications, additional testing, validation, and regulatory compliance would be required.