# Driver Drowsiness Detection using Deep Learning

A real-time drowsiness detection system that uses Convolutional Neural Networks (CNN) to identify signs of driver fatigue through facial feature analysis. This project aims to enhance road safety by alerting drivers when drowsiness is detected.

## Authors

- **Mohammadreza Hendiani**
- **Aakash Vashist**
- **Negin Ghanei**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

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

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd Drowsiness-Detection
```

2. Install required dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

3. Download the dataset:
   - Download the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) from Kaggle
   - Extract the dataset to your preferred location
   - Update the `data_dir` path in the notebook

## Usage

### Training the Model

1. Open the Jupyter notebook:
```bash
jupyter notebook drowsiness_detection.ipynb
```

2. Update the dataset path in Cell 2:
```python
data_dir = "/path/to/your/Driver Drowsiness Dataset (DDD)"
```

3. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Build the CNN model
   - Train the model (10 epochs)
   - Save the trained model as `drowsiness_cnn.h5`

### Real-time Detection

To use the real-time drowsiness detection system:

1. Ensure you have a working webcam
2. Run the webcam detection cells in the notebook (Cells 9-11)
3. The system will:
   - Detect faces using Haar Cascade
   - Identify eyes in detected faces
   - Predict drowsiness probability for each eye
   - Display real-time alerts when drowsiness is detected

**Detection Parameters**:
- `DROWSY_THRESHOLD`: 0.60 (probability threshold for drowsiness)
- `FRAMES_THRESHOLD`: 15 (consecutive drowsy frames before alert)

**Controls**:
- Press `q` to quit the detection window

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
├── LICENSE                        # Project license
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

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
