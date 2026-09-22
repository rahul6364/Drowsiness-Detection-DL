# DriverGuard AI

This repository keeps the original machine learning detector in place and adds a modern FastAPI + React dashboard around it.

## Project structure

- `scripts/integrated_detection.py` contains the original working ML detection flow.
- `models/` stores the existing TensorFlow and PyTorch weights.
- `backend/` exposes the detector through a FastAPI server and WebSocket stream.
- `frontend/` contains the React + Vite + Tailwind dashboard.

## Architecture

Browser -> React dashboard -> WebSocket -> FastAPI -> existing OpenCV + CNN + MobileNetV2 detector -> webcam -> results back to UI.

## Keep the existing model logic intact

The current weights and inference flow are preserved:

- Face model: TensorFlow CNN
- Eye model: MobileNetV2 in PyTorch
- Fusion: 30% face + 70% eye
- Alert threshold: 70%

The backend reuses these values rather than retraining or replacing them.

## Run the backend

```bash
cd backend
python -m venv .venv
# Windows
# .venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The API runs at:

- http://localhost:8000/health
- http://localhost:8000/api/system
- ws://localhost:8000/ws

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, typically http://localhost:5173.

## Notes

- The backend loads the models once during startup and reuses them for every frame.
- The WebSocket sends structured detection results including the drowsiness score, alert state, eye status, and a base64 JPEG frame.
- The original standalone detection script remains runnable independently.

## Existing ML script

The original project script can still be run directly:

```bash
python scripts/integrated_detection.py
```

This keeps the original OpenCV window workflow working while the dashboard uses the new web integration layer.


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

- **Python:** 3.8 - 3.12 (recommended: Python 3.11 or 3.12). Python 3.13+ is **not supported** by TensorFlow.
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
   python3.12 -m venv venv        # Use Python 3.12 explicitly
   source venv/bin/activate        # Linux/macOS
   # or
   venv\Scripts\activate           # Windows
   ```

   > **Important:** If your system default Python is 3.13+, you **must** specify `python3.12` or `python3.11` when creating the venv. TensorFlow does not support Python 3.13+.

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

**Using webcam (default):**
```bash
python scripts/run_detection.py
```

**Using a video file:**
```bash
python scripts/run_detection.py path/to/video.mp4
```

**Integrated detection (face + eye models combined):**
```bash
# Webcam
python scripts/integrated_detection.py

# Video file
python scripts/integrated_detection.py path/to/video.mp4
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
