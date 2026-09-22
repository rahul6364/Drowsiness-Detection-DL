Academic Deep Learning Project Report Analysis: Driver Drowsiness Detection System
This analysis provides an evidence-based audit of the Driver Drowsiness Detection repository. Every metric, hyperparameter, parameter count, and architectural detail presented here is verified directly from the project source files, actual executed cell outputs in 

drowsiness_detection.ipynb
, saved model checkpoints in 

models/
, scripts, backend implementation, and frontend codebase.

TASK 1 — Comprehensive Analysis of the CNN Model (Face-Based Drowsiness Detection)
1.1 Model Configuration & Hyperparameters
Verified from 

drowsiness_detection.ipynb
 (Cells 5, 7, 21, 22) and 

drowsiness_cnn.keras
:

Model Name: DrowsinessCNN
Input Resolution: $64 \times 64 \times 3$ (RGB)
Data Augmentation Pipeline (Training Only):
RandomFlip("horizontal")
RandomRotation(factor=0.1)
RandomContrast(factor=0.1)
Normalization: Rescaling(1./255)
Optimizer: Adam (learning_rate=0.001, jit_compile=False)
Loss Function: Binary Cross-Entropy (binary_crossentropy)
Batch Size: 32
Training Epochs Configured: 10
Callbacks:
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
ModelCheckpoint(filepath='../models/drowsiness_checkpoint.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
Weights Restored: Restored from the end of Epoch 9 (Best validation loss: $0.000383$)
1.2 Detailed Layer Architecture & Parameter Audit
Verified by querying the layer hierarchy of the saved artifact 

drowsiness_cnn.keras
:

Layer Index	Layer Type	Output Shape	Activation	Kernel / Pool Size	Param #
0	InputLayer	(None, 64, 64, 3)	—	—	0
1	Sequential (Augmentation)	(None, 64, 64, 3)	—	Flip, Rotation, Contrast	0
2	Rescaling	(None, 64, 64, 3)	—	$1/255$	0
3	Conv2D	(None, 64, 64, 32)	ReLU	$3 \times 3$, padding='same'	896
4	MaxPooling2D	(None, 32, 32, 32)	—	$2 \times 2$	0
5	Conv2D	(None, 32, 32, 64)	ReLU	$3 \times 3$, padding='same'	18,496
6	MaxPooling2D	(None, 16, 16, 64)	—	$2 \times 2$	0
7	Conv2D	(None, 16, 16, 128)	ReLU	$3 \times 3$, padding='same'	73,856
8	MaxPooling2D	(None, 8, 8, 128)	—	$2 \times 2$	0
9	Flatten	(None, 8192)	—	—	0
10	Dropout	(None, 8192)	—	rate = 0.5	0
11	Dense	(None, 128)	ReLU	—	1,048,704
12	Dense (Output)	(None, 1)	Sigmoid	—	129
Total Parameters: 1,142,081 (4.36 MB)
Trainable Parameters: 1,142,081 (100%)
Non-trainable Parameters: 0
Optimizer Parameter Footprint: 2,284,164 (8.71 MB)
1.3 Dataset & Partitioning
Dataset Name: Driver Drowsiness Dataset (DDD)
Source: Kaggle (ismailnasri20/driver-drowsiness-dataset-ddd)
Total Image Count: 41,793
Class Names & Mapping: ['Drowsy', 'Non Drowsy'] (Class 0: Drowsy, Class 1: Non Drowsy)
Training Partition: 33,435 images (80.0%, 1,045 batches of 32)
Validation Partition: 8,358 images (20.0%, 262 batches of 32)
Test Partition: None. No separate held-out test split exists in the notebook for the DDD dataset.
1.4 Actual Notebook Training History (Epoch-by-Epoch)
Extracted directly from cell 22 standard output:

Epoch	Training Loss	Training Accuracy	Validation Loss	Validation Accuracy	Checkpoint Status
1/10	0.2491	87.84% (0.8784)	0.0400	98.91% (0.98911)	Saved (drowsiness_checkpoint.keras)
2/10	0.0342	98.91% (0.9891)	0.0104	99.69% (0.99689)	Saved (drowsiness_checkpoint.keras)
3/10	0.0179	99.43% (0.9943)	0.0031	99.90% (0.99904)	Saved (drowsiness_checkpoint.keras)
4/10	0.0136	99.58% (0.9958)	0.0013	99.96% (0.99964)	Saved (drowsiness_checkpoint.keras)
5/10	0.0148	99.55% (0.9955)	0.0016	99.96%	Not improved
6/10	0.0088	99.72% (0.9972)	0.000531	99.99% (0.99988)	Saved (drowsiness_checkpoint.keras)
7/10	0.0086	99.73% (0.9973)	0.0033	99.93%	Not improved
8/10	0.0097	99.69% (0.9969)	0.0160	99.49%	Not improved
9/10	0.0073	99.73% (0.9973)	0.000383	99.99%	Best validation loss
10/10	0.0058	99.82% (0.9982)	0.0132	99.44%	Restored Epoch 9 weights
1.5 Post-Training Evaluation Metrics on Validation Set (8,358 Samples)
Extracted from cell 30, cell 33, cell 35, cell 37, and cell 41:

Evaluation Split: Validation Set (8,358 samples)
Accuracy: 0.9999 (99.99%) (8,357 correct out of 8,358)
Precision: 1.0000 (100.0%)
Recall: 0.9997 (99.97%)
F1-Score: 0.9999 (99.99%)
ROC-AUC: 1.0000
Classification Report (Validation Set):
text
              precision    recall  f1-score   support
      Drowsy       1.00      1.00      1.00      4497
  Non-Drowsy       1.00      1.00      1.00      3861
    accuracy                           1.00      8358
   macro avg       1.00      1.00      1.00      8358
weighted avg       1.00      1.00      1.00      8358
Confusion Matrix (Validation Set):
Actual Drowsy, Predicted Drowsy (True Negatives for drowsiness class): 4,497
Actual Drowsy, Predicted Non-Drowsy (False Positives): 0
Actual Non-Drowsy, Predicted Drowsy (False Negatives): 1
Actual Non-Drowsy, Predicted Non-Drowsy (True Positives): 3,860
Total Evaluated: 8,358
TASK 2 — Comprehensive Analysis of MobileNetV2 Eye Model
2.1 Model Architecture & Transfer Learning
Verified from 

drowsiness_detection.ipynb
 (Cells 45, 49, 51, 57, 59):

Base Architecture: MobileNetV2 (torchvision)
Pre-trained Source: ImageNet (weights="IMAGENET1K_V1")
Classifier Adaptation: Replaced final classification layer eye_model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
Total Parameters: 2,226,434
Trainable Parameters: 2,226,434 (Full fine-tuning)
Non-trainable Parameters: 0
Input Resolution: $224 \times 224 \times 3$ (RGB)
Optimizer: Adam (lr=0.001)
Learning Rate Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=2)
Loss Function: CrossEntropyLoss()
Batch Size: 32
Target Epochs: 10
2.2 Data Augmentation & Normalization
Training Transforms:
Resize((224, 224))
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=15)
ColorJitter(brightness=0.2, contrast=0.2)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Validation & Test Transforms:
Resize((224, 224))
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
2.3 Dataset & Splits
Dataset Name: Open-Closed Eyes Dataset
Source: Kaggle (sehriyarmemmedli/open-closed-eyes-dataset)
Class Mapping: Class 0 = closed (DROWSY), Class 1 = open (AWAKE)
Dataset Partitions:
Training Split: 139,804 images (4,369 batches) — Class distribution: 23.8% closed (33,273), 76.2% open (106,531)
Validation Split: 27,961 images (874 batches)
Test Split: 6,991 images (219 batches)
Total Dataset Size: 174,756 images
2.4 Actual MobileNetV2 Training History (Epoch-by-Epoch)
Extracted directly from cell 61 output:

Epoch	Training Loss	Training Accuracy	Validation Loss	Validation Accuracy	Epoch Time	Checkpoint Event
1/10	0.0429	98.50%	0.0264	99.03%	502.0s	Model initialized
2/10	0.0299	98.96%	0.0242	99.15%	504.1s	Improvement
3/10	0.0254	99.12%	0.0207	99.26%	508.8s	Improvement
4/10	0.0228	99.20%	0.0228	99.16%	510.6s	Loss plateau
5/10	0.0209	99.26%	0.0164	99.40%	507.2s	Improvement
6/10	0.0192	99.35%	0.0170	99.38%	511.4s	Steady
7/10	0.0185	99.37%	0.0148	99.48%	513.2s	Improvement
8/10	0.0170	99.38%	0.0137	99.54%	512.1s	Saved Best Model (eye_model_best.pth)
9/10	0.0164	99.42%	0.0151	99.49%	508.7s	Completed
10/10	0.0157	99.45%	0.0134	99.52%	509.5s	Completed
Best Validation Accuracy: 99.54% (Achieved at Epoch 8)
Total Training Execution Time: $\approx 5,087.6 \text{ s}$ ($\approx 84.8 \text{ minutes}$)
2.5 Actual MobileNetV2 Test Set Evaluation (6,991 Test Samples)
Extracted from cell 63, cell 65, cell 67, cell 69, and cell 73:

Evaluation Split: Test Set (Held-out test partition of 6,991 images)
Test Loss: 0.0161
Test Accuracy: 99.41% (0.9941) (6,950 correct out of 6,991)
Test Precision: 0.9946 (99.46%)
Test Recall: 0.9977 (99.77%)
Test F1-Score: 0.9962 (99.62%)
Test ROC-AUC: 0.9998
Classification Report (Test Set):
text
                 precision    recall  f1-score   support
Closed (DROWSY)       0.99      0.98      0.99      1666
   Open (AWAKE)       0.99      1.00      1.00      5325
       accuracy                           0.99      6991
      macro avg       0.99      0.99      0.99      6991
   weighted avg       0.99      0.99      0.99      6991
Confusion Matrix (Test Set):
Actual Closed (DROWSY), Predicted Closed (DROWSY): 1,637
Actual Closed (DROWSY), Predicted Open (AWAKE): 29
Actual Open (AWAKE), Predicted Closed (DROWSY): 12
Actual Open (AWAKE), Predicted Open (AWAKE): 5,313
Total Evaluated: 6,991
TASK 3 — Traditional Machine Learning Baselines (Random Forest & SVM)
Verified from 

drowsiness_detection.ipynb
 (Cells 11, 24, 26, 30, 33, 35, 37, 41):

3.1 Feature Extraction Protocol
Images from the Driver Drowsiness Dataset (DDD) were resized to $64 \times 64 \times 3$, normalized to $[0, 1]$, and flattened into 1D feature vectors of size 12,288.
Subsets evaluated:
Training subset: 8,000 samples
Validation subset: 2,000 samples (Class support: 1,106 Drowsy, 894 Non-Drowsy)
3.2 Random Forest Classifier
Configuration: n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1.
Evaluation Split: Validation Subset (2,000 samples)
Accuracy: 0.9985 (99.85%) (1,997 / 2,000 correct)
Precision: 0.9967 (99.67%)
Recall: 1.0000 (100.0%)
F1-Score: 0.9983 (99.83%)
ROC-AUC: 1.0000
Confusion Matrix (2,000 samples):
Actual Drowsy, Predicted Drowsy: 1,103
Actual Drowsy, Predicted Non-Drowsy: 3
Actual Non-Drowsy, Predicted Drowsy: 0
Actual Non-Drowsy, Predicted Non-Drowsy: 894
Classification Report:
text
              precision    recall  f1-score   support
      Drowsy       1.00      1.00      1.00      1106
  Non-Drowsy       1.00      1.00      1.00       894
    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000
3.3 Support Vector Machine (SVM)
Configuration: Feature scaling using StandardScaler(); SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42).
Support Vectors Identified: 763
Evaluation Split: Validation Subset (2,000 samples)
Accuracy: 1.0000 (100.0%) (2,000 / 2,000 correct)
Precision: 1.0000 (100.0%)
Recall: 1.0000 (100.0%)
F1-Score: 1.0000 (100.0%)
ROC-AUC: 1.0000
Confusion Matrix (2,000 samples):
Actual Drowsy, Predicted Drowsy: 1,106
Actual Drowsy, Predicted Non-Drowsy: 0
Actual Non-Drowsy, Predicted Drowsy: 0
Actual Non-Drowsy, Predicted Non-Drowsy: 894
Classification Report:
text
              precision    recall  f1-score   support
      Drowsy       1.00      1.00      1.00      1106
  Non-Drowsy       1.00      1.00      1.00       894
    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000
TASK 4 — Model Comparison & Discrepancy Audit
4.1 Actual Reported Values in Notebook
Derived from Cells 30, 41, and 73:

Model	Dataset	Evaluation Split	Sample Count	Accuracy	Precision	Recall	F1-Score	ROC-AUC
CNN (Face)	DDD	Validation	8,358	0.9999	1.0000	0.9997	0.9999	1.0000
Random Forest	DDD	Validation (subset)	2,000	0.9985	0.9967	1.0000	0.9983	1.0000
SVM	DDD	Validation (subset)	2,000	1.0000	1.0000	1.0000	1.0000	1.0000
MobileNetV2 (Eyes)	Open-Closed Eyes	Test	6,991	0.9941	0.9946	0.9977	0.9962	0.9998
(Note: In the notebook cell 61, the best validation accuracy recorded for MobileNetV2 was 0.9954 / 99.54% at Epoch 8).

4.2 Discrepancy Analysis: README.md vs. Notebook Outputs
README.md Values (Lines 73–79):
markdown
| Model | Dataset | Framework | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|---------|-----------|----------|-----------|--------|----------|-----|
| CNN (Face) | DDD | TensorFlow | 99.92% | 99.91% | 99.93% | 99.92% | 0.9999 |
| Random Forest | DDD | scikit-learn | 97.85% | 97.62% | 98.12% | 97.87% | 0.9951 |
| SVM | DDD | scikit-learn | 96.50% | 96.23% | 96.81% | 96.52% | 0.9912 |
| MobileNetV2 (Eyes) | Open-Closed Eyes | PyTorch | 99.56% | 99.56% | 99.56% | 99.56% | 0.9992 |
Notebook Actual Values (Cells 30, 41, 63, 73):
CNN (Face): Accuracy: 99.99%, Precision: 100.00%, Recall: 99.97%, F1: 99.99%, AUC: 1.0000 (Validation split)
Random Forest: Accuracy: 99.85%, Precision: 99.67%, Recall: 100.00%, F1: 99.83%, AUC: 1.0000 (Validation subset)
SVM: Accuracy: 100.00%, Precision: 100.00%, Recall: 100.00%, F1: 100.00%, AUC: 1.0000 (Validation subset)
MobileNetV2 (Eyes): Accuracy: 99.41%, Precision: 99.46%, Recall: 99.77%, F1: 99.62%, AUC: 0.9998 (Test split)
Possible Reason:
Reason not established from project files. While 

PROJECT_DOCUMENTATION_SOURCE.md
 acknowledges discrepancies between high-level documentation and notebook executions, no log or script in the repository records the specific experiment from which the numbers 99.92%, 97.85%, 96.50%, and 99.56% were produced. For academic reporting, the verified notebook outputs must take precedence.

TASK 5 — Integrated Detection System (Face + Eye Fusion)
Verified from 

scripts/integrated_detection.py
, 

notebooks/drowsiness_detection.ipynb
 (Cells 76–80), and 

backend/detection/detector.py
:

5.1 Architecture & Components
Face Classification Model: Custom CNN (drowsiness_cnn.keras)
Eye Classification Model: MobileNetV2 (eye_model_best.pth)
Face Detector: OpenCV Haar Feature-based Cascade Classifier (haarcascade_frontalface_default.xml)
Parameters: scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
Input Resolutions:
Face Model: $64 \times 64 \times 3$
Eye Model: $224 \times 224 \times 3$
Weights:
Face Model Weight ($W_{\text{face}}$): 0.3 (30%)
Eye Model Weight ($W_{\text{eye}}$): 0.7 (70%)
Thresholds:
Alert Threshold: 70.0% (Combined score $\ge 0.70$)
Eye Closure Threshold: Probability $\ge 0.50$ (Class index 0 = closed)
5.2 Eye Crop Geometry
Extracted from detected face bounding box ($x, y, w, h$):

Vertical Region: $y_1 = y + 0.18h$ to $y_2 = y + 0.45h$
Left Eye Region: $x_{1,\text{left}} = x + 0.12w$ to $x_{2,\text{left}} = x + 0.42w$
Right Eye Region: $x_{1,\text{right}} = x + 0.58w$ to $x_{2,\text{right}} = x + 0.88w$
5.3 Mathematical Fusion Formulation
The system fuses the continuous probability scores from both deep learning pipelines:

$$\text{Eye Drowsiness Score} = \frac{P(\text{Left Eye Closed}) + P(\text{Right Eye Closed})}{2}$$

$$\text{Combined Score} = (0.3 \times \text{Face Drowsiness Score}) + (0.7 \times \text{Eye Drowsiness Score})$$

$$\text{Drowsiness Percentage} = \text{Combined Score} \times 100$$

$$\text{Alert Triggered} = \begin{cases} \text{True} & \text{if } \text{Drowsiness Percentage} \ge 70.0% \ \text{False} & \text{otherwise} \end{cases}$$

Fallback Strategy: If facial landmarks or eyes cannot be isolated ($left_eye$ is None or $right_eye$ is None), the system falls back to face-only inference: $$\text{Combined Score} = \text{Face Drowsiness Score}$$
5.4 Temporal & Sequential Logic
In the active production backend (

detector.py
) and 

integrated_detection.py
, evaluation is per-frame. Consecutive-frame accumulation was prototyped in notebook cell 80 (EYE_CLOSED_SECONDS = 2.0), but the deployed system computes real-time instant fusion.
TASK 6 — Runtime & System Performance Audit
A comprehensive search of the entire project repository (scripts, documentation, benchmark files, logs) was conducted for empirical operational measurements.

Performance Metric	Measured Value in Repository	Source / Evidence
FPS Benchmark (Offline / Formal)	Not measured in the current project.	Dynamically computed on live webcam feed ($30 / \Delta t$); no static benchmark dataset or recorded table exists.
Inference Latency (Face CNN)	Not measured in the current project.	No timestamp profiling per layer/model in repository.
Inference Latency (MobileNetV2)	Not measured in the current project.	No per-frame latency benchmarks recorded.
End-to-End Processing Time	Not measured in the current project.	Not measured in the current project.
WebSocket Stream Latency	Not measured in the current project.	Network ping/transfer latency is not benchmarked.
Backend Route Latency	Not measured in the current project.	No API benchmarking suite present.
Frontend Render Frequency	Not measured in the current project.	Determined dynamically by WebSocket event receipt.
Alert Response Time (Empirical)	Not measured in the current project.	Determined strictly by mathematical threshold crossing per frame.
CPU Utilization Profile	Not measured in the current project.	No CPU utilization logging script or output.
Memory Utilization Profile	Not measured in the current project.	No memory profiling artifacts present.
TASK 7 — Full-Stack Application Architecture (Frontend & Backend)
7.1 Backend Architecture (

backend/
)
Framework: FastAPI (Python) running on Uvicorn ASGI server
Lifecycle Management: FastAPI lifespan context manager loads the CNN model (drowsiness_cnn.keras), MobileNetV2 weights (eye_model_best.pth), and Haar Cascade detector once into global memory during server startup.
REST Endpoints:
GET /health: Returns system health, model load status, and alert threshold configuration.
GET /api/system: Delivers model metadata, framework types, image resolutions, fusion weights, and hardware acceleration device.
POST /api/start: Activates video capture (cv2.VideoCapture) and monitoring state.
POST /api/stop: Releases camera resource and halts stream.
POST /api/reset: Resets session alert count, duration, and metrics.
POST /api/mute: Toggles audio/visual alert suppression state.
WebSocket Endpoint (ws://localhost:8000/ws):
Captures live frames via OpenCV.
Feeds frames to process_frame() in 

backend/detection/detector.py
.
Encodes the annotated frame as Base64 JPEG.
Emits JSON payload at stream rate:
json
{
  "status": "awake" | "drowsy",
  "drowsiness_score": 24.50,
  "face_score": 15.20,
  "eye_score": 28.50,
  "left_eye": "OPEN" | "CLOSED",
  "right_eye": "OPEN" | "CLOSED",
  "fps": 0.0,
  "alert": false,
  "timestamp": "2026-09-22T00:00:00.000Z",
  "face": {"x": 180, "y": 95, "width": 240, "height": 240},
  "frame_data": "<base64_jpeg_string>",
  "alert_count": 0,
  "frames_processed": 142
}
7.2 Frontend Architecture (

frontend/
)
Framework & Tooling: React 18.3.1, Vite 5.4.10, Tailwind CSS 3.4.15
Component Layout (

frontend/src/App.jsx
):
Header Bar: System status pill (SYSTEM ONLINE), connection state (Connected / Disconnected), session clock, model indicator.
Live Driver Monitor: Real-time video frame rendering from Base64 stream, live indicator, dynamic alert overlay banner (DROWSINESS ALERT).
Temporal Graph: Recharts line chart displaying the past 60 temporal drowsiness score entries against a dashed 70% threshold line.
Driver Status Card: Real-time aggregated drowsiness percentage with qualitative classification pill (AWAKE, WARNING, DROWSY).
Model Analysis Panel: Decomposed breakdown displaying independent scores for Face Analysis (CNN), Eye Analysis (MobileNetV2), and Final Fusion Score.
Eye Status Card: Independent binary status badges for Left Eye (OPEN/CLOSED) and Right Eye (OPEN/CLOSED).
Alert & Session Metrics Panel: Running count of triggered alerts, timestamp of latest alert, session elapsed time, total frames processed.
Control Toolbar: Interactive controls for START MONITORING, STOP MONITORING, MUTE/UNMUTE ALERT, and RESET SESSION.
TASK 8 — Report-Ready Results (Structured for Academic Project Report)
SECTION A — CNN Results (Face Model)
Metric	Result	Evaluation Type	Source
Training Accuracy (Final Epoch 10)	99.82%	Training (33,435 samples)	Notebook Cell 22 Output
Training Loss (Final Epoch 10)	0.0058	Training (33,435 samples)	Notebook Cell 22 Output
Best Validation Accuracy (Epoch 9)	99.99%	Validation (8,358 samples)	Notebook Cell 22 Output
Best Validation Loss (Epoch 9)	0.000383	Validation (8,358 samples)	Notebook Cell 22 Output
Post-Training Accuracy	99.99%	Validation (8,358 samples)	Notebook Cell 30 & 41 Output
Precision	100.00%	Validation (8,358 samples)	Notebook Cell 30 & 41 Output
Recall	99.97%	Validation (8,358 samples)	Notebook Cell 30 & 41 Output
F1-Score	99.99%	Validation (8,358 samples)	Notebook Cell 30 & 41 Output
ROC-AUC	1.0000	Validation (8,358 samples)	Notebook Cell 37 & 41 Output
Confusion Matrix (TN / FP / FN / TP)	4497 / 0 / 1 / 3860	Validation (8,358 samples)	Notebook Cell 33 Display Data
SECTION B — MobileNetV2 Results (Eye Model)
Metric	Result	Evaluation Type	Source
Training Accuracy (Final Epoch 10)	99.45%	Training (139,804 samples)	Notebook Cell 61 Output
Training Loss (Final Epoch 10)	0.0157	Training (139,804 samples)	Notebook Cell 61 Output
Best Validation Accuracy (Epoch 8)	99.54%	Validation (27,961 samples)	Notebook Cell 61 Output
Best Validation Loss (Epoch 10)	0.0134	Validation (27,961 samples)	Notebook Cell 61 Output
Test Accuracy	99.41%	Test (6,991 held-out samples)	Notebook Cell 63 & 73 Output
Test Loss	0.0161	Test (6,991 held-out samples)	Notebook Cell 63 Output
Test Precision	99.46%	Test (6,991 held-out samples)	Notebook Cell 73 Output
Test Recall	99.77%	Test (6,991 held-out samples)	Notebook Cell 73 Output
Test F1-Score	99.62%	Test (6,991 held-out samples)	Notebook Cell 73 Output
Test ROC-AUC	0.9998	Test (6,991 held-out samples)	Notebook Cell 69 & 73 Output
Confusion Matrix (TN / FP / FN / TP)	1637 / 29 / 12 / 5313	Test (6,991 held-out samples)	Notebook Cell 67 Display Data
SECTION C — Traditional Machine Learning Results
Model	Evaluated Subset	Accuracy	Precision	Recall	F1-Score	ROC-AUC	Support Vectors / Trees	Source
Random Forest	Validation (2,000 samples)	99.85%	99.67%	100.00%	99.83%	1.0000	100 Trees (max_depth=20)	Notebook Cell 24, 30, 41
SVM (RBF)	Validation (2,000 samples)	100.00%	100.00%	100.00%	100.00%	1.0000	763 Support Vectors	Notebook Cell 26, 30, 41
SECTION D — Overall Model Comparison
Model	Dataset	Framework	Accuracy	Precision	Recall	F1-Score	ROC-AUC	Evaluation Split
CNN (Face)	DDD	TensorFlow	99.99%	1.0000	0.9997	0.9999	1.0000	Validation (8,358 samples)
Random Forest	DDD	scikit-learn	99.85%	0.9967	1.0000	0.9983	1.0000	Validation (2,000 subset)
SVM (RBF)	DDD	scikit-learn	100.00%	1.0000	1.0000	1.0000	1.0000	Validation (2,000 subset)
MobileNetV2 (Eyes)	Open-Closed Eyes	PyTorch	99.41%	0.9946	0.9977	0.9962	0.9998	Test (6,991 held-out)
SECTION E — Training Progression & Overfitting Analysis
CNN Convergence:
Initial epoch validation accuracy reached $98.91%$ immediately, converging to $>99.9%$ by Epoch 3.
Training loss dropped from $0.2491 \to 0.0058$; validation loss attained its minimum of $0.000383$ at Epoch 9.
Validation loss tracking closely with training loss, along with spatial dropout ($0.5$) and random flip/rotation augmentation, prevented catastrophic overfitting.
MobileNetV2 Convergence:
Started at $98.50%$ training accuracy and $99.03%$ validation accuracy on Epoch 1 due to pre-trained ImageNet representations.
Progressed monotonically to $99.45%$ training accuracy and $99.54%$ validation accuracy at Epoch 8.
Generalization was confirmed by the held-out test evaluation, retaining $99.41%$ accuracy and an ROC-AUC of $0.9998$.
SECTION F — Integrated System Specification
Face Model Weight ($W_{\text{face}}$): 0.3 (30%)
Eye Model Weight ($W_{\text{eye}}$): 0.7 (70%)
Fusion Mathematical Formula: $$\text{Score}{\text{combined}} = 0.3 \times \text{Score}{\text{face}} + 0.7 \times \left( \frac{P(\text{Left Eye Closed}) + P(\text{Right Eye Closed})}{2} \right)$$
System Alert Threshold: $\ge 70.0%$ combined score triggers drowsiness alarm
Face Detector Engine: OpenCV Haar Feature-based Cascade (haarcascade_frontalface_default.xml)
Input Resolutions: Face: $64 \times 64 \times 3$ (RGB) | Eyes: $224 \times 224 \times 3$ (RGB)
SECTION G — Application Implementation Breakdown
mermaid
flowchart LR
    subgraph Offline_ML [1. ML Model Performance]
        CNN[Face CNN: 99.99% Val Acc]
        MNV2[Eye MobileNetV2: 99.41% Test Acc]
    end
    subgraph RealTime [2. Real-Time Detection]
        Webcam[Webcam Stream] --> Haar[Haar Face Detector]
        Haar --> Fusion[Weighted Fusion: 0.3 Face + 0.7 Eyes]
        Fusion --> Alert[Threshold Check: 70%]
    end
    subgraph Backend_Layer [3. Backend System]
        FastAPI[FastAPI Server]
        WS[WebSocket /ws Endpoint]
        FastAPI --- WS
    end
    subgraph Frontend_Layer [4. Dashboard Frontend]
        React[React + Vite Dashboard]
        Charts[Recharts Temporal Trend]
        Overlay[Visual Alert Banners]
        React --- Charts
        React --- Overlay
    end
    Offline_ML --> RealTime
    RealTime --> Backend_Layer
    Backend_Layer -->|Base64 Frame + JSON| Frontend_Layer
ML Model Performance: Verified offline deep learning accuracy ($99.99%$ validation on DDD, $99.41%$ test on Open-Closed Eyes).
Real-Time Detection Performance: Combines face and eye probability distributions per frame; calculates dynamic FPS during runtime.
Backend Performance: Asynchronous FastAPI + WebSocket streaming server handling on-demand camera acquisition, frame encoding, and JSON broadcasting.
Frontend Performance: React dashboard displaying visual stream, Recharts temporal graphs, and system diagnostics. (Note: Operational benchmarks such as end-to-end latency in milliseconds or CPU profiling were not measured in the current repository).
SECTION H — Checklist of Project Artifacts vs. Academic Report Requirements
Standard Report Requirement	Project Status	Explanatory Note
Face Model Training Metrics	Available	Verified in notebook Cell 22 output
Face Model Validation Metrics	Available	Verified in notebook Cells 22, 30, 41
Face Model Held-Out Test Set	Not available	DDD dataset was split 80% train / 20% validation; no test folder used.
Face Model Confusion Matrix	Available	Verified in notebook Cell 33 plot
Eye Model Training Metrics	Available	Verified in notebook Cell 61 output
Eye Model Validation Metrics	Available	Verified in notebook Cell 61 output
Eye Model Held-Out Test Metrics	Available	Verified in notebook Cells 63, 65, 73 (6,991 images)
Eye Model Confusion Matrix	Available	Verified in notebook Cell 67 plot
Traditional ML Baselines (RF, SVM)	Available	Evaluated on 2,000 validation image feature vectors
Formal FPS Benchmark	Not available	Not measured in the current project (computed on-the-fly only)
Inference Latency Benchmark	Not measured	Not measured in the current project
Hardware Specification Sheet	Not available	Training hardware not explicitly specified in code
WebSocket Stream Latency Benchmark	Not measured	Not measured in the current project
FINAL OUTPUT
1. Verified Results
Face CNN Model: Achieved $99.82%$ training accuracy, $99.99%$ validation accuracy on 8,358 validation images, with $1.0000$ precision, $0.9997$ recall, $0.9999$ F1-score, and $1.0000$ ROC-AUC. Total trainable parameters: 1,142,081.
Eye MobileNetV2 Model: Fine-tuned on 139,804 training images. Achieved $99.54%$ best validation accuracy (Epoch 8) and $99.41%$ test accuracy on 6,991 held-out test images, with $0.9946$ precision, $0.9977$ recall, $0.9962$ F1-score, and $0.9998$ ROC-AUC. Total trainable parameters: 2,226,434.
Traditional ML Baselines: Random Forest scored $99.85%$ validation accuracy and SVM scored $100.00%$ validation accuracy on a 2,000-sample flattened feature subset.
Integration Layer: Mathematically formulated as $0.3 \times \text{Face Score} + 0.7 \times \text{Eye Score}$, triggering a warning when the combined score is $\ge 70.0%$.
2. Results Requiring Clarification
README.md Discrepancy: The root 

README.md
 reports CNN accuracy as $99.92%$, Random Forest as $97.85%$, SVM as $96.50%$, and MobileNetV2 as $99.56%$. However, the actual executed outputs in 

drowsiness_detection.ipynb
 evaluate to $99.99%$ (CNN), $99.85%$ (RF), $100.00%$ (SVM), and $99.41%$ test accuracy ($99.54%$ best validation) for MobileNetV2. Because the experimental source of the README numbers is not recorded in the repository files, the actual notebook outputs must be used in the academic report.
Lack of Face Model Test Set: The CNN model was evaluated exclusively on a $20%$ validation partition ($8,358$ images). In academic documentation, this must be accurately cited as validation accuracy rather than test accuracy.
3. Missing Measurements
Real-time inference latency (milliseconds per frame) for both models.
Empirical benchmark FPS under defined CPU/GPU specifications.
End-to-end WebSocket round-trip transmission latency.
Client-side CPU and memory resource consumption profiles.
4. Final Report Tables (Directly Usable in VTU Report)
Table I: Deep Learning Model Architecture and Training Configuration
Parameter / Attribute	Face Model (CNN)	Eye State Model (MobileNetV2)
Framework	TensorFlow / Keras	PyTorch / TorchVision
Input Dimensions	$64 \times 64 \times 3$ (RGB)	$224 \times 224 \times 3$ (RGB)
Base Architecture	Custom 3-Block CNN	MobileNetV2 (ImageNet Pre-trained)
Classifier Top	Flatten $\to$ Dropout(0.5) $\to$ Dense(128) $\to$ Dense(1)	Linear(1280, 2)
Total Parameters	1,142,081	2,226,434
Trainable Parameters	1,142,081	2,226,434
Optimizer	Adam ($\text{lr} = 0.001$)	Adam ($\text{lr} = 0.001$)
Loss Function	Binary Cross-Entropy	Cross-Entropy Loss
Batch Size	32	32
Training Epochs	10	10
Training Dataset	Driver Drowsiness Dataset (33,435 images)	Open-Closed Eyes Dataset (139,804 images)
Validation Dataset	Driver Drowsiness Dataset (8,358 images)	Open-Closed Eyes Dataset (27,961 images)
Test Dataset	None (Validation set only)	Open-Closed Eyes Dataset (6,991 images)
Table II: Experimental Results and Performance Comparison
Model	Evaluated Split	Sample Size	Accuracy (%)	Precision (%)	Recall (%)	F1-Score (%)	ROC-AUC
CNN (Face)	Validation	8,358	99.99	100.00	99.97	99.99	1.0000
Random Forest	Validation Subset	2,000	99.85	99.67	100.00	99.83	1.0000
SVM (RBF)	Validation Subset	2,000	100.00	100.00	100.00	100.00	1.0000
MobileNetV2 (Eyes)	Test (Held-out)	6,991	99.41	99.46	99.77	99.62	0.9998
Table III: Confusion Matrix Summary
Model	Evaluation Split	True Negatives (TN)	False Positives (FP)	False Negatives (FN)	True Positives (TP)
CNN (Face)	Validation (8,358 samples)	4,497 (Drowsy)	0	1	3,860 (Non-Drowsy)
Random Forest	Validation (2,000 samples)	1,103 (Drowsy)	3	0	894 (Non-Drowsy)
SVM	Validation (2,000 samples)	1,106 (Drowsy)	0	0	894 (Non-Drowsy)
MobileNetV2 (Eyes)	Test (6,991 samples)	1,637 (Closed)	29	12	5,313 (Open)
Results and Discussion
In this work, driver drowsiness detection was addressed using a dual-modality computer vision framework comprising a custom 3-block convolutional neural network for facial classification and a transfer learning MobileNetV2 architecture for eye-state detection. Empirical evaluation on the Driver Drowsiness Dataset demonstrated that the facial CNN converged to a validation accuracy of $99.99%$ with an F1-score of $0.9999$ and an ROC-AUC of $1.0000$ across $8,358$ validation images, while baseline Random Forest and Support Vector Machine classifiers achieved validation accuracies of $99.85%$ and $100.00%$ on a $2,000$-sample flattened feature subset. On the Open-Closed Eyes dataset, the MobileNetV2 model attained a peak validation accuracy of $99.54%$ at Epoch 8 and confirmed robust generalization on a dedicated, held-out test split of $6,991$ images with $99.41%$ accuracy ($6,950/6,991$ correct classifications), $99.46%$ precision, $99.77%$ recall, and an ROC-AUC of $0.9998$. To integrate both behavioral cues for real-time monitoring, a weighted fusion mechanism assigning $30%$ weight to facial drowsiness probability and $70%$ weight to average eye-closure probability was deployed alongside an OpenCV Haar Cascade face detector with an alert threshold of $70.0%$. The detection pipeline was integrated into a FastAPI backend streaming Base64 frames over WebSockets to a responsive React and Tailwind CSS dashboard with Recharts temporal telemetry, decoupling model inference from UI rendering. Operational hardware latency and empirical frame-rate benchmarks remain unmeasured in the static repository artifacts and constitute key objectives for future vehicular field validation.