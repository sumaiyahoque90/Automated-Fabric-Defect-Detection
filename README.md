# FabDetector: Automated Fabric Defect Detection using Computer Vision

**CSE400C Capstone Project · Summer 2026 · East West University, Dhaka**

A lightweight YOLO-based object detection framework for automated garment quality inspection. The system detects **where** a defect is on the fabric and **what type** it is, and is deployed as a Streamlit web app (**FabDetector**) that works on images and videos.

---

## Overview

Small surface defects (holes, knots/slubs, stains, yarn faults) can make a finished garment unsellable, especially for export-grade buyers. Manual inspection is still the norm in most factories, but human accuracy drops with fatigue, poor lighting, and repetitive fabric patterns.

This project trains and compares four lightweight YOLO architectures on a custom, field-collected dataset and deploys the best one as a usable inspection tool.

**Objectives**
- Build an automated garment defect detection system using deep learning.
- Train and compare YOLOv8s, YOLOv9s, YOLOv11s, and YOLO26s on the same dataset.
- Design a complete preprocessing → training → evaluation pipeline.
- Identify the most accurate and efficient model for industrial deployment.

---

## Dataset

- Images were collected on-site through repeated visits to **APT Sweater Limited (APT Garments), Gazipur**, a 100% export-oriented sweater manufacturer.
- All **1,887 instances** were manually annotated with bounding boxes in **Roboflow**.
- Split: 70% train / 20% validation / 10% test.

| Fabric Class | Original Instances | Train (Augmented) |
| --- | --- | --- |
| Defect-Free | 470 | 470 |
| Spot | 443 | 443 |
| Hole | 367 | 367 |
| Knot Slub | 308 | ≈ 1,232 |
| Yarn | 299 | ≈ 1,196 |
| **Total** | **1,887** | **≈ 3,708** |

> The dataset is not included in this repository because of its size.

---

## Methodology

1. **Data collection**: manual field collection at the garment factory.
2. **Annotation**: bounding boxes in Roboflow.
3. **Preprocessing & augmentation**: offline Albumentations pass for minority classes (Knot Slub, Yarn).
4. **Training**: transfer learning from pretrained YOLO weights, fine-tuned end-to-end.
5. **Evaluation**: Precision, Recall, F1-score, mAP@50, mAP@50:95.
6. **Deployment**: Streamlit web app with image and video inference.

**Training setup**

| Setting | Value |
| --- | --- |
| Input size | 640 × 640 |
| Optimizer | AdamW, lr0 = 1e-3, cosine decay (lrf = 0.01), 5-epoch warmup |
| Epochs | Up to 400 (Kaggle GPU), early stopping patience = 20 |
| Augmentation | Mosaic (1.0), mixup (0.1), copy-paste (0.1), flips, HSV and rotation jitter |
| Regularization | Label smoothing (0.1) |
| Precision | Automatic mixed precision (AMP) |

---

## Results

| Model | Best Epoch | Precision (%) | Recall (%) | F1 (%) | mAP@50 (%) | mAP@50:95 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **YOLOv8s** | 372 | **96.27** | **94.28** | **95.27** | **98.56** | **75.83** |
| YOLOv9s | 377 | 93.71 | 89.86 | 91.75 | 95.73 | 66.69 |
| YOLOv11s | 381 | 93.07 | 89.58 | 91.29 | 96.26 | 71.24 |
| YOLO26s | 349 | 94.02 | 93.48 | 93.75 | 97.95 | 70.75 |

**YOLOv8s per-class mAP@50**

| Class | mAP@50 |
| --- | --- |
| Defect-free | 0.993 |
| Spot | 0.732 |
| Hole | 0.719 |
| Knot Slub | 0.693 |
| Yarn | 0.588 |

**Key findings**
- **YOLOv8s** is the most balanced model, with stable convergence and the best score on every metric.
- **YOLO26s** converges fastest (best epoch 349) while staying competitive.
- Hole and spot defects are easiest to detect; **yarn and knot-slub** are the hardest, since they resemble normal fabric texture.
- Errors are mostly missed detections, not misclassifications.

---

## FabDetector Web App

| Feature | Description |
| --- | --- |
| Framework | Streamlit |
| Model | YOLOv8s (best performing, `best.pt`) |
| Input | Images (`.jpg`, `.png`, `.jpeg`) and videos (`.mp4`, `.avi`, `.mov`) |
| Confidence slider | Adjustable threshold (0.0 – 1.0) |
| Output | Bounding boxes, class labels, confidence scores |
| Summary | Count of detected defects by category |

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/sumaiyahoque90/Automated-Fabric-Defect-Detection.git
cd Automated-Fabric-Defect-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Make sure `best.pt` is in the same folder as `app.py`.

---

## Repository Structure

```
├── app.py                      # FabDetector Streamlit web app
├── best.pt                     # Trained YOLOv8s weights
├── requirements.txt            # Python dependencies
├── YOLOv8s.ipynb               # Training notebook: YOLOv8s
├── YOLOV9s.ipynb               # Training notebook: YOLOv9s
├── YOLOv11s.ipynb              # Training notebook: YOLOv11s
├── YOLO26s.ipynb               # Training notebook: YOLO26s
├── Automated_Fabrics_Defect_Detection-Final_Report_CSE400C.pdf
└── Automated_Fabric_Defect _Detection Final Slide.pdf
```

---

## Limitations & Future Work

**Limitations**
- Only five defect categories, which does not cover the full range seen in industry.
- Yarn and knot-slub defects are visually close to normal fabric texture.
- Images were captured under controlled conditions; real factory lighting and backgrounds vary.
- Only lightweight YOLO models were tested (no RT-DETR, EfficientDet, or transformer-based detectors).
- No real-time benchmarking on edge hardware.

**Future work**
- Expand the dataset with more fabric types, lighting conditions, and defect categories.
- Explore attention mechanisms and transformer-based feature extraction.
- Deploy on edge AI hardware (e.g. NVIDIA Jetson) for low-latency inspection.
- Integrate with industrial cameras, conveyor systems, and MES for full automation.

---

## Documents

| Resource | Link |
| --- | --- |
| Final report (PDF) | [In this repo](Automated_Fabrics_Defect_Detection-Final_Report_CSE400C.pdf) |
| Final slides (PDF) | [In this repo](Automated_Fabric_Defect%20_Detection%20Final%20Slide.pdf) |

---

## Acknowledgement

Data collected at APT Sweater Limited (APT Garments), Gazipur. Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), Roboflow, and Streamlit.
