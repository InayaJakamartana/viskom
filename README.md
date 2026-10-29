# 🗑️ MobileNetV2 for Multiclass Garbage Image Classification
### Course: CAK4WBB3 – Computer Vision | Assignment: Task 1 – Observation of Various CNN Architectures

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14+cpu-EE4C2C.svg)](https://pytorch.org/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-92.76%25-brightgreen.svg)]()
[![Macro F1](https://img.shields.io/badge/Macro%20F1-92.75%25-brightgreen.svg)]()
[![Model Size](https://img.shields.io/badge/Model%20Size-8.74%20MB-blue.svg)]()
[![CPU Latency](https://img.shields.io/badge/CPU%20Latency-15.47%20ms-orange.svg)]()

---

## 📌 Executive Summary

This repository contains the end-to-end implementation, training pipeline, empirical evaluation, and academic report for **Task 1: Observation of Various CNN Architectures in Object Classification** for the **CAK4WBB3 Computer Vision** course.

The project focuses on **MobileNetV2**, evaluating its lightweight depthwise separable convolutions, inverted residual blocks, and linear bottlenecks on a real-world multiclass garbage classification task (5 categories: *cardboard, glass, metal, paper, plastic*). The experiment was conducted in an resource-constrained environment (**CPU execution**) using transfer learning from ImageNet-1K pretrained weights followed by selective fine-tuning of upper backbone layers.

---

## 📊 Key Experimental Results

| Metric | Measured Value | Analysis / Significance |
|:---|:---:|:---|
| **Test Accuracy** | **92.76%** | Evaluated once on 359 held-out test images |
| **Macro F1-Score** | **92.75%** | Balanced across all 5 classes (robust against frequency variations) |
| **Macro Precision** | **92.62%** | High true-positive fidelity across categories |
| **Macro Recall** | **93.11%** | Minimal false-negative omissions |
| **Phase 1 Best Val Acc** | **85.79%** | Frozen backbone baseline (head only: 6,405 parameters) |
| **Phase 2 Best Val Acc** | **94.15%** | Fine-tuned `features[14:]` (+8.36% improvement over Phase 1) |
| **Total Parameters** | **2,230,277** | ~2.23 M (vs. 138 M for VGG-16, a ~62× reduction) |
| **Model Disk Footprint** | **8.74 MB** | Highly compact, directly deployable on edge/IoT microcontrollers |
| **CPU Inference Latency** | **15.47 ms ± 1.66 ms** | Single-image latency on CPU (averaged over 50 runs) |
| **Inference Throughput** | **47.80 img/sec** | Batch throughput (batch size 32) on general CPU |
| **Total Training Duration** | **22.5 minutes** | Phase 1 (9.8 min) + Phase 2 (12.7 min) on CPU |

### Per-Class Test Performance Breakdown

| Class | Support | Precision | Recall | F1-Score | Key Phenomenological Observation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Cardboard** | 61 | 95.24% | 98.36% | **96.77%** | Strong boundary definition; minor confusion with pressed paper |
| **Glass** | 75 | 90.91% | 93.33% | **92.11%** | Specular highlights; primary confusion with clear plastic containers |
| **Metal** | 62 | 88.41% | 98.39% | **93.13%** | Rigid cylindrical cans and metallic sheen transfer well from ImageNet |
| **Paper** | 89 | 97.62% | 92.13% | **94.80%** | Highest precision (97.6%); textured flat sheets distinguish well |
| **Plastic** | 72 | 90.91% | 83.33% | **86.96%** | High intra-class geometric variation (crumpled bags, bottles) |

---

## 🗂️ Dataset Details & Partitioning

- **Source:** Local Garbage Classification dataset (`Garbage classification/Garbage classification/`).
- **Pruning Decision (Option B):** The sixth raw directory (`trash`, 137 images) was pruned to eliminate severe class imbalance (>4.3×) and maintain a balanced, benchmark-aligned 5-class distribution.
- **Verification:** 100% of images were verified via Pillow (`img.verify()`); zero corrupted or unreadable files.
- **Splits (Stratified, Seed = 42):**
  - **Training (70%):** 1,672 images
  - **Validation (15%):** 359 images
  - **Testing (15%):** 359 images (held out for final evaluation)
  - **Assertion:** Strict set intersection check confirmed **zero data leakage** across splits.

---

## 🏗️ Architectural Mechanism of MobileNetV2

MobileNetV2 introduces two primary mechanisms that govern its exceptional parameter and computational efficiency:

1. **Depthwise Separable Convolutions:**
   Decomposes standard convolution into:
   - *Depthwise convolution (3×3):* Spatial filtering per input channel independently.
   - *Pointwise convolution (1×1):* Linear combination across channels.
   - *FLOP Reduction:* Reduces computational cost by $\approx \frac{1}{9} + \frac{1}{C_{out}}$ (~8–9× fewer operations than standard 3×3 conv).

2. **Inverted Residuals with Linear Bottlenecks:**
   - Standard ResNet blocks follow a wide $\to$ narrow $\to$ wide structure.
   - MobileNetV2 inverts this to narrow $\to$ wide $\to$ narrow (expansion factor $t = 6$).
   - A 1×1 convolution expands channels into higher-dimensional space where non-linear ReLU6 is applied without information loss.
   - The final 1×1 projection uses a **purely linear activation** to preserve manifold topology and prevent feature collapse in low dimensions.
   - Shortcut residual connections are added between bottlenecks of identical spatial dimensions and stride = 1.

---

## 🚀 Two-Phase Transfer Learning Strategy

1. **Phase 1: Feature Extraction (Frozen Backbone):**
   - Backbone `features[0:18]` is completely locked (`requires_grad = False`).
   - Classifier head replaced with: `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, 5))`.
   - Trained for up to 15 epochs with Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
   - Reached **85.79% validation accuracy** with only **6,405 trainable parameters**.

2. **Phase 2: Domain-Specific Fine-Tuning:**
   - Unfroze top 4 inverted residual stages (`features[14:]`) and classifier head (**1,687,749 trainable parameters**).
   - Trained with a conservative learning rate ($\text{lr} = 10^{-4}$) and `ReduceLROnPlateau` scheduler (factor $= 0.5$, patience $= 3$).
   - Validation accuracy improved to **94.15% (+8.36%)**, proving that fine-tuning upper layers successfully adapted generic ImageNet features to garbage surface textures.

---

## 📁 Repository Structure & Deliverables

```
viskom/
│
├── mobilenetv2_garbage_classification.ipynb    # Primary deliverable: Full executed Jupyter Notebook
├── MobileNetV2_Garbage_Classification_Report.pdf # Final deliverable: 4-page academic report (1 cover + 3 content)
├── README.md                                   # Comprehensive project documentation
│
├── outputs/                                    # Generated experimental artifacts
│   ├── ckpt_phase1_best.pth                    # Best Phase 1 checkpoint (val acc: 85.79%)
│   ├── ckpt_phase2_best.pth                    # Best Phase 2 checkpoint (val acc: 94.15%)
│   ├── mobilenetv2_class_distribution.png      # Class counts bar & donut distribution chart
│   ├── mobilenetv2_sample_images.png           # Sample images (one per category)
│   ├── mobilenetv2_training_curves.png         # Phase 1 + 2 loss & accuracy curves
│   ├── mobilenetv2_confusion_matrix.png        # Raw & normalized test confusion matrices
│   ├── mobilenetv2_per_class_metrics.png       # Per-class precision, recall, and F1 bar plot
│   ├── mobilenetv2_predictions.png             # Correct vs. incorrect prediction gallery
│   ├── mobilenetv2_efficiency_summary.png      # Parameters donut, latency histogram & summary
│   ├── mobilenetv2_test_metrics.csv            # Comprehensive test metrics in CSV format
│   ├── mobilenetv2_classification_report.txt   # Scikit-learn test classification report
│   └── mobilenetv2_model/
│       └── mobilenetv2_best.pth                # Final packaged PyTorch model dictionary
│
└── Garbage classification/                     # Local dataset root
    └── Garbage classification/
        ├── cardboard/                          # 403 images
        ├── glass/                              # 501 images
        ├── metal/                              # 410 images
        ├── paper/                              # 594 images
        ├── plastic/                            # 482 images
        └── trash/                              # 137 images (pruned for balance)
```

---

## ⚙️ How to Reproduce

### 1. Requirements
Ensure Python 3.10+ and standard machine learning packages are installed:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow reportlab
```

### 2. Inspect or Run the Notebook
Open [`mobilenetv2_garbage_classification.ipynb`](mobilenetv2_garbage_classification.ipynb) in Jupyter Notebook, JupyterLab, or VS Code. All 45 cells already contain full execution outputs, markdown documentation, and inline high-resolution figures.

### 3. Re-generate the Academic PDF Report
To re-compile the formal 4-page academic PDF report from the saved metrics:
```bash
python generate_report_pdf.py
```
The resulting PDF [`MobileNetV2_Garbage_Classification_Report.pdf`](MobileNetV2_Garbage_Classification_Report.pdf) strictly conforms to the maximum 3 content pages requirement (excluding cover).

---

## 📜 Academic Report Summary

The included academic report [`MobileNetV2_Garbage_Classification_Report.pdf`](MobileNetV2_Garbage_Classification_Report.pdf) contains:
- **Cover Page:** Formal Telkom University header, Course details (CAK4WBB3), assignment metadata, executive abstract, and experimental specification table.
- **Page 1 (Content):** Introduction, problem formulation, dataset characteristics & distribution figures, and MobileNetV2 architecture deep-dive (inverted residuals and linear bottlenecks).
- **Page 2 (Content):** Two-phase transfer learning methodology, training convergence curves, empirical test metrics table, and normalized confusion matrix analysis.
- **Page 3 (Content):** Computational efficiency profiling (latency, throughput, model footprint), prediction error diagnostics, comparative discussion against ResNet-18 and EfficientNet-B0, conclusion, and academic references.
