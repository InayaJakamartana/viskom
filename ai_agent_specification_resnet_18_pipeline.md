# AI Coding Agent Specification: ResNet-18 Image Classification Pipeline

## 1. Project Context & Objective
You are an expert Machine Learning Engineer tasked with implementing a complete, well-documented, and reproducible **ResNet-18** object classification pipeline in Python (PyTorch). This code is part of an academic benchmark study (**CAK4WBB3 Computer Vision - Tugas 1**) comparing multiple CNN architectures on a multi-class dataset.

To achieve the highest rubric evaluation (**>85 / "Sangat Mampu"**), the code must be modular, free of data leakage, thoroughly commented, and must track **both classification performance and computational efficiency metrics** so ResNet-18 can be directly compared with the team's other two models (e.g., MobileNetV2, EfficientNet-B0).

---

## 2. Dataset Specification
* **Source:** [Kaggle Garbage Classification Dataset (`asdasdasasdas/garbage-classification`)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
* **Total Images:** 2,527 images (Satisfies the assignment minimum of $\ge 1,000$ images).
* **Classes (6 Classes):** `cardboard` (403), `glass` (501), `metal` (410), `paper` (594), `plastic` (482), `trash` (137) (Satisfies the assignment minimum of $\ge 5$ classes).
* **Directory Handling:** The dataset contains a folder `Garbage classification/Garbage classification/` with 6 class subfolders. Write a dataset ingestion utility (using `kagglehub` or `opendatasets`, or local path resolution) that locates the folder containing the 6 class subdirectories and splits the dataset cleanly.

---

## 3. Strict Experimental & Preprocessing Standards (20% Rubric Weight)
To ensure fair comparison across different CNN architectures and avoid data leakage, implement the following pipeline strictly:

1. **Reproducibility:**
   * Set global random seeds (`SEED = 42`) for `random`, `numpy`, `torch`, and `torch.cuda`.
2. **Stratified Dataset Splitting:**
   * Split the 2,527 images using a **Stratified Split** (via `sklearn.model_selection.train_test_split`) to preserve class ratios (crucial because `trash` only has 137 images):
     * **Training Set:** 70%
     * **Validation Set:** 15%
     * **Test Set:** 15%
   * **CRITICAL:** Create custom Dataset wrappers or subset wrappers so that **data augmentation is ONLY applied to the Training Set**, while Validation and Test sets receive only deterministic resizing and normalization.
3. **Transforms (`torchvision.transforms`):**
   * **Input Resolution:** $224 \times 224$ pixels.
   * **Normalization:** ImageNet mean $[0.485, 0.456, 0.406]$ and standard deviation $[0.229, 0.224, 0.225]$.
   * **Training Augmentation:**
     * `RandomResizedCrop(224, scale=(0.8, 1.0))` or `Resize((224, 224))`
     * `RandomHorizontalFlip(p=0.5)`
     * `RandomRotation(degrees=15)`
     * `ColorJitter(brightness=0.1, contrast=0.1)`
     * `ToTensor()` and `Normalize(...)`
   * **Validation / Test Preprocessing:**
     * `Resize((224, 224))`
     * `ToTensor()` and `Normalize(...)`
4. **DataLoaders:**
   * Use `batch_size = 32`, `num_workers = 2` (or OS-appropriate), `pin_memory = True`.

---

## 4. Model Architecture & Training Specification (30% Rubric Weight)
1. **Model Initialization:**
   * Load pretrained **ResNet-18** via `torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)`.
   * Replace the final fully connected layer (`model.fc`) with `nn.Linear(model.fc.in_features, 6)`.
   * Unfreeze all layers for full fine-tuning (or support an option for head-only warmup followed by full fine-tuning).
2. **Hyperparameters (Standardized for Multi-Model Comparison):**
   * **Loss Function:** `nn.CrossEntropyLoss()`
   * **Optimizer:** `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)`
   * **Learning Rate Scheduler:** `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)`
   * **Epochs:** `20` epochs (with Best Model Checkpointing based on highest Validation Accuracy / lowest Validation Loss).

---

## 5. Required Evaluation & Efficiency Metrics (30% Rubric Weight)
The assignment requires analyzing both **classification performance** and **computational complexity/efficiency**. Your code must automatically compute, print, and export a summary dictionary/CSV containing:

### A. Classification Metrics (Evaluated on the held-out Test Set using the best checkpoint)
1. **Overall Test Accuracy** (%)
2. **Macro-Averaged & Weighted-Averaged Precision, Recall, and F1-Score** (via `sklearn.metrics.precision_recall_fscore_support`)
3. **Full Per-Class Classification Report** (`sklearn.metrics.classification_report` across all 6 garbage classes)

### B. Computational Complexity & Efficiency Metrics
1. **Total Parameters & Trainable Parameters:** Counted via `sum(p.numel() for p in model.parameters())`.
2. **Model Disk Size (MB):** Size of the saved `.pth` state dict file in megabytes.
3. **Total Training Time (seconds) & Average Time per Epoch (seconds):** Measured using `time.perf_counter()` (using `torch.cuda.synchronize()` if on GPU).
4. **Inference Latency (milliseconds per image) & Throughput (images/sec):** Measure the exact forward-pass time over the entire Test Set and calculate:
   $$\text{Latency (ms/image)} = \frac{\text{Total Test Inference Time (s)} \times 1000}{\text{Total Test Images}}$$

---

## 6. Required Visualizations (Saved as High-DPI PNGs for the 3-Page Report)
Generate and save all plots at `dpi=300` with clean titles, axis labels, and legends:

1. **`resnet18_training_curves.png`:**
   * Side-by-side subplot (`1x2`):
     * Left: Training vs. Validation Loss across epochs.
     * Right: Training vs. Validation Accuracy across epochs.
2. **`resnet18_confusion_matrix.png`:**
   * Annotated heatmap (`seaborn.heatmap`) of the $6 \times 6$ Confusion Matrix on the Test Set, showing both raw counts (or normalized percentages) with exact class names on the axes.
3. **`resnet18_sample_predictions.png`:**
   * A $3 \times 4$ grid (12 test images) showing the de-normalized image, **True Label**, **Predicted Label**, and **Prediction Confidence (%)**. Color-code the title green for correct predictions and red for misclassifications.

---

## 7. Code Structure & Documentation Standards (10% Code + 10% Report Rubric Weight)
* **Format:** Deliver a clean, self-contained Jupyter Notebook (`.ipynb`) or modular Python script (`.py`) that runs end-to-end on Google Colab (T4 GPU) or local CUDA environment without manual intervention.
* **Documentation:** Include clear Markdown cells / docstrings explaining:
  1. Why each preprocessing step was chosen (e.g., preventing overfitting on the 2,527-image dataset, handling class imbalance in `trash`).
  2. How ResNet-18's architecture works (specifically explaining basic $3 \times 3$ residual blocks and identity skip connections $y = \mathcal{F}(x, \{W_i\}) + x$).
  3. Automated Markdown/text output at the end of the script summarizing the exact metrics in a formatted table ready to be pasted into the 3-page PDF report.