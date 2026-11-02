# Garbage classification: EfficientNet-B0, MobileNetV2 and ResNet-18


# ==================== EfficientNet-B0 ====================

# BLOCK 1: IMPORTS & GLOBAL CONFIGURATION
import os
import sys
import time
import copy
import zipfile
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# Configure visual style for plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

# Optimize CPU multi-threading for PyTorch execution
NUM_CPU_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CPU_CORES)

# Set deterministic seeds for full reproducibility across runs
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hardware device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory & Hyperparameter Configuration
PROJECT_DIR = Path.cwd()
WORKSPACE_DIR = PROJECT_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Image and Training Hyperparameters for EfficientNet-B0
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 6
NUM_EPOCHS = 20
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4

# Standard ImageNet Normalization Statistics (RGB channels)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"PyTorch Version       : {torch.__version__}")
print(f"Torchvision Version   : {torchvision.__version__}")
print(f"Computation Device    : {DEVICE} (CPU Threads: {torch.get_num_threads()})")
print(f"Project Directory     : {PROJECT_DIR}")
print(f"Workspace Directory   : {WORKSPACE_DIR}")

# BLOCK 2.1: KAGGLE API DATASET DOWNLOAD & DIRECTORY VERIFICATION
def download_or_locate_dataset(workspace_dir: Path) -> Path:
    """
    Locates the Garbage Classification dataset locally or downloads it via the
    official Kaggle API command:
        `kaggle datasets download -d asdasdasasdas/garbage-classification`
    """
    candidate_paths = [
        workspace_dir / "Garbage classification" / "Garbage classification",
        workspace_dir / "Garbage classification",
        PROJECT_DIR / "Garbage classification" / "Garbage classification",
    ]

    expected_classes = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}

    for path in candidate_paths:
        if path.exists() and path.is_dir():
            subdirs = {d.name for d in path.iterdir() if d.is_dir()}
            if expected_classes.issubset(subdirs):
                print(f"[INFO] Verified local dataset directory at: {path}")
                return path

    # If dataset is not found locally, download via Kaggle CLI
    print("[INFO] Dataset not found locally. Downloading via Kaggle API...")
    download_cmd = [
        "kaggle", "datasets", "download",
        "-d", "asdasdasasdas/garbage-classification",
        "-p", str(workspace_dir)
    ]
    print(f"[CMD] Running: {' '.join(download_cmd)}")
    subprocess.run(download_cmd, check=True)

    zip_path = workspace_dir / "garbage-classification.zip"
    if zip_path.exists():
        print(f"[INFO] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(workspace_dir)

    for path in candidate_paths:
        if path.exists() and expected_classes.issubset({d.name for d in path.iterdir() if d.is_dir()}):
            return path

    raise FileNotFoundError("Could not locate the 6 garbage classification folders after download.")


DATASET_DIR = download_or_locate_dataset(WORKSPACE_DIR)

# Inspect class folders and image counts
base_dataset = datasets.ImageFolder(root=str(DATASET_DIR))
CLASS_NAMES = base_dataset.classes
CLASS_TO_IDX = base_dataset.class_to_idx
TOTAL_IMAGES = len(base_dataset)

print(f"Total Images Found    : {TOTAL_IMAGES} (Requirement: >1000 images)")
print(f"Number of Classes     : {len(CLASS_NAMES)} (Requirement: >5 classes)")
print(f"Class Mapping         : {CLASS_TO_IDX}")

# BLOCK 2.2: TRANSFORMS, STRATIFIED SPLIT & PYTORCH DATALOADERS

# 1. Define Training Transforms (Resizing + Data Augmentation + ImageNet Normalization)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 2. Define Validation & Test Transforms (Deterministic Resizing + ImageNet Normalization ONLY)
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 3. Perform Stratified 70% Train / 15% Validation / 15% Test Split
all_indices = np.arange(TOTAL_IMAGES)
all_targets = np.array(base_dataset.targets)

# First split: 70% Train, 30% Temp (Validation + Test)
train_idx, temp_idx, y_train, y_temp = train_test_split(
    all_indices,
    all_targets,
    test_size=0.30,
    stratify=all_targets,
    random_state=SEED,
)

# Second split: Divide the 30% Temp equally into 15% Validation and 15% Test
val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED,
)

# Create separate ImageFolder instances so Train and Val/Test have independent transforms
train_full_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=train_transforms)
eval_full_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=eval_transforms)

train_dataset = Subset(train_full_dataset, train_idx)
val_dataset = Subset(eval_full_dataset, val_idx)
test_dataset = Subset(eval_full_dataset, test_idx)

# 4. Create PyTorch DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Build a summary table of class distribution across splits
split_records = []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    tr_cnt = int(np.sum(y_train == cls_idx))
    va_cnt = int(np.sum(y_val == cls_idx))
    te_cnt = int(np.sum(y_test == cls_idx))
    tot_cnt = tr_cnt + va_cnt + te_cnt
    split_records.append({
        "Class ID": cls_idx,
        "Class Name": cls_name,
        "Train (70%)": tr_cnt,
        "Validation (15%)": va_cnt,
        "Test (15%)": te_cnt,
        "Total": tot_cnt,
        "Share (%)": round(100.0 * tot_cnt / TOTAL_IMAGES, 2),
    })

split_df = pd.DataFrame(split_records)
print("\n=== Stratified Dataset Split Summary ===")
print(split_df.to_string(index=False))
print(f"\nTotal Split Sizes -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# BLOCK 2.3: VISUALIZE CLASS DISTRIBUTION & AUGMENTED TRAINING SAMPLES
def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization on a (C, H, W) tensor for Matplotlib visualization."""
    img = tensor_img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0.0, 1.0)

# 1. Plot Stratified Class Distribution
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(CLASS_NAMES))
width = 0.26

bars1 = ax.bar(x - width, split_df["Train (70%)"], width, label="Train (70%)", color="#2b5c8f")
bars2 = ax.bar(x, split_df["Validation (15%)"], width, label="Validation (15%)", color="#38a3a5")
bars3 = ax.bar(x + width, split_df["Test (15%)"], width, label="Test (15%)", color="#f4a261")

ax.set_title("Stratified Class Distribution Across Train, Validation, and Test Splits", fontsize=13, fontweight="bold")
ax.set_xlabel("Garbage Category", fontsize=11)
ax.set_ylabel("Number of Images", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=10)
ax.legend(frameon=True)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "01_class_distribution.png", bbox_inches="tight")
plt.show()

# 2. Visualize Sample Augmented Training Images (2 samples per class = 12 images)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
fig.suptitle("Sample Preprocessed & Augmented Training Images (224x224, Flip/Rotation/ColorJitter)", fontsize=13, fontweight="bold")

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_positions = np.where(y_train == cls_idx)[0][:2]
    for row_idx, pos in enumerate(cls_positions):
        img_tensor, label = train_dataset[pos]
        ax = axes[row_idx, cls_idx]
        ax.imshow(denormalize_image(img_tensor))
        ax.set_title(f"Class: {cls_name}", fontsize=10, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "02_sample_augmented_images.png", bbox_inches="tight")
plt.show()

# BLOCK 3: EFFICIENTNET-B0 MODEL BUILDER & COMPLEXITY ANALYSIS
def build_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
    freeze_early_layers: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Loads pre-trained EfficientNet-B0 on ImageNet-1K, freezes early feature extraction
    stages (features[0..6]), unfreezes the top semantic blocks (features[7..8]) for
    domain adaptation, and replaces the 1000-class head with a 6-class classifier.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    if freeze_early_layers:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the top convolutional stages (features[-2:]) for domain adaptation
        for param in model.features[-2:].parameters():
            param.requires_grad = True

    # Modify final classification head: 1280 input features -> 6 output garbage classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model.to(DEVICE)


# Instantiate EfficientNet-B0 and inspect stage-by-stage parameter breakdown
efficientnet_model = build_efficientnet_b0(num_classes=NUM_CLASSES, freeze_early_layers=True)

stage_rows = []
for stage_idx, stage_module in enumerate(efficientnet_model.features):
    tot_p = sum(p.numel() for p in stage_module.parameters())
    trn_p = sum(p.numel() for p in stage_module.parameters() if p.requires_grad)
    stage_rows.append({
        "Component": f"features[{stage_idx}] ({type(stage_module).__name__})",
        "Total Params": f"{tot_p:,}",
        "Trainable Params": f"{trn_p:,}",
        "Status": "Trainable (Fine-Tuned)" if trn_p > 0 else "Frozen (Pre-Trained)",
    })

cls_tot = sum(p.numel() for p in efficientnet_model.classifier.parameters())
cls_trn = sum(p.numel() for p in efficientnet_model.classifier.parameters() if p.requires_grad)
stage_rows.append({
    "Component": "classifier (Dropout + Linear 1280->6)",
    "Total Params": f"{cls_tot:,}",
    "Trainable Params": f"{cls_trn:,}",
    "Status": "Trainable (New Head)",
})

stage_df = pd.DataFrame(stage_rows)
total_params = sum(p.numel() for p in efficientnet_model.parameters())
trainable_params = sum(p.numel() for p in efficientnet_model.parameters() if p.requires_grad)
model_size_mb = sum(p.numel() * p.element_size() for p in efficientnet_model.parameters()) / (1024 ** 2)

print("=== EfficientNet-B0 Modified Classification Head ===")
print(efficientnet_model.classifier)
print("\n=== EfficientNet-B0 Stage-by-Stage Parameter Breakdown ===")
print(stage_df.to_string(index=False))
print(
    f"\nTotal Parameters     : {total_params:,}\n"
    f"Trainable Parameters : {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)\n"
    f"Frozen Parameters    : {total_params - trainable_params:,}\n"
    f"Model Memory Size    : {model_size_mb:.2f} MB"
)

# BLOCK 4: EFFICIENTNET-B0 TRAINING LOOP WITH TRAINING & INFERENCE TIME TRACKING
def train_efficientnet_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> tuple[nn.Module, dict]:
    """
    Trains and validates EfficientNet-B0 while recording loss, accuracy,
    per-epoch training duration, and per-batch training/inference latency.
    Saves and restores the best checkpoint based on highest Validation Accuracy.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_train_time_s": [],
        "batch_train_time_ms": [],
        "batch_val_time_ms": [],
        "learning_rate": [],
    }

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    total_start_time = time.perf_counter()

    print("=" * 92)
    print(f"TRAINING EFFICIENTNET-B0 ({num_epochs} Epochs | Batch Size: {BATCH_SIZE} | Device: {DEVICE})")
    print("=" * 92)

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        # 1. TRAINING PHASE
        model.train()
        running_train_loss = 0.0
        running_train_corrects = 0
        total_train_samples = 0
        batch_train_times = []

        epoch_train_start = time.perf_counter()
        for inputs, labels in train_loader:
            b_start = time.perf_counter()
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            b_end = time.perf_counter()

            batch_train_times.append((b_end - b_start) * 1000.0)
            _, preds = torch.max(outputs, 1)
            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data).item()
            total_train_samples += inputs.size(0)

        epoch_train_time = time.perf_counter() - epoch_train_start
        scheduler.step()

        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = running_train_corrects / total_train_samples
        mean_batch_train_ms = float(np.mean(batch_train_times))

        # 2. VALIDATION PHASE
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val_samples = 0
        batch_val_times = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                b_start = time.perf_counter()
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                b_end = time.perf_counter()

                batch_val_times.append((b_end - b_start) * 1000.0)
                _, preds = torch.max(outputs, 1)
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data).item()
                total_val_samples += inputs.size(0)

        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = running_val_corrects / total_val_samples
        mean_batch_val_ms = float(np.mean(batch_val_times))

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["epoch_train_time_s"].append(epoch_train_time)
        history["batch_train_time_ms"].append(mean_batch_train_ms)
        history["batch_val_time_ms"].append(mean_batch_val_ms)
        history["learning_rate"].append(current_lr)

        # Save best checkpoint
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, MODELS_DIR / "efficientnet_b0_best.pth")

        marker = "*BEST*" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:6.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:6.2f}% | "
            f"Train Time: {epoch_train_time:5.2f}s ({mean_batch_train_ms:6.1f} ms/batch) | "
            f"Val Latency: {mean_batch_val_ms:5.1f} ms/batch {marker}"
        )

    total_training_time = time.perf_counter() - total_start_time
    history["total_training_time_s"] = total_training_time
    history["best_val_acc"] = best_val_acc

    print("=" * 92)
    print(f"Finished Training EfficientNet-B0 in {total_training_time:.2f}s | Best Val Accuracy: {best_val_acc*100:.2f}%")
    print("=" * 92)

    # Restore best model weights
    model.load_state_dict(best_model_wts)
    return model, history


efficientnet_model, training_history = train_efficientnet_model(
    model=efficientnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
)

# Display epoch-by-epoch training history table
history_df = pd.DataFrame({
    "Epoch": training_history["epoch"],
    "Train Loss": [round(x, 4) for x in training_history["train_loss"]],
    "Train Acc (%)": [round(x * 100, 2) for x in training_history["train_acc"]],
    "Val Loss": [round(x, 4) for x in training_history["val_loss"]],
    "Val Acc (%)": [round(x * 100, 2) for x in training_history["val_acc"]],
    "Epoch Time (s)": [round(x, 2) for x in training_history["epoch_train_time_s"]],
    "Train Batch (ms)": [round(x, 1) for x in training_history["batch_train_time_ms"]],
    "Val Batch (ms)": [round(x, 1) for x in training_history["batch_val_time_ms"]],
})
history_df.to_csv(OUTPUTS_DIR / "efficientnet_b0_training_history.csv", index=False)
print("\n=== EfficientNet-B0 Epoch-by-Epoch Training & Efficiency Log ===")
print(history_df.to_string(index=False))

# BLOCK 5: TEST SET EVALUATION (ACCURACY, PRECISION, RECALL, F1, CONFUSION MATRIX)
def evaluate_efficientnet_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
) -> dict:
    """
    Evaluates the fine-tuned EfficientNet-B0 model on the unseen test set, tracking
    inference time per batch and per image, and computing Accuracy, Precision,
    Recall, F1-Score, and the Confusion Matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    batch_infer_times_ms = []

    total_infer_start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            b_start = time.perf_counter()
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)
            b_end = time.perf_counter()

            batch_infer_times_ms.append((b_end - b_start) * 1000.0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    total_infer_time_s = time.perf_counter() - total_infer_start
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Compute Mandatory Metrics via sklearn.metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    mean_batch_infer_ms = float(np.mean(batch_infer_times_ms))
    mean_image_infer_ms = (total_infer_time_s * 1000.0) / len(y_true)
    throughput_img_per_sec = len(y_true) / total_infer_time_s

    print("=" * 84)
    print("TEST SET CLASSIFICATION REPORT: EfficientNet-B0")
    print("=" * 84)
    print(cls_report)
    print("=" * 84)
    print("SUMMARY OF MANDATORY EVALUATION & EFFICIENCY METRICS")
    print("=" * 84)
    print(f"Test Set Accuracy               : {acc * 100:.2f}%")
    print(f"Macro Precision / Recall / F1   : {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted Precision / Recall / F1: {prec_weighted:.4f} / {rec_weighted:.4f} / {f1_weighted:.4f}")
    print(f"Total Training Time             : {training_history['total_training_time_s']:.2f} seconds")
    print(f"Mean Training Time per Epoch    : {np.mean(training_history['epoch_train_time_s']):.2f} seconds")
    print(f"Mean Training Time per Batch    : {np.mean(training_history['batch_train_time_ms']):.2f} ms/batch")
    print(f"Mean Test Inference per Batch   : {mean_batch_infer_ms:.2f} ms/batch")
    print(f"Mean Test Inference per Image   : {mean_image_infer_ms:.2f} ms/image ({throughput_img_per_sec:.1f} images/sec)")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "prec_per_class": prec_per_class,
        "rec_per_class": rec_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "mean_batch_infer_ms": mean_batch_infer_ms,
        "mean_image_infer_ms": mean_image_infer_ms,
        "throughput_img_per_sec": throughput_img_per_sec,
        "total_infer_time_s": total_infer_time_s,
    }


test_results = evaluate_efficientnet_on_test_set(efficientnet_model, test_loader)

# Save per-class metrics table to CSV
per_class_df = pd.DataFrame({
    "Class Name": CLASS_NAMES,
    "Test Samples": [int(np.sum(test_results["y_true"] == i)) for i in range(NUM_CLASSES)],
    "Precision": [round(float(x), 4) for x in test_results["prec_per_class"]],
    "Recall": [round(float(x), 4) for x in test_results["rec_per_class"]],
    "F1-Score": [round(float(x), 4) for x in test_results["f1_per_class"]],
})
per_class_df.to_csv(OUTPUTS_DIR / "efficientnet_b0_test_metrics.csv", index=False)

# BLOCK 6.1: TRAINING & VALIDATION ACCURACY, LOSS, AND TIMING CURVES
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("EfficientNet-B0 Training & Validation Learning Curves and Efficiency Profile", fontsize=14, fontweight="bold")

epochs_range = training_history["epoch"]

# 1. Accuracy Curve
ax1 = axes[0]
ax1.plot(epochs_range, [a * 100 for a in training_history["train_acc"]], marker="o", linewidth=2.2, color="#1f77b4", label="Train Accuracy")
ax1.plot(epochs_range, [a * 100 for a in training_history["val_acc"]], marker="s", linewidth=2.2, color="#2ca02c", label="Validation Accuracy")
ax1.set_title("Accuracy vs. Epochs", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.set_xticks(epochs_range)
ax1.legend(loc="lower right")

# 2. Loss Curve
ax2 = axes[1]
ax2.plot(epochs_range, training_history["train_loss"], marker="o", linewidth=2.2, color="#d62728", label="Train Loss")
ax2.plot(epochs_range, training_history["val_loss"], marker="s", linewidth=2.2, color="#ff7f0e", label="Validation Loss")
ax2.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Cross-Entropy Loss")
ax2.set_xticks(epochs_range)
ax2.legend(loc="upper right")

# 3. Batch Training & Validation Inference Time per Epoch
ax3 = axes[2]
ax3.plot(epochs_range, training_history["batch_train_time_ms"], marker="^", linewidth=2.2, color="#9467bd", label="Train Time (ms/batch)")
ax3.plot(epochs_range, training_history["batch_val_time_ms"], marker="d", linewidth=2.2, color="#17becf", label="Val Inference (ms/batch)")
ax3.set_title("Batch Processing Latency per Epoch", fontsize=12, fontweight="bold")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Time per Batch (ms)")
ax3.set_xticks(epochs_range)
ax3.legend(loc="center right")

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "03_efficientnet_learning_curves.png", bbox_inches="tight")
plt.show()

# BLOCK 6.2: SEABORN CONFUSION MATRIX & PER-CLASS METRICS VISUALIZATION
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6))
fig.suptitle(
    f"EfficientNet-B0 Test Set Evaluation (Accuracy: {test_results['accuracy']*100:.2f}% | "
    f"Macro F1: {test_results['f1_macro']:.4f})",
    fontsize=14,
    fontweight="bold",
)

# 1. Confusion Matrix (Counts + Row-Normalized %)
cm = test_results["confusion_matrix"]
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100.0

annot_matrix = np.empty_like(cm, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.0f}%)"

ax1 = axes[0]
sns.heatmap(
    cm,
    annot=annot_matrix,
    fmt="",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cbar=True,
    ax=ax1,
    linewidths=0.6,
    linecolor="lightgray",
)
ax1.set_title("Confusion Matrix (Counts & Row %)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Predicted Garbage Category", fontsize=11, fontweight="bold")
ax1.set_ylabel("True Garbage Category", fontsize=11, fontweight="bold")
ax1.tick_params(axis="x", rotation=25)
ax1.tick_params(axis="y", rotation=0)

# 2. Per-Class Precision, Recall, and F1-Score Bar Chart
ax2 = axes[1]
x = np.arange(NUM_CLASSES)
width = 0.26

b1 = ax2.bar(x - width, test_results["prec_per_class"] * 100, width, label="Precision (%)", color="#2b5c8f")
b2 = ax2.bar(x, test_results["rec_per_class"] * 100, width, label="Recall (%)", color="#38a3a5")
b3 = ax2.bar(x + width, test_results["f1_per_class"] * 100, width, label="F1-Score (%)", color="#f4a261")

ax2.set_title("Per-Class Precision, Recall, and F1-Score on Test Set", fontsize=12, fontweight="bold")
ax2.set_xlabel("Garbage Category", fontsize=11, fontweight="bold")
ax2.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax2.set_ylim(50, 108)
ax2.set_xticks(x)
ax2.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
ax2.legend(loc="lower right")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "04_efficientnet_confusion_matrix_and_metrics.png", bbox_inches="tight")
plt.show()

# BLOCK 6.3: SAMPLE TEST IMAGES WITH TRUE LABELS & EFFICIENTNET-B0 PREDICTIONS
y_true_test = test_results["y_true"]
y_pred_test = test_results["y_pred"]
y_prob_test = test_results["y_prob"]

# Select 3 test samples per class (18 test images total) for a balanced visual grid
selected_test_positions = []
for cls_idx in range(NUM_CLASSES):
    cls_positions = np.where(y_true_test == cls_idx)[0][:3]
    selected_test_positions.extend(cls_positions)

fig, axes = plt.subplots(3, 6, figsize=(16, 8.5))
fig.suptitle(
    f"Sample Test Set Predictions Using EfficientNet-B0 "
    f"(Test Accuracy: {test_results['accuracy']*100:.2f}%)",
    fontsize=14,
    fontweight="bold",
)

for col_idx in range(NUM_CLASSES):
    for row_idx in range(3):
        pos = selected_test_positions[col_idx * 3 + row_idx]
        img_tensor, true_idx = test_dataset[pos]
        pred_idx = int(y_pred_test[pos])
        confidence = float(y_prob_test[pos, pred_idx]) * 100.0

        true_name = CLASS_NAMES[true_idx]
        pred_name = CLASS_NAMES[pred_idx]
        is_correct = (true_idx == pred_idx)

        ax = axes[row_idx, col_idx]
        ax.imshow(denormalize_image(img_tensor))
        title_color = "#1b7837" if is_correct else "#d73027"
        status_tag = "OK" if is_correct else "ERR"
        ax.set_title(
            f"[{status_tag}] True: {true_name}\nPred: {pred_name} ({confidence:.1f}%)",
            fontsize=9,
            color=title_color,
            fontweight="bold",
        )
        ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "05_efficientnet_test_predictions_grid.png", bbox_inches="tight")
plt.show()


# ==================== MobileNetV2 ====================

import subprocess, sys, importlib

def ensure(package, import_as=None, pip_name=None):
    name = import_as or package
    pip  = pip_name  or package
    try:
        m   = importlib.import_module(name)
        ver = getattr(m, '__version__', 'installed')
        print(f"  ✅  {package:<22} {ver}")
    except ImportError:
        print(f"  📦  Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip])
        m   = importlib.import_module(name)
        ver = getattr(m, '__version__', 'installed')
        print(f"  ✅  {package:<22} {ver}  ← just installed")

print("Checking dependencies ...")
ensure("torch")
ensure("torchvision")
ensure("numpy")
ensure("pandas")
ensure("matplotlib")
ensure("seaborn")
ensure("scikit-learn", "sklearn")
ensure("Pillow", "PIL")
ensure("opencv-python", "cv2")

import torch
print(f"\nPyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
else:
    print("GPU     : Not available — training on CPU")
print(f"Device  : {DEVICE}")
print(f"Python  : {sys.version.split()[0]}")


import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print("All imports successful.")
print(f"  PyTorch     : {torch.__version__}")
print(f"  torchvision : {torchvision.__version__}")
print(f"  NumPy       : {np.__version__}")
print(f"  pandas      : {pd.__version__}")
print(f"  matplotlib  : {matplotlib.__version__}")


# Dataset
DATASET_ROOT = r"c:\Users\Asus\Downloads\viskom\Garbage classification\Garbage classification"
CLASSES      = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

# Hyperparameters
IMG_SIZE       = 224
BATCH_SIZE     = 32
NUM_EPOCHS_P1  = 15      # Phase 1: train head with frozen backbone
NUM_EPOCHS_P2  = 10      # Phase 2: fine-tune upper backbone layers
LR_P1          = 1e-3
LR_P2          = 1e-4
WEIGHT_DECAY   = 1e-4
EARLY_STOP_PAT = 5

# ImageNet normalization (required for MobileNetV2 pretrained weights)
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# Split fractions
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# Output directories
# Outputs go to  viskom/outputs/
OUTPUT_DIR = os.path.normpath(
    os.path.join(DATASET_ROOT, "..", "..", "outputs"))
MODEL_DIR  = os.path.join(OUTPUT_DIR, "mobilenetv2_model")
CKPT_P1    = os.path.join(OUTPUT_DIR, "ckpt_phase1_best.pth")
CKPT_P2    = os.path.join(OUTPUT_DIR, "ckpt_phase2_best.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# Color palette (one per class)
COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

print(f"Dataset root : {DATASET_ROOT}")
print(f"Classes      : {CLASSES}  ({NUM_CLASSES} classes)")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"Device       : {DEVICE}")


print("=" * 62)
print("DATASET INSPECTION")
print("=" * 62)

all_paths    = []
all_labels   = []
corrupted    = []
class_counts = {}

for cls in CLASSES:
    cls_dir = os.path.join(DATASET_ROOT, cls)
    valid   = []
    for fname in sorted(os.listdir(cls_dir)):
        fpath = os.path.join(cls_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with Image.open(fpath) as img:
                img.verify()          # integrity check without full decode
            with Image.open(fpath) as img:
                _ = img.mode          # confirm the mode is accessible
            valid.append(fpath)
            all_paths.append(fpath)
            all_labels.append(CLASS_TO_IDX[cls])
        except Exception as e:
            corrupted.append((fpath, str(e)))
    class_counts[cls] = len(valid)
    print(f"  {cls:<12} : {len(valid):>4} valid images")

print()
print(f"Total valid images     : {len(all_paths)}")
print(f"Total classes          : {NUM_CLASSES}")
print(f"Corrupted / unreadable : {len(corrupted)}")

if corrupted:
    print("\n⚠️  Corrupted files (NOT modified):")
    for fp, err in corrupted:
        print(f"   {fp}  →  {err}")
else:
    print("✅ No corrupted files found.")

print()
if len(all_paths) >= 1000:
    print(f"✅ Minimum 1,000-image requirement satisfied ({len(all_paths)} images).")
else:
    print(f"⚠️  Only {len(all_paths)} images found — below 1,000 minimum.")


# Class distribution bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars = ax1.bar(class_counts.keys(), class_counts.values(),
               color=COLORS, edgecolor='white', linewidth=1.5)
for bar, cnt in zip(bars, class_counts.values()):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(cnt), ha='center', fontweight='bold', fontsize=11)
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Class', fontsize=12)
ax1.set_ylabel('Number of Images', fontsize=12)
ax1.set_ylim(0, max(class_counts.values()) * 1.18)
ax1.grid(axis='y', alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)
ax1.tick_params(axis='x', rotation=15)

# Donut pie chart
wedges, texts, autotexts = ax2.pie(
    class_counts.values(),
    labels=[c.capitalize() for c in class_counts],
    autopct='%1.1f%%',
    colors=COLORS,
    startangle=90,
    pctdistance=0.78,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax2.add_artist(plt.Circle((0, 0), 0.55, fc='white'))
ax2.set_title('Class Proportion', fontsize=14, fontweight='bold', pad=12)

plt.suptitle('Garbage Classification — 5 Classes', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_class_distribution.png')
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {dist_path}")

counts_arr = np.array(list(class_counts.values()))
ratio = counts_arr.max() / counts_arr.min()
print(f"\nImbalance ratio (max / min): {ratio:.2f}×")
if ratio > 3:
    print("⚠️  Significant imbalance detected — class weights will be applied.")
else:
    print("✅ Imbalance ratio acceptable (< 3×) — class weights will still be used.")


X = np.array(all_paths)
y = np.array(all_labels)

# Step 1: Reserve TEST_FRAC as test set
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=TEST_FRAC, stratify=y, random_state=RANDOM_SEED)

# Step 2: From remaining 85%, take val_frac_of_tv ≈ 17.65% as validation
val_frac_of_tv = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=val_frac_of_tv, stratify=y_tv, random_state=RANDOM_SEED)

total = len(X)
print("Dataset split:")
print(f"  Training   : {len(X_train):>5}  ({100*len(X_train)/total:.1f}%)")
print(f"  Validation : {len(X_val):>5}  ({100*len(X_val)/total:.1f}%)")
print(f"  Test       : {len(X_test):>5}  ({100*len(X_test)/total:.1f}%)")
print(f"  Total      : {total:>5}  (100%)")

# Data leakage check
assert not (set(X_train) & set(X_val)),  "LEAKAGE: train ∩ val!"
assert not (set(X_train) & set(X_test)), "LEAKAGE: train ∩ test!"
assert not (set(X_val)   & set(X_test)), "LEAKAGE: val ∩ test!"
print("\n✅ No data leakage detected between any split pair.")

# Per-class counts across splits
print(f"\n  {'Class':<12}{'Train':>7}{'Val':>7}{'Test':>7}")
print("  " + "-" * 31)
for idx in range(NUM_CLASSES):
    cls = IDX_TO_CLASS[idx]
    t   = int((y_train == idx).sum())
    v   = int((y_val   == idx).sum())
    s   = int((y_test  == idx).sum())
    print(f"  {cls:<12}{t:>7}{v:>7}{s:>7}")


# Robust Tensor Transform (NumPy C-API independent)
class RobustToTensor:
    """Converts a PIL Image or ndarray to a FloatTensor in [0.0, 1.0].
    Directly uses PyTorch's native memory buffers without calling torch.from_numpy(),
    ensuring 100% compatibility across Python 3.11/3.13 environments and preventing
    'RuntimeError: Numpy is not available'.
    """
    def __call__(self, pic):
        if hasattr(pic, 'convert'):
            return torchvision.transforms.functional.pil_to_tensor(pic).float().div(255.0)
        return torchvision.transforms.functional.to_tensor(pic)
    
    def __repr__(self):
        return "RobustToTensor()"

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
    RobustToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    RobustToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("Training transform pipeline:")
for t in train_tf.transforms:
    print(f"  {type(t).__name__}")
print("\nValidation / Test transform pipeline:")
for t in eval_tf.transforms:
    print(f"  {type(t).__name__}")


class GarbageDataset(Dataset):
    """Custom PyTorch Dataset for garbage image classification."""

    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


# Instantiate datasets
train_ds = GarbageDataset(X_train, y_train, transform=train_tf)
val_ds   = GarbageDataset(X_val,   y_val,   transform=eval_tf)
test_ds  = GarbageDataset(X_test,  y_test,  transform=eval_tf)

# DataLoaders (num_workers=0 for Windows / Python 3.13 compat)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

imgs_b, lbls_b = next(iter(train_dl))
print(f"Batch shape — images : {imgs_b.shape}  (B, C, H, W)")
print(f"Batch shape — labels : {lbls_b.shape}")
print(f"Image dtype          : {imgs_b.dtype}")
print(f"Pixel range (approx) : [{imgs_b.min():.2f}, {imgs_b.max():.2f}]")
print(f"\nTrain batches : {len(train_dl)} × {BATCH_SIZE}")
print(f"Val   batches : {len(val_dl)}")
print(f"Test  batches : {len(test_dl)}")


# Sample images — one per class (from training set)
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(3.8 * NUM_CLASSES, 4.2))
fig.suptitle('Sample Images — One per Class (Training Set)',
             fontsize=13, fontweight='bold', y=1.02)

for ci, cls in enumerate(CLASSES):
    mask = np.where(y_train == ci)[0]
    img  = Image.open(X_train[mask[0]]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    axes[ci].imshow(img)
    axes[ci].set_title(cls.capitalize(), fontsize=12, fontweight='bold', pad=7)
    axes[ci].axis('off')
    for spine in axes[ci].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(COLORS[ci])
        spine.set_linewidth(4)

plt.tight_layout()
samples_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_sample_images.png')
plt.savefig(samples_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {samples_path}")


# Load pretrained MobileNetV2
try:
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    print("✅ Loaded MobileNetV2 — new API (MobileNet_V2_Weights.IMAGENET1K_V1)")
except (ImportError, AttributeError):
    model = models.mobilenet_v2(pretrained=True)
    print("✅ Loaded MobileNetV2 — legacy API (pretrained=True)")

# Phase 1: freeze entire backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head
#  Original : Sequential(Dropout(0.2), Linear(1280, 1000))
#  New      : Sequential(Dropout(0.3), Linear(1280, NUM_CLASSES))
in_feats = model.last_channel

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_feats, NUM_CLASSES)
)
model = model.to(DEVICE)

# Parameter summary
total_p  = sum(p.numel() for p in model.parameters())
train_p  = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_p = total_p - train_p

print(f"\nClassifier head : {model.classifier}")
print(f"\nParameter counts (Phase 1):")
print(f"  Total            : {total_p:>10,}")
print(f"  Trainable (head) : {train_p:>10,}")
print(f"  Frozen (backbone): {frozen_p:>10,}")


# Class weights
cw_vals   = compute_class_weight('balanced',
                                  classes=np.arange(NUM_CLASSES),
                                  y=y_train)
cw_tensor = torch.tensor(cw_vals, dtype=torch.float32).to(DEVICE)

print("Class weights:")
for cls, w in zip(CLASSES, cw_vals):
    print(f"  {cls:<12} : {w:.4f}")

criterion = nn.CrossEntropyLoss(weight=cw_tensor)

# Phase 1 optimizer (head only)
opt_p1 = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_P1, weight_decay=WEIGHT_DECAY
)
sch_p1 = optim.lr_scheduler.ReduceLROnPlateau(
    opt_p1, mode='min', factor=0.5, patience=3
)

print(f"\nLoss      : CrossEntropyLoss (weighted)")
print(f"Optimizer : Adam  (lr={LR_P1}, weight_decay={WEIGHT_DECAY})")
print(f"Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == lbls).sum().item()
        n          += imgs.size(0)
    return total_loss / n, 100.0 * correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        loss   = criterion(logits, lbls)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == lbls).sum().item()
        n          += imgs.size(0)
    return total_loss / n, 100.0 * correct / n


def train_model(model, tr_dl, va_dl, criterion, optimizer, scheduler,
                n_epochs, ckpt_path, phase, device,
                patience=EARLY_STOP_PAT):
    hist = {'tr_loss': [], 'va_loss': [],
            'tr_acc':  [], 'va_acc':  [], 'lr': []}
    best_va_acc = 0.0
    pat_ctr     = 0
    t0          = time.time()

    print(f"\n{'='*66}")
    print(f"  {phase}")
    print(f"{'='*66}")
    hdr = (f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  "
           f"{'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}  Status")
    print(hdr)
    print("  " + "-"*63)

    for ep in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_dl, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, va_dl, criterion, device)
        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step(va_loss)

        hist['tr_loss'].append(tr_loss)
        hist['va_loss'].append(va_loss)
        hist['tr_acc'].append(tr_acc)
        hist['va_acc'].append(va_acc)
        hist['lr'].append(cur_lr)

        status = ''
        if va_acc > best_va_acc:
            best_va_acc = va_acc
            pat_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            status = '✅ BEST'
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                status = '⏹ STOP'

        print(f"  {ep:>5}  {tr_loss:>8.4f}  {tr_acc:>6.2f}%  "
              f"{va_loss:>8.4f}  {va_acc:>6.2f}%  {cur_lr:>9.2e}  {status}")

        if '⏹' in status:
            print(f"\n  Early stopping triggered at epoch {ep}.")
            break

    elapsed = time.time() - t0
    print(f"\n  Best Val Acc : {best_va_acc:.2f}%")
    print(f"  Training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return hist, elapsed, best_va_acc


print("Training helper functions defined successfully.")


print(f"Phase 1 trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

hist_p1, time_p1, best_acc_p1 = train_model(
    model, train_dl, val_dl,
    criterion, opt_p1, sch_p1,
    n_epochs  = NUM_EPOCHS_P1,
    ckpt_path = CKPT_P1,
    phase     = "PHASE 1 — Frozen Backbone (Head Only)",
    device    = DEVICE
)
print(f"\n✅ Phase 1 best checkpoint saved: {CKPT_P1}")


# Load best Phase 1 weights
model.load_state_dict(torch.load(CKPT_P1, map_location=DEVICE))
print(f"Loaded Phase 1 best checkpoint: {CKPT_P1}")

# Unfreeze features[14:] + classifier
UNFREEZE_FROM = 14
for i, layer in enumerate(model.features):
    for param in layer.parameters():
        param.requires_grad = (i >= UNFREEZE_FROM)
for param in model.classifier.parameters():
    param.requires_grad = True

p2_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nUnfreezing features[{UNFREEZE_FROM}:] + classifier")
print(f"Phase 2 trainable parameters: {p2_trainable:,}")

# Phase 2 optimizer and scheduler
opt_p2 = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_P2, weight_decay=WEIGHT_DECAY
)
sch_p2 = optim.lr_scheduler.ReduceLROnPlateau(
    opt_p2, mode='min', factor=0.5, patience=3
)

hist_p2, time_p2, best_acc_p2 = train_model(
    model, train_dl, val_dl,
    criterion, opt_p2, sch_p2,
    n_epochs  = NUM_EPOCHS_P2,
    ckpt_path = CKPT_P2,
    phase     = "PHASE 2 — Fine-Tuning (features[14:] + classifier)",
    device    = DEVICE
)

# Select best checkpoint
if best_acc_p2 > best_acc_p1:
    BEST_CKPT = CKPT_P2
    print(f"\n✅ Phase 2 improved val accuracy → best checkpoint: {CKPT_P2}")
else:
    BEST_CKPT = CKPT_P1
    print(f"\nℹ️  Phase 1 was better or equal → best checkpoint: {CKPT_P1}")

print(f"\nPhase 1 best val acc : {best_acc_p1:.2f}%")
print(f"Phase 2 best val acc : {best_acc_p2:.2f}%")
print(f"Selected             : {BEST_CKPT}")


# Load best checkpoint
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
model.eval()
print(f"Evaluating checkpoint: {BEST_CKPT}\n")

# Inference over test set
all_preds = []
all_true  = []
all_probs = []

with torch.no_grad():
    for imgs, lbls in test_dl:
        imgs  = imgs.to(DEVICE)
        out   = model(imgs)
        probs = torch.softmax(out, dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_true.extend(lbls.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
all_probs = np.array(all_probs)

test_acc = accuracy_score(all_true, all_preds) * 100
print(f"Test samples  : {len(all_true)}")
print(f"Test Accuracy : {test_acc:.2f}%")


cls_labels = [c.capitalize() for c in CLASSES]

report_str = classification_report(
    all_true, all_preds, target_names=cls_labels, digits=4)

print("=" * 64)
print("CLASSIFICATION REPORT — TEST SET")
print("=" * 64)
print(report_str)

rpt_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_classification_report.txt')
with open(rpt_path, 'w', encoding='utf-8') as f:
    f.write("MobileNetV2 Garbage Classification — Test Set\n")
    f.write("=" * 64 + "\n")
    f.write(f"Best checkpoint : {BEST_CKPT}\n\n")
    f.write(report_str)
print(f"✅ Saved: {rpt_path}")

# Aggregate metrics
precision, recall, f1, support = precision_recall_fscore_support(
    all_true, all_preds, average=None, labels=np.arange(NUM_CLASSES))
macro_p,   macro_r,   macro_f1,   _ = precision_recall_fscore_support(
    all_true, all_preds, average='macro')
weight_p,  weight_r,  weight_f1,  _ = precision_recall_fscore_support(
    all_true, all_preds, average='weighted')

print("\nSummary metrics:")
print(f"  Test Accuracy       : {test_acc:.2f}%")
print(f"  Macro Precision     : {macro_p  *100:.2f}%")
print(f"  Macro Recall        : {macro_r  *100:.2f}%")
print(f"  Macro F1-Score      : {macro_f1 *100:.2f}%")
print(f"  Weighted Precision  : {weight_p *100:.2f}%")
print(f"  Weighted Recall     : {weight_r *100:.2f}%")
print(f"  Weighted F1-Score   : {weight_f1*100:.2f}%")


# Confusion matrix (matplotlib imshow)
cm      = confusion_matrix(all_true, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm],
    ['d', '.2f'],
    ['Confusion Matrix (Counts)', 'Confusion Matrix (Normalized)']
):
    im = ax.imshow(data, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=(1 if fmt == '.2f' else None))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(cls_labels, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(cls_labels, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    thresh = data.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = 'white' if data[i, j] > thresh else 'black'
            ax.text(j, i, format(data[i, j], fmt),
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

plt.suptitle('MobileNetV2 — Test Set Confusion Matrix',
             fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {cm_path}")


# Per-class metrics bar chart
x, w = np.arange(NUM_CLASSES), 0.25
fig, ax = plt.subplots(figsize=(12, 6))

for i, (metric, vals, color) in enumerate([
    ('Precision', precision, '#4E79A7'),
    ('Recall',    recall,    '#F28E2B'),
    ('F1-Score',  f1,        '#59A14F')
]):
    bars = ax.bar(x + (i-1)*w, vals*100, w,
                  label=metric, color=color, alpha=0.9, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(cls_labels, fontsize=11)
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Per-Class Precision, Recall, and F1-Score — Test Set',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mobilenetv2_per_class_metrics.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: mobilenetv2_per_class_metrics.png")


# Combine Phase 1 + Phase 2 histories
comb_tr_loss = hist_p1['tr_loss'] + hist_p2['tr_loss']
comb_va_loss = hist_p1['va_loss'] + hist_p2['va_loss']
comb_tr_acc  = hist_p1['tr_acc']  + hist_p2['tr_acc']
comb_va_acc  = hist_p1['va_acc']  + hist_p2['va_acc']
p1_end       = len(hist_p1['tr_loss'])
epochs_rng   = range(1, len(comb_tr_loss) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(epochs_rng, comb_tr_loss, '-o', color='#E15759',
         label='Train Loss',       ms=3, lw=2)
ax1.plot(epochs_rng, comb_va_loss, '--s', color='#4E79A7',
         label='Validation Loss',  ms=3, lw=2)
ax1.axvline(x=p1_end + 0.5, color='#B0B0B0', linestyle=':',
            lw=1.5, label='Phase 2 start')
ax1.set_title('Loss Curves',    fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch',         fontsize=11)
ax1.set_ylabel('Loss',          fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)

ax2.plot(epochs_rng, comb_tr_acc, '-o', color='#E15759',
         label='Train Accuracy',      ms=3, lw=2)
ax2.plot(epochs_rng, comb_va_acc, '--s', color='#4E79A7',
         label='Validation Accuracy', ms=3, lw=2)
ax2.axvline(x=p1_end + 0.5, color='#B0B0B0', linestyle=':',
            lw=1.5, label='Phase 2 start')
ax2.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch',          fontsize=11)
ax2.set_ylabel('Accuracy (%)',   fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)

plt.suptitle('MobileNetV2 — Training Curves (Phase 1 + Phase 2)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
tc_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_training_curves.png')
plt.savefig(tc_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {tc_path}")


# Separate correct and incorrect predictions
conf_scores   = all_probs.max(axis=1)
correct_idx   = [i for i in range(len(all_preds)) if all_preds[i] == all_true[i]]
incorrect_idx = [i for i in range(len(all_preds)) if all_preds[i] != all_true[i]]

n_show = 8
c_idx  = correct_idx[:min(n_show, len(correct_idx))]
w_idx  = incorrect_idx[:min(n_show, len(incorrect_idx))]

fig, axes = plt.subplots(2, n_show, figsize=(2.8 * n_show, 7.0))
fig.suptitle(
    'MobileNetV2 — Prediction Grid\n'
    '🟢 Top row: Correct predictions  |  🔴 Bottom row: Incorrect predictions',
    fontsize=13, fontweight='bold'
)

row_configs = [
    (c_idx, '#27AE60'),
    (w_idx, '#E74C3C')
]

for row, (idxs, border_color) in enumerate(row_configs):
    for col in range(n_show):
        ax = axes[row][col]
        if col < len(idxs):
            idx      = idxs[col]
            img      = Image.open(X_test[idx]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            true_cls = CLASSES[all_true[idx]].capitalize()
            pred_cls = CLASSES[all_preds[idx]].capitalize()
            conf     = conf_scores[idx] * 100
            ax.imshow(img)
            ax.set_title(f"T: {true_cls}\nP: {pred_cls}\n{conf:.1f}%",
                         fontsize=7.5, color=border_color, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.5)
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
pred_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_predictions.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {pred_path}")
print(f"  Correct   shown : {len(c_idx)}")
print(f"  Incorrect shown : {len(w_idx)}")


# Parameter counts
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params

print("=" * 55)
print("COMPUTATIONAL EFFICIENCY ANALYSIS")
print("=" * 55)
print(f"\nParameters:")
print(f"  Total        : {total_params:>10,}")
print(f"  Trainable    : {trainable_params:>10,}")
print(f"  Frozen       : {frozen_params:>10,}")

# Save full model checkpoint
os.makedirs(MODEL_DIR, exist_ok=True)
model_save_path = os.path.join(MODEL_DIR, 'mobilenetv2_best.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'classes'         : CLASSES,
    'num_classes'     : NUM_CLASSES,
    'img_size'        : IMG_SIZE,
    'architecture'    : 'MobileNetV2',
    'imagenet_mean'   : IMAGENET_MEAN,
    'imagenet_std'    : IMAGENET_STD,
}, model_save_path)
model_mb = os.path.getsize(model_save_path) / 1024**2
print(f"\nModel saved  : {model_save_path}")
print(f"Model size   : {model_mb:.2f} MB")

# Single-image inference latency
model.eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

with torch.no_grad():
    for _ in range(5): _ = model(dummy)

latencies = []
with torch.no_grad():
    for _ in range(50):
        t0 = time.perf_counter()
        _  = model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

mean_lat = float(np.mean(latencies))
std_lat  = float(np.std(latencies))
print(f"\nSingle-image inference (device={DEVICE}, n=50):")
print(f"  Mean   : {mean_lat:.2f} ms ± {std_lat:.2f} ms")
print(f"  Min    : {min(latencies):.2f} ms")
print(f"  Max    : {max(latencies):.2f} ms")

# Batch throughput
dummy_b = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    for _ in range(3): _ = model(dummy_b)
btimes = []
with torch.no_grad():
    for _ in range(10):
        t0 = time.perf_counter()
        _  = model(dummy_b)
        btimes.append(time.perf_counter() - t0)
throughput = BATCH_SIZE / float(np.mean(btimes))
print(f"\nBatch throughput (batch={BATCH_SIZE}, n=10):")
print(f"  Mean batch time : {np.mean(btimes)*1000:.1f} ms")
print(f"  Throughput      : {throughput:.1f} images/second")

# Training time summary
total_train_time = time_p1 + time_p2
print(f"\nTraining time:")
print(f"  Phase 1 : {time_p1:.0f}s ({time_p1/60:.1f} min)")
print(f"  Phase 2 : {time_p2:.0f}s ({time_p2/60:.1f} min)")
print(f"  Total   : {total_train_time:.0f}s ({total_train_time/60:.1f} min)")
print(f"\nHardware : {DEVICE}")


# Efficiency summary figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('MobileNetV2 — Computational Efficiency Summary',
             fontsize=14, fontweight='bold')

# Donut: parameter breakdown
p_data = [trainable_params, frozen_params]
p_labs = [f'Trainable\n{trainable_params/1e6:.2f}M',
          f'Frozen\n{frozen_params/1e6:.2f}M']
axes[0].pie(p_data, labels=p_labs, autopct='%1.1f%%',
            colors=['#4E79A7', '#AECDE8'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            startangle=90, pctdistance=0.78)
axes[0].add_artist(plt.Circle((0, 0), 0.55, fc='white'))
axes[0].set_title(f'Parameters\n(Total: {total_params/1e6:.2f}M)',
                  fontsize=11, fontweight='bold')

# Histogram: latency distribution
axes[1].hist(latencies, bins=15, color='#F28E2B',
             edgecolor='white', linewidth=1.5, alpha=0.9)
axes[1].axvline(mean_lat, color='#E15759', linestyle='--', linewidth=2.5,
                label=f'Mean: {mean_lat:.1f} ms')
axes[1].set_title('Inference Latency Distribution (CPU)',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Latency (ms)', fontsize=10)
axes[1].set_ylabel('Count',        fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].spines[['top', 'right']].set_visible(False)

# Bar: key metrics
m_labels = ['Test Acc\n(%)', 'Macro F1\n(%)', 'Model\nSize (MB)',
            'Params\n(M)', 'Latency\n(ms)', 'Throughput\n(img/s)']
m_vals   = [test_acc, macro_f1*100, model_mb,
            total_params/1e6, mean_lat, throughput]
m_colors = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#76B7B2', '#B07AA1']
bars = axes[2].bar(m_labels, m_vals, color=m_colors,
                   edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, m_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')
axes[2].set_title('Key Metrics at a Glance', fontsize=11, fontweight='bold')
axes[2].tick_params(axis='x', labelsize=8)
axes[2].grid(axis='y', alpha=0.3)
axes[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
eff_path = os.path.join(OUTPUT_DIR, 'mobilenetv2_efficiency_summary.png')
plt.savefig(eff_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {eff_path}")


# Save all test metrics to CSV
rows = [
    ('Test Accuracy (%)',            f'{test_acc:.4f}'),
    ('Macro Precision (%)',          f'{macro_p*100:.4f}'),
    ('Macro Recall (%)',             f'{macro_r*100:.4f}'),
    ('Macro F1-Score (%)',           f'{macro_f1*100:.4f}'),
    ('Weighted Precision (%)',       f'{weight_p*100:.4f}'),
    ('Weighted Recall (%)',          f'{weight_r*100:.4f}'),
    ('Weighted F1-Score (%)',        f'{weight_f1*100:.4f}'),
]
for ci, cls in enumerate(CLASSES):
    rows.append((f'F1 {cls.capitalize()} (%)', f'{f1[ci]*100:.4f}'))
rows += [
    ('Total Parameters',            f'{total_params}'),
    ('Trainable Parameters',        f'{trainable_params}'),
    ('Frozen Parameters',           f'{frozen_params}'),
    ('Model Size (MB)',              f'{model_mb:.4f}'),
    ('Mean Inference Latency (ms)', f'{mean_lat:.4f}'),
    ('Inference Throughput (img/s)',f'{throughput:.2f}'),
    ('Phase 1 Training Time (s)',   f'{time_p1:.1f}'),
    ('Phase 2 Training Time (s)',   f'{time_p2:.1f}'),
    ('Total Training Time (s)',     f'{total_train_time:.1f}'),
    ('Training Device',             str(DEVICE)),
    ('Best Checkpoint',             BEST_CKPT),
]
df_metrics = pd.DataFrame(rows, columns=['Metric', 'Value'])
csv_path   = os.path.join(OUTPUT_DIR, 'mobilenetv2_test_metrics.csv')
df_metrics.to_csv(csv_path, index=False)
print(f"✅ Saved: {csv_path}\n")
print(df_metrics.to_string(index=False))


# Final output file inventory
print("\n" + "=" * 68)
print("  OUTPUT FILES")
print("=" * 68)
for root, dirs, files in os.walk(OUTPUT_DIR):
    level  = root.replace(OUTPUT_DIR, '').count(os.sep)
    indent = '  ' * level
    if level > 0:
        print(f"{indent}📁 {os.path.basename(root)}/")
    for fname in sorted(files):
        fp   = os.path.join(root, fname)
        size = os.path.getsize(fp)
        print(f"{'  '*(level+1)}📄 {fname:<52} {size:>10,} B")

print("\n" + "=" * 68)
print("  EXPERIMENT COMPLETE")
print(f"  Architecture    : MobileNetV2")
print(f"  Test Accuracy   : {test_acc:.2f}%")
print(f"  Macro F1-Score  : {macro_f1*100:.2f}%")
print(f"  Total params    : {total_params:,}")
print(f"  Model size      : {model_mb:.2f} MB")
print(f"  Training time   : {total_train_time/60:.1f} min (CPU)")
print("=" * 68)


# ==================== ResNet-18 ====================


import os
import sys
import time
import random
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

from PIL import Image

warnings.filterwarnings("ignore")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Reproducibility
SEED = 42

def set_seed(seed: int = SEED):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
print(f"✅ Global seed set to {SEED} for reproducibility.")

# Configuration
CONFIG = {
    "num_classes": 6,
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 0,

    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,

    # Normalization (ImageNet statistics)
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],

    "model_name": "resnet18",
    "output_dir": "outputs",
}

# Device selection: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"📱 Device: {DEVICE}")

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"📂 Outputs will be saved to: {CONFIG['output_dir']}/")

# Dataset Path Resolution

def find_dataset_root(search_paths: list) -> str:
    """
    Locate the directory containing the 6 class subdirectories.

    The Kaggle Garbage Classification dataset often extracts to a nested
    structure: 'Garbage classification/Garbage classification/<classes>/'.
    This function searches common locations and resolves the correct root.

    Args:
        search_paths: List of directories to search for the dataset.

    Returns:
        Path to the directory containing class subdirectories.

    Raises:
        FileNotFoundError: If the dataset cannot be found.
    """
    expected_classes = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}

    for base_path in search_paths:
        base = Path(base_path)
        if not base.exists():
            continue
        # Walk up to 3 levels deep to find the folder with 6 class subdirs
        for root, dirs, _ in os.walk(base):
            found_classes = set(dirs) & expected_classes
            if found_classes == expected_classes:
                print(f"✅ Dataset found at: {root}")
                return root
            # Don't recurse too deep
            if root.count(os.sep) - str(base).count(os.sep) >= 3:
                break

    raise FileNotFoundError(
        "Could not find the Garbage Classification dataset. "
        "Please download it from Kaggle and place it in one of these paths:\n"
        + "\n".join(f"  - {p}" for p in search_paths)
    )


# Search in common locations (adjust paths as needed)
SEARCH_PATHS = [
    ".",                                          # Current directory
    "..",                                         # Parent directory
    "../Garbage classification",                  # Sibling folder
    "Garbage classification",                     # Subdirectory
    os.path.expanduser("~/datasets"),             # Home datasets folder
    "/content/drive/MyDrive",                     # Google Drive on Colab
    "/content",                                   # Colab root
]

DATASET_ROOT = find_dataset_root(SEARCH_PATHS)
print(f"📁 Dataset root: {DATASET_ROOT}")

# Verify Dataset Structure

# Use ImageFolder to load the full dataset (for indexing & label extraction)
full_dataset = ImageFolder(root=DATASET_ROOT)

# Class names and distribution
class_names = full_dataset.classes
num_images = len(full_dataset)
targets = np.array(full_dataset.targets)

print(f"\n{'='*55}")
print(f"  GARBAGE CLASSIFICATION DATASET SUMMARY")
print(f"{'='*55}")
print(f"  Total images : {num_images}")
print(f"  Classes ({len(class_names)}): {class_names}")
print(f"{'='*55}")
print(f"\n  Per-class distribution:")

class_counts = {}
for idx, name in enumerate(class_names):
    count = int(np.sum(targets == idx))
    pct = count / num_images * 100
    class_counts[name] = count
    print(f"    {name:>12s}: {count:>4d} images ({pct:5.1f}%)")

print(f"\n  ✅ Satisfies ≥1,000 images and ≥5 classes requirements.")

# Stratified Split

indices = np.arange(num_images)
labels = targets

# First split: 70% train, 30% temp (val+test)
train_idx, temp_idx, train_labels, temp_labels = train_test_split(
    indices, labels,
    test_size=(1 - CONFIG["train_ratio"]),
    stratify=labels,
    random_state=SEED,
)

# Second split: 50% of temp → 15% val, 15% test
val_idx, test_idx, val_labels, test_labels = train_test_split(
    temp_idx, temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED,
)

print(f"{'='*55}")
print(f"  STRATIFIED SPLIT SUMMARY")
print(f"{'='*55}")
print(f"  Training set   : {len(train_idx):>5d} images ({len(train_idx)/num_images*100:.1f}%)")
print(f"  Validation set : {len(val_idx):>5d} images ({len(val_idx)/num_images*100:.1f}%)")
print(f"  Test set       : {len(test_idx):>5d} images ({len(test_idx)/num_images*100:.1f}%)")
print(f"  Total          : {len(train_idx)+len(val_idx)+len(test_idx):>5d} images")
print(f"{'='*55}")

# Verify stratification — class distribution should be similar across splits
print(f"\n  Per-class distribution across splits:")
print(f"  {'Class':>12s} | {'Train':>6s} | {'Val':>5s} | {'Test':>5s}")
print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")
for idx, name in enumerate(class_names):
    tr = int(np.sum(train_labels == idx))
    va = int(np.sum(val_labels == idx))
    te = int(np.sum(test_labels == idx))
    print(f"  {name:>12s} | {tr:>6d} | {va:>5d} | {te:>5d}")

# Transform Pipelines

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
])

print("✅ Transform pipelines defined.")
print(f"\n🔄 Training augmentation pipeline:")
for t in train_transform.transforms:
    print(f"   → {t}")
print(f"\n📐 Validation/Test pipeline (deterministic):")
for t in val_test_transform.transforms:
    print(f"   → {t}")

# Custom Dataset Wrapper

class SubsetWithTransform(Dataset):
    """
    Wraps a Subset of an ImageFolder dataset with a specific transform.

    This prevents data leakage by ensuring augmentation is ONLY applied
    to the training split, while val/test get deterministic preprocessing.
    """

    def __init__(self, dataset: ImageFolder, indices: np.ndarray, transform=None):
        """
        Args:
            dataset: The full ImageFolder dataset.
            indices: Array of indices for this subset.
            transform: Transform pipeline to apply to images in this subset.
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image_path, label = self.dataset.imgs[real_idx]

        # Load image (ensures we start from raw pixels every time)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Create subset datasets with appropriate transforms
train_dataset = SubsetWithTransform(full_dataset, train_idx, transform=train_transform)
val_dataset   = SubsetWithTransform(full_dataset, val_idx,   transform=val_test_transform)
test_dataset  = SubsetWithTransform(full_dataset, test_idx,  transform=val_test_transform)

print(f"✅ Dataset wrappers created:")
print(f"   Train : {len(train_dataset):>5d} images (WITH augmentation)")
print(f"   Val   : {len(val_dataset):>5d} images (deterministic only)")
print(f"   Test  : {len(test_dataset):>5d} images (deterministic only)")

# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=(DEVICE.type == "cuda"),
)

print(f"✅ DataLoaders created (batch_size={CONFIG['batch_size']}, num_workers={CONFIG['num_workers']}):")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}")
print(f"   Test batches  : {len(test_loader)}")

# Model Initialization

def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet-18 model with a custom classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        Modified ResNet-18 model.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        print("✅ Loaded pretrained ResNet-18 (ImageNet1K_V1 weights).")
    else:
        model = models.resnet18(weights=None)
        print("⚠️  Initialized ResNet-18 from scratch (no pretrained weights).")

    # Replace the final fully connected layer
    # Original: nn.Linear(512, 1000) → New: nn.Linear(512, 6)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"   Replaced FC layer: {in_features} → {num_classes} classes")

    # Unfreeze ALL layers for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    return model


model = build_resnet18(num_classes=CONFIG["num_classes"])
model = model.to(DEVICE)

# Parameter Count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*55}")
print(f"  RESNET-18 MODEL SUMMARY")
print(f"{'='*55}")
print(f"  Total parameters     : {total_params:>12,d}")
print(f"  Trainable parameters : {trainable_params:>12,d}")
print(f"  Non-trainable params : {total_params - trainable_params:>12,d}")
print(f"  Device               : {DEVICE}")
print(f"{'='*55}")

# Loss, Optimizer, Scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=CONFIG["scheduler_factor"],
    patience=CONFIG["scheduler_patience"],
)

print(f"✅ Training components initialized:")
print(f"   Loss      : CrossEntropyLoss")
print(f"   Optimizer : AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
print(f"   Scheduler : ReduceLROnPlateau (factor={CONFIG['scheduler_factor']}, patience={CONFIG['scheduler_patience']})")
print(f"   Epochs    : {CONFIG['epochs']}")

# Training & Validation Functions

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate the model on a validation/test set.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Main Training Loop

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
    "lr": [], "epoch_time": [],
}

best_val_acc = 0.0
best_val_loss = float("inf")
best_model_state = None
best_epoch = 0

checkpoint_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['model_name']}_best.pth")

print(f"\n{'='*70}")
print(f"  TRAINING RESNET-18 — {CONFIG['epochs']} Epochs")
print(f"{'='*70}")

# Measure total training time
total_train_start = time.perf_counter()

for epoch in range(1, CONFIG["epochs"] + 1):
    epoch_start = time.perf_counter()

    # Synchronize GPU timing if applicable
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    # Synchronize GPU timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    current_lr = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)
    history["epoch_time"].append(epoch_time)

    # Step scheduler (based on validation loss)
    scheduler.step(val_loss)

    # Best model checkpointing (based on validation accuracy)
    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, checkpoint_path)
        improved = " ★ BEST"

    print(
        f"  Epoch [{epoch:>2d}/{CONFIG['epochs']}] "
        f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
        f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% "
        f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{improved}"
    )

# Synchronize final timing
if torch.cuda.is_available():
    torch.cuda.synchronize()

total_train_time = time.perf_counter() - total_train_start
avg_epoch_time = total_train_time / CONFIG["epochs"]

print(f"\n{'='*70}")
print(f"  ✅ Training complete!")
print(f"  Best epoch           : {best_epoch}")
print(f"  Best validation acc  : {best_val_acc*100:.2f}%")
print(f"  Best validation loss : {best_val_loss:.4f}")
print(f"  Total training time  : {total_train_time:.2f}s")
print(f"  Avg time per epoch   : {avg_epoch_time:.2f}s")
print(f"  Checkpoint saved to  : {checkpoint_path}")
print(f"{'='*70}")

# Load Best Model

model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
model.eval()
print(f"✅ Loaded best model from epoch {best_epoch} (val_acc={best_val_acc*100:.2f}%)")

# Collect Predictions

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Overall Accuracy
test_accuracy = accuracy_score(all_labels, all_preds)

# Precision, Recall, F1
macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0
)
weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0
)

# Full Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

print(f"\n{'='*60}")
print(f"  RESNET-18 — TEST SET CLASSIFICATION RESULTS")
print(f"{'='*60}")
print(f"  Overall Test Accuracy : {test_accuracy*100:.2f}%")
print(f"{'='*60}")
print(f"  Macro-Averaged    Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f} | F1: {macro_f1:.4f}")
print(f"  Weighted-Averaged Precision: {weighted_prec:.4f} | Recall: {weighted_rec:.4f} | F1: {weighted_f1:.4f}")
print(f"{'='*60}")
print(f"\n  Full Per-Class Classification Report:")
print(report)

# Model Size on Disk
model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

# Inference Latency & Throughput
# Warm-up run (for GPU JIT compilation / caching)
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Measure inference time over entire test set
num_test_images = len(test_dataset)

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_start = time.perf_counter()
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(DEVICE)
        _ = model(images)

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_time = time.perf_counter() - inference_start
latency_ms = (inference_time * 1000) / num_test_images
throughput = num_test_images / inference_time

# Summary Dictionary
metrics_summary = {
    "Model": "ResNet-18",
    "Total Parameters": f"{total_params:,}",
    "Trainable Parameters": f"{trainable_params:,}",
    "Model Size (MB)": f"{model_size_mb:.2f}",
    "Total Training Time (s)": f"{total_train_time:.2f}",
    "Avg Time per Epoch (s)": f"{avg_epoch_time:.2f}",
    "Test Accuracy (%)": f"{test_accuracy*100:.2f}",
    "Macro Precision": f"{macro_prec:.4f}",
    "Macro Recall": f"{macro_rec:.4f}",
    "Macro F1-Score": f"{macro_f1:.4f}",
    "Weighted Precision": f"{weighted_prec:.4f}",
    "Weighted Recall": f"{weighted_rec:.4f}",
    "Weighted F1-Score": f"{weighted_f1:.4f}",
    "Inference Latency (ms/img)": f"{latency_ms:.2f}",
    "Throughput (img/s)": f"{throughput:.2f}",
    "Best Epoch": str(best_epoch),
}

print(f"\n{'='*60}")
print(f"  RESNET-18 — COMPUTATIONAL EFFICIENCY METRICS")
print(f"{'='*60}")
for key, value in metrics_summary.items():
    print(f"  {key:<30s}: {value}")
print(f"{'='*60}")

# Save Metrics to CSV
metrics_df = pd.DataFrame([metrics_summary])
metrics_csv_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['model_name']}_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\n📄 Metrics saved to: {metrics_csv_path}")

# Plot 1: Training Curves

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, CONFIG["epochs"] + 1)

ax1.plot(epochs_range, history["train_loss"], "b-o", markersize=4, label="Train Loss")
ax1.plot(epochs_range, history["val_loss"], "r-o", markersize=4, label="Val Loss")
ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], "b-o", markersize=4, label="Train Acc")
ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], "r-o", markersize=4, label="Val Acc")
ax2.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle("ResNet-18 — Training Curves", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

curves_path = os.path.join(CONFIG["output_dir"], "resnet18_training_curves.png")
fig.savefig(curves_path, dpi=300, bbox_inches="tight")
print(f"📊 Saved: {curves_path}")
plt.show()

# Plot 2: Confusion Matrix

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"shrink": 0.8},
    ax=ax,
    linewidths=0.5,
    linecolor="gray",
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("ResNet-18 — Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(CONFIG["output_dir"], "resnet18_confusion_matrix.png")
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
print(f"📊 Saved: {cm_path}")
plt.show()

# Plot 3: Sample Predictions

def denormalize(tensor, mean, std):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# Select 12 random test images (mix of correct and incorrect if possible)
np.random.seed(SEED)
sample_indices = np.random.choice(len(test_dataset), size=12, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, idx in enumerate(sample_indices):
    image, true_label = test_dataset[idx]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(DEVICE))
        prob = torch.softmax(output, dim=1)
        confidence, pred_label = prob.max(1)
        confidence = confidence.item() * 100
        pred_label = pred_label.item()

    # Denormalize for display
    img_display = denormalize(image, CONFIG["mean"], CONFIG["std"])
    img_np = img_display.permute(1, 2, 0).numpy()

    axes[i].imshow(img_np)
    axes[i].axis("off")

    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    is_correct = true_label == pred_label
    color = "green" if is_correct else "red"

    axes[i].set_title(
        f"True: {true_name}\nPred: {pred_name} ({confidence:.1f}%)",
        fontsize=10,
        color=color,
        fontweight="bold",
    )

fig.suptitle("ResNet-18 — Sample Test Predictions", fontsize=16, fontweight="bold")
plt.tight_layout()

samples_path = os.path.join(CONFIG["output_dir"], "resnet18_sample_predictions.png")
fig.savefig(samples_path, dpi=300, bbox_inches="tight")
print(f"📊 Saved: {samples_path}")
plt.show()

# Final Summary Table

print(f"\n{'='*70}")
print(f"  RESNET-18 — COMPLETE RESULTS SUMMARY")
print(f"  (Ready for 3-Page Report)")
print(f"{'='*70}")
print(f"")
print(f"  ┌──────────────────────────────────────────────────────────────┐")
print(f"  │                 CLASSIFICATION METRICS                       │")
print(f"  ├──────────────────────────────┬───────────────────────────────┤")
print(f"  │ Overall Test Accuracy        │ {test_accuracy*100:>27.2f}% │")
print(f"  │ Macro Precision              │ {macro_prec:>28.4f} │")
print(f"  │ Macro Recall                 │ {macro_rec:>28.4f} │")
print(f"  │ Macro F1-Score               │ {macro_f1:>28.4f} │")
print(f"  │ Weighted Precision           │ {weighted_prec:>28.4f} │")
print(f"  │ Weighted Recall              │ {weighted_rec:>28.4f} │")
print(f"  │ Weighted F1-Score            │ {weighted_f1:>28.4f} │")
print(f"  ├──────────────────────────────┼───────────────────────────────┤")
print(f"  │              EFFICIENCY METRICS                              │")
print(f"  ├──────────────────────────────┼───────────────────────────────┤")
print(f"  │ Total Parameters             │ {total_params:>28,} │")
print(f"  │ Trainable Parameters         │ {trainable_params:>28,} │")
print(f"  │ Model Size (MB)              │ {model_size_mb:>27.2f}  │")
print(f"  │ Total Training Time (s)      │ {total_train_time:>27.2f}  │")
print(f"  │ Avg Time per Epoch (s)       │ {avg_epoch_time:>27.2f}  │")
print(f"  │ Inference Latency (ms/img)   │ {latency_ms:>27.2f}  │")
print(f"  │ Throughput (img/s)           │ {throughput:>27.2f}  │")
print(f"  │ Best Epoch                   │ {best_epoch:>28d} │")
print(f"  └──────────────────────────────┴───────────────────────────────┘")
print(f"")
print(f"  📊 Visualizations saved:")
print(f"     • {curves_path}")
print(f"     • {cm_path}")
print(f"     • {samples_path}")
print(f"  📄 Metrics CSV: {metrics_csv_path}")
print(f"  💾 Best model : {checkpoint_path}")
print(f"{'='*70}")

# Save Training History

history_df = pd.DataFrame({
    "epoch": list(range(1, CONFIG["epochs"] + 1)),
    "train_loss": history["train_loss"],
    "train_acc": history["train_acc"],
    "val_loss": history["val_loss"],
    "val_acc": history["val_acc"],
    "learning_rate": history["lr"],
    "epoch_time_s": history["epoch_time"],
})

history_csv_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['model_name']}_training_history.csv")
history_df.to_csv(history_csv_path, index=False)
print(f"📄 Training history saved to: {history_csv_path}")
print(history_df.to_string(index=False))
