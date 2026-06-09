import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
DATA_DIR = "./data/patches"   # Structure: ./data/patches/slide_01/patch_1.png
LABELS_CSV = "./data/labels.csv" # Columns: slide_id, label
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ----------------------------------------------------------------------
# 2. DATA LOAD & SIMPLE QC
# ----------------------------------------------------------------------
def is_valid_patch(img):
    """Simple check to ignore mostly white/empty background patches."""
    gray = img.convert("L")
    mean_val = np.mean(gray)
    return mean_val < 220  # If average brightness is near white (255), drop it


def load_dataset_file_paths(data_dir, labels_csv):
    """Loads CSV labels and finds all valid patch images per slide."""
    df = pd.read_csv(labels_csv)
    slide_to_label = dict(zip(df['slide_id'], df['label']))
    
    # Simple patient/slide-level split to prevent data leakage
    all_slides = list(slide_to_label.keys())
    train_slides, test_slides = train_test_split(all_slides, test_size=0.2, random_state=42)
    
    train_paths, train_labels, train_slide_ids = [], [], []
    test_paths, test_labels, test_slide_ids = [], [], []

    for slide_id in all_slides:
        slide_folder = os.path.join(data_dir, slide_id)
        if not os.path.exists(slide_folder):
            continue
            
        label = slide_to_label[slide_id]
        patch_files = glob.glob(os.path.join(slide_folder, "*.png"))
        
        for p in patch_files:
            img = Image.open(p)
            if is_valid_patch(img):  # QC filtering
                if slide_id in train_slides:
                    train_paths.append(p)
                    train_labels.append(label)
                    train_slide_ids.append(slide_id)
                else:
                    test_paths.append(p)
                    test_labels.append(label)
                    test_slide_ids.append(slide_id)

    print(f"Loaded {len(train_paths)} train patches and {len(test_paths)} test patches.")
    return (train_paths, train_labels, train_slide_ids), (test_paths, test_labels, test_slide_ids)


# ----------------------------------------------------------------------
# 3. PYTORCH DATASET
# ----------------------------------------------------------------------
class SimplePatchDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data augmentation for training, basic normalization for testing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ----------------------------------------------------------------------
# 4. MODEL & TRAINING LOOP
# ----------------------------------------------------------------------
def train_model(train_loader, num_classes=2):
    # Pretrained ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")

    return model


# ----------------------------------------------------------------------
# 5. SLIDE-LEVEL EVALUATION (MAJORITY VOTE)
# ----------------------------------------------------------------------
def evaluate_by_slide(model, test_paths, test_labels, test_slide_ids):
    dataset = SimplePatchDataset(test_paths, test_labels, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    patch_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            patch_preds.extend(predicted.cpu().numpy())

    # Group patch predictions by slide ID
    results = {}
    true_labels = {}
    
    for slide_id, pred, true_label in zip(test_slide_ids, patch_preds, test_labels):
        if slide_id not in results:
            results[slide_id] = []
            true_labels[slide_id] = true_label
        results[slide_id].append(pred)

    # Majority vote per slide
    final_true = []
    final_pred = []
    
    for slide_id, preds in results.items():
        majority_vote = 1 if sum(preds) > (len(preds) / 2) else 0
        final_pred.append(majority_vote)
        final_true.append(true_labels[slide_id])

    print("\n--- Slide-Level Metrics ---")
    print(f"Accuracy: {accuracy_score(final_true, final_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(final_true, final_pred))
    print("\nClassification Report:")
    print(classification_report(final_true, final_pred))


# ----------------------------------------------------------------------
# 6. MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")

    # 1. Load data
    train_data, test_data = load_dataset_file_paths(DATA_DIR, LABELS_CSV)
    
    if len(train_data[0]) == 0:
        print("No image patches found. Please check your data directory.")
        return

    train_dataset = SimplePatchDataset(train_data[0], train_data[1], transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Train model
    print("\nStarting Training...")
    model = train_model(train_loader)

    # 3. Evaluate at slide-level
    print("\nEvaluating Model...")
    evaluate_by_slide(model, test_data[0], test_data[1], test_data[2])


if __name__ == "__main__":
    main()
