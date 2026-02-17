import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset_loader import DeepfakeDataset
import numpy as np
import os

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Dataset
full_dataset = DeepfakeDataset("dataset/train_faces_fixed")

# --- AUTO-CALCULATE CLASS WEIGHTS ---
# We count how many 'real' (0) and 'fake' (1) images we have
targets = full_dataset.labels
class_counts = np.bincount(targets)
total_samples = len(targets)

print(f"Dataset Counts -> Real: {class_counts[0]}, Fake: {class_counts[1]}")

# Calculate weights: Weight = Total / (Num_Classes * Class_Count)
# This gives higher weight to the smaller class
class_weights = 1. / class_counts
sample_weights = class_weights[targets]

# Create a sampler that draws the minority class more often
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(full_dataset, batch_size=16, sampler=sampler)
# ------------------------------------

# 3. Load Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.to(device)

# 4. Training Config
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) # Added weight_decay to reduce overfitting
criterion = nn.CrossEntropyLoss()

print("Starting Balanced Training...")

# Train for 3 Epochs (usually enough if the weights are balanced)
for epoch in range(3):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy on the fly
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 10 == 0:
            print(f"   Step {i}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} Complete! Average Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# 5. Save the improved model
torch.save(model.state_dict(), "vit_model_balanced.pth")
print("Saved better model to: vit_model_balanced.pth")