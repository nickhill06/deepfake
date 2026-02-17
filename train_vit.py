import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from dataset_loader import DeepfakeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DeepfakeDataset("dataset/train_faces")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch} Loss: {total_loss}")

torch.save(model.state_dict(), "vit_model.pth")
