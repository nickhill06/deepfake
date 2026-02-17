from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []
        
        for label, folder in enumerate(["real", "fake"]):
            path = os.path.join(root_dir, folder)
            for img in os.listdir(path):
                self.images.append(os.path.join(path, img))
                self.labels.append(label)
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
