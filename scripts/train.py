import torch
import torchvision
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

# Dataset path
DATA_DIR = "data/"

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dummy dataset class (replace with actual loader)
class FODDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

# Dummy data for example
images = ["data/images/img1.jpg", "data/images/img2.jpg"]
labels = [0, 1]  # 0: Metal, 1: Plastic, 2: Concrete

dataset = FODDataset(images, labels, transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 3)  # 3 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")