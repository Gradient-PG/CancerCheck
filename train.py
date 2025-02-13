import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import os

import warnings

warnings.filterwarnings("ignore")

from utils import CustomImageMaskDataset, bce_dice_loss, plot_metrics
from model import Unet
from utils import Trainer

# Load dataset

masks = glob.glob("data/Dataset_BUSI_with_GT/*/*_mask.png")
images = [mask_images.replace("_mask", "") for mask_images in masks]
series = list(zip(images, masks))

dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])

# Train/test split
train, test = train_test_split(dataset, test_size=0.2)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Define transformations
image_size = 128
transform = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = CustomImageMaskDataset(train, transform)
test_dataset = CustomImageMaskDataset(test, transform)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Select device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, loss function
unet = Unet(1).to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001, weight_decay=1e-6)

# Train model
trainer = Trainer(model=unet, num_epochs=10, optimizer=optimizer, criterion=bce_dice_loss, device=device)
trainer.train(train_dataloader, test_dataloader)

# Save metrics
metrics = trainer.get_metrics()
plot_metrics(metrics)

print("Training complete. Best model saved in 'log/' directory.")
