import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import glob
import random
import os

from utils import CustomImageMaskDataset, plot_subplots
from model import Unet

# Set paths
log_directory = "log"
dataset_path = "data/Dataset_BUSI_with_GT/*/*_mask.png"

# Find best model dynamically
best_model_files = glob.glob(f"{log_directory}/best_model_dice_*.pth")
if not best_model_files:
    raise FileNotFoundError("No best model found! Check if training has completed successfully.")

best_model_path = sorted(best_model_files)[-1]  # Get latest best model
print(f"Loading best model: {best_model_path}")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(1).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
model.eval()

# Load dataset
masks = glob.glob(dataset_path)
if len(masks) == 0:
    raise ValueError("No mask images found! Check dataset path.")

images = [mask.replace("_mask", "") for mask in masks]
dataset = pd.DataFrame(list(zip(images, masks)), columns=['image_path', 'mask_path'])

# Define transformations
image_size = 128
transform = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
])

# Create dataset & dataloader
test_dataset = CustomImageMaskDataset(dataset, transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Select 5 random test samples
random_indices = random.sample(range(len(test_dataset)), 5)

# Plot predictions
for idx in random_indices:
    image, mask = test_dataset[idx]
    image = image.to(device).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        pred = model(image).squeeze()

    plot_subplots(image.squeeze(), mask, pred)

print("Results displayed successfully!")
