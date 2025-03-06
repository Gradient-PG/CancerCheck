import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import warnings
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Load dataset
masks = glob.glob("Dataset_BUSI_with_GT/*/*_mask.png")
if len(masks) == 0:
    raise ValueError("No mask images found! Check dataset path.")

images = [mask_images.replace("_mask", "") for mask_images in masks]
series = list(zip(images, masks))

dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])


print(f"Number of samples found: {len(dataset)}")

# Train/test split
train, test = train_test_split(dataset, test_size=0.2)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Define transformations
image_size = 128
transform = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
])

# Custom dataset class
class CustomImageMaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        mask_path = self.dataframe.iloc[idx, 1]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Create datasets and dataloaders
train_dataset = CustomImageMaskDataset(train, transform)
test_dataset = CustomImageMaskDataset(test, transform)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
image_processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance",
    use_fast=True
)

model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

# Additional metric tracking
train_dice_scores = []
val_dice_scores = []

# Dice coefficient function
def dice_coefficient(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-6)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_dice = 0
    for images, masks in train_dataloader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(pixel_values=images)
        masks_pred = outputs.class_queries_logits
        masks_pred = masks_pred.permute(0, 2, 1).contiguous().view(masks.size(0), -1)
        masks = masks.view(masks.size(0), -1)

        if masks_pred.size(1) != masks.size(1):
            masks = masks[:, :masks_pred.size(1)]

        loss = criterion(masks_pred, masks.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_coefficient(masks_pred, masks).item()

    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_dice = train_dice / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_dice_scores.append(avg_train_dice)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Train Dice: {avg_train_dice}")

    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for images, masks in test_dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images)
            masks_pred = outputs.class_queries_logits
            masks_pred = masks_pred.permute(0, 2, 1).contiguous().view(masks.size(0), -1)
            masks = masks.view(masks.size(0), -1)

            if masks_pred.size(1) != masks.size(1):
                masks = masks[:, :masks_pred.size(1)]

            loss = criterion(masks_pred, masks.float())
            val_loss += loss.item()
            val_dice += dice_coefficient(masks_pred, masks).item()

    avg_val_loss = val_loss / len(test_dataloader)
    avg_val_dice = val_dice / len(test_dataloader)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)

    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Dice: {avg_val_dice}")

# Plotting the graphs
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(14, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", color="blue")
plt.plot(epochs, val_losses, label="Val Loss", color="orange")
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Dice Coefficient plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_dice_scores, label="Train Dice", color="blue")
plt.plot(epochs, val_dice_scores, label="Val Dice", color="orange")
plt.title("Training and Validation Dice Coefficients")
plt.xlabel("Epoch")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('data.png')
plt.show()

print("Training complete. Graphs saved as 'data.png'.")
