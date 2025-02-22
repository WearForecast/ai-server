import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import wandb

from transformers import CLIPProcessor, CLIPModel

class KoreanFashionDataset(Dataset):
    def __init__(self, csv_file, images_dir):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            images_dir (string): Directory with all the images.
        """
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row['image_name'])
        image = Image.open(image_path).convert("RGB")
        caption = row['caption']
        return image, caption
    
# Collate function that uses CLIP processor to preprocess batches of images and texts
def collate_fn(batch):
    images, texts = zip(*batch)
    # Use processor to batch-process images and texts
    inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True)
    return inputs

# Setup: device, model, processor, dataloader
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model_name = "./mlx_model"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.to(device)
model.train()

csv_file = "./kfashion-dataset.csv"
images_dir = "./images"

batch_size = 32
learning_rate = 5e-6
num_epochs = 10

dataset = KoreanFashionDataset(csv_file, images_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize wandb
wandb.init(project="kfashion-clip", entity="kevvnbk-1")
wandb_config = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs
}
wandb.config.update(wandb_config)

for epoch in range(num_epochs):
    total_loss = 0.0
    # Dataloader Iteration, Move batch to device if necessary
    for batch in dataloader:
        for key in batch:
            batch[key] = batch[key].to(device)
        
        # Zeroing Gradients (clears old gradients from the previous step)
        optimizer.zero_grad()
        # Forward Pass (Preprocessed batch passed to CLIP)
        # Outputs: logits_per_image (similarity score between each image and every text in batch)
        # logits_per_text (simialrity score between each text and every image in batch)
        outputs = model(**batch)

        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        # Create tensor of labels [0, 1, 2, ..., batch_size-1] 
        # Since each image should match with corresponding caption at the same index in the batch
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size).to(device)

        # Symmetric Cross-Entropy Loss
        # Compute loss using logits_per_image/logits_per_text against labels
        # Average the 2 losses to ensure model learns to align both image-to-text and text-to-image
        loss_img = nn.functional.cross_entropy(logits_per_image, labels)
        loss_text = nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_text) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

# Save the model
output_dir = "./kfashion-clip"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

wandb.finish()