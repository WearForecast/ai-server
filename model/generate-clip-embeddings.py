import csv
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def generate_clip_image_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings

def save_embedding_to_csv(image_folder, csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name"] + [f"dim_{i}" for i in range(512)])

        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg")):
                image_path = os.path.join(image_folder, filename)
                embedding = generate_clip_image_embeddings(image_path)
                image_name = os.path.splitext(filename)[0]
                writer.writerow([image_name] + embedding.squeeze().tolist())

image_folder = "./images"
csv_file = "image_embeddings.csv"
save_embedding_to_csv(image_folder, csv_file)
print("Image embeddings saved to CSV file.")