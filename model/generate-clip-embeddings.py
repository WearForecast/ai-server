import torch
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# # Initialize Supabase client
# url = SUPABASE_URL
# key = SUPABASE_KEY
# supabase = create_client(url, key)

def is_image_file(file_path):
    try:
        Image.open(file_path).verify()  # Verify if the file is a valid image
        return True
    except (IOError, SyntaxError):
        return False

def generate_clip_image_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings

image_folder = "./images"
image_paths = [
    os.path.join(image_folder, image)
    for image in os.listdir(image_folder)
    if is_image_file(os.path.join(image_folder, image))
]

csv_file_path = "./image_embeddings.csv"

with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "image_name", "embedding"])
    id = 0

    for image_path in image_paths:
        embedding = generate_clip_image_embeddings(image_path)
        id += 1
        image_name = os.path.basename(image_path)
        embedding_list = embedding.cpu().numpy().flatten().tolist()

        writer.writerow([id, image_name, embedding_list])
        print(f"Saved embedding for {image_path}")

print(f"All image embeddings saved to {csv_file_path}")