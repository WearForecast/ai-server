import torch
import os
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
# import google.generativeai as genai

api_key = ""

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

clothing_images_folder = "./images"
clothing_images = {
    os.path.splitext(filename)[0]: os.path.join(clothing_images_folder, filename)
    for filename in os.listdir(clothing_images_folder)
    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
}

class ClothingRecommender:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()

    # def generate_prompt(self, weather, category):
    #     genai.configure(api_key=api_key)
    #     model = genai.GenerativeModel("gemini-1.5-flash")
    #     prompt = model.generate(f"Recommend a {category} using the following weather data: " 
    #                             + weather 
    #                             + "Return only the names of the items and separate them using 'or'.")
    #     return prompt

    def recommend_clothing(self, prompt):
        images = [Image.open(path) for path in clothing_images.values()]
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        index = torch.argmax(logits_per_image, dim=0).item()
        recommended_item = list(clothing_images.keys())[index]

        return recommended_item