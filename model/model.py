import torch
import os
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai

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

    def generate_prompt(self, weather, categories) -> list:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = model.generate(f"""
                                Please recommend an outfit for me based on the following weather conditions
                                and available clothes in my wardrobe: {weather}, {categories}. 
                                Ensure the recommendations are practical, comfortable, and stylish. 
                                Do not include any clothing that is not in my wardrobe.
                                If the weather includes extreme conditions (e.g., rain, snow, strong wind, high humidity), 
                                include appropriate protective clothing or accessories.
            
                                Return the names of the clothing in a python list format. 
                                Do not include any other information and only return the list.
                                """)
        return prompt

    def recommend_clothing(self, weather, categories) -> list:
        images = [Image.open(path) for path in clothing_images.values()]
        prompt = self.generate_prompt(weather, categories)
        recommended_outfit = []

        for item in prompt:
            inputs = self.processor(
                text = [item],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image
            index = torch.argmax(logits_per_image, dim=0).item()
            recommended_outfit.append(list(clothing_images.keys())[index])

        return recommended_outfit