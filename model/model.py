import torch
import os
import requests
from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("API_KEY")
model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

class ClothingRecommender:
    def __init__(self):
        genai.configure(api_key=API_KEY)
        self.genai_model = genai.GenerativeModel("gemini-1.5-flash")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def fetch_images_from_server(self):

        return clothing_images

    def generate_prompt(self, weather, gender):
        if not weather or not isinstance(weather, str):
            raise ValueError("Invalid 'weather' input. It must be a non-empty string.")
        if not gender:
            raise ValueError("Invalid 'gender' input. Must be 'men', 'women', or 'neutral'.")

        prompt = self.genai_model.generate_content(f"""
                                You are an AI fashion consultant. Your job is to produce one short text description of an outfit for a user, 
                                based on provided weather details and user preferences. The text should be concise and must include:
                                           
		                        1.	A mention of the season (e.g., “spring,” “summer,” “fall,” or “winter”).
	                            2.	A mention of the gender (e.g., “women’s,” “men’s,” or neutral if appropriate).
	                            3.	Clothing items (e.g., shirts, pants, jackets) and relevant attributes (style, color, fabric, etc.).
	                            4.	Reference to the weather conditions provided (temperature, rain, wind, etc.).
                                
                                Weather: {weather}
                                Gender: {gender}

                                Do not provide additional commentary, disclaimers, or multiple paragraphs. 
                                Do not provide suggestions for accessories, shoes, or other items not explicitly mentioned in the prompt.
                                Output only the single descriptive sentence or brief paragraph that CLIP will use to match images.    

                                This is an example of a good prompt: 
                                "A women’s spring outfit featuring a lightweight pastel cardigan over a breathable cotton T-shirt, 
                                and paired with slim-fit jeans, perfect for mild 60°F weather with a light breeze."
                                """)
        return prompt.text

    def recommend_clothing(self, weather, gender):
        images = [Image.open(path) for path in clothing_images.values()]
        prompt = self.generate_prompt(weather, gender)

        with torch.no_grad():
            inputs = self.processor(
                text = [prompt],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            index = torch.argmax(logits_per_image).item()
            recommended_item = list(clothing_images.keys())[index]

        return recommended_item