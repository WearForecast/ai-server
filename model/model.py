import torch
import os
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai

from supabase import create_client

load_dotenv()

API_KEY = os.getenv("API_KEY")
model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class ClothingRecommender:
    def __init__(self):
        genai.configure(api_key=API_KEY)
        self.genai_model = genai.GenerativeModel("gemini-1.5-flash")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def generate_prompt(self, weather, gender):
        if not weather or not isinstance(weather, str):
            raise ValueError("Invalid 'weather' input. It must be a non-empty string.")
        if not gender:
            raise ValueError("Invalid 'gender' input. Must be 'men', 'women', or 'neutral'.")

        prompt = self.genai_model.generate_content(f"""
                                You are an AI fashion consultant. Your job is to produce one short text description of an outfit for a user, 
                                based on the current weather conditions and weather forecast of the day.
                                The text should be concise and must include:
                                           
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
                                and paired with slim-fit jeans, perfect for mild 20°C weather with a light breeze."
                                """)
        return prompt.text
    
    def translate_prompt(self, prompt):
        translation = self.genai_model.generate_content(f"""
                                                        Translate the following English text into Korean:
                                                        {prompt}

                                                        Remove any information about colors and gender.
                                                        The translation should sound natural and fluent in Korean. 
                                                        First mention the weather, then the clothing items.
                                                        Do not include specific temperatures or other numerical values.
                                                        Only return the Korean translation of the text.
                                                        """)
        return translation.text

    def find_best_match(self, text_embedding):
        text_embedding_list = text_embedding.squeeze().tolist()

        response = supabase.rpc(
            "find_best_match",
            {"text_embedding": text_embedding_list}
        ).execute()

        if hasattr(response, "error") and response.error:  # Check for an error attribute
            raise Exception(f"Supabase Error: {response.error}")

        best_match = response.data
        if not best_match:
            print("No matches found in the database.")
            raise ValueError("No matches found in the database.")
        
        image_names = [row["image_name"] for row in best_match]

        return image_names

    def recommend_clothing(self, weather, gender):
        prompt = self.generate_prompt(weather, gender)
        translated_prompt = self.translate_prompt(prompt)

        with torch.no_grad():
            inputs = self.processor(
                text = [prompt],
                return_tensors="pt",
                padding=True,
            )
            text_embedding = self.model.get_text_features(**inputs)

        image_names = self.find_best_match(text_embedding)

        response = supabase.storage.from_("photos").create_signed_urls(
            image_names, 3600
        )
        if hasattr(response, "error") and response.error:  # Check for an error attribute
            raise Exception(f"Supabase Error: {response.error}")
        
        bucket_response = response
        if not bucket_response:
            print("No matches found in the database.")
            raise ValueError("No matches found in the database.")
        
        best_match_image_urls = [row["signedURL"] for row in bucket_response]

        return best_match_image_urls, translated_prompt