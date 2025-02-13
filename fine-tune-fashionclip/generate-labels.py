from PIL import Image
import csv
import os 
import dotenv
import time
import google.generativeai as genai

from natsort import natsorted

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.  Please set it in your .env file or environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_description(image_path):
    try:
        img = Image.open(image_path) 
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    try:
        prompt="""
        You are a fashion expert. When given an image of a model wearing an outfit, please generate a detailed caption that adheres to the following format:
        ‘A [gender] [season] outfit featuring [top details] and [bottom details], perfect for [weather/temperature conditions].’
        Ensure you include the gender (e.g., women’s, men’s), the season (e.g., spring, summer, fall, winter), key clothing items, and any relevant weather cues.\
        Do not mention accessories except scarves, hats and shoes.

        For example, if shown an image of a model in a light pastel cardigan with a cotton T-shirt and jeans, you might say:
        ‘A women’s spring outfit featuring a lightweight pastel cardigan over a breathable cotton T-shirt, paired with slim-fit jeans, perfect for mild 20°C weather with a light breeze.’
        """
        response = model.generate_content([prompt, img])
        response.resolve() 
        return response.text
    except Exception as e:
        print(f"Error generating description: {e}")
        print(f"Error details: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'No feedback available'}") # Added error handling
        return None

def process_dataset(image_directory, output_file_path):
    file_exists = os.path.isfile(output_file_path)

    with open(output_file_path, mode="a", newline="") as csvfile:
        fieldnames = ["filename", "description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        count = 0
        for filename in natsorted(os.listdir(image_directory)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, filename)

                description = generate_description(image_path)

                writer.writerow({"filename": filename, "description": description})
                count += 1
                print(f"Processed image {count}: {filename}")

                time.sleep(3)

if __name__ == "__main__":
    image_directory = "./images/1"  
    csv_file_path = "./image_labels.csv"

    process_dataset(image_directory, csv_file_path)

    print(f"Image descriptions saved to: {csv_file_path}")