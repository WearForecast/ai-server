from fastapi import FastAPI
from model.model import ClothingRecommender

app = FastAPI()
model = ClothingRecommender()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/recommend")
async def recommend_clothing(weather: str, gender: str):
    recommended_clothings, translated_prompt = model.recommend_clothing(weather, gender)
    return recommended_clothings, translated_prompt