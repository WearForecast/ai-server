from fastapi import FastAPI
from model.model import ClothingRecommender

app = FastAPI()
model = ClothingRecommender()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/recommend")
async def recommend_clothing(prompt: str):
    recommended_item = model.recommend_clothing(prompt)
    return {"recommended_item": recommended_item}