from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from data import OpenCriticDataClient
from ml import ScorePredictor, GameRecommender
from news import GamingNewsService

app = FastAPI(title="OpenCritic ML API")

print("Loading data and models")
client = OpenCriticDataClient()
client._load_database()
df = client.to_dataframe()

predictor = ScorePredictor()
predictor.fit(df)

recommender = GameRecommender()
recommender.fit(df)

news_service = GamingNewsService(enable_deduplication=True)

# Request & response models
class PredictRequest(BaseModel):
    reviews: int = 0
    percent_recommended: float = 0
    genre: str
    platform: str
    developer: str
    publisher: str
    
class RecommendRequest(BaseModel):
    title: str
    k: int = 5

# API Endpoints
@app.post("/predict")
def predict_score(game: PredictRequest):
    features = game.dict()
    score = predictor.predict_score(features)
    return {"predicted_score": score}

@app.post("/recommend")
def recommend_games(game: RecommendRequest):
    recs = recommender.recommend(req.title, req.k)
    return {"recommendations": recs}

@app.get("/news")
def get_news(per_source: int = 5, deduplicate: bool = True):
    items = news_service.fetch(per_source=per_source, deduplicate=deduplicate)
    return {"news": [item.to_dict() for item in items]}

@app.get("/game/{title}")
def game_details(title: str):
    matches = df[df['title'].str.lower() == title.lower()]
    
    if matches.empty:
        raise HTTPException(status_code=404, detail="Game not found.")
    row.