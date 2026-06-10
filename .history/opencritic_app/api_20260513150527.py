from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from opencritic_app.data import OpenCriticDataClient
from opencritic_app.ml import ScorePredictor, GameRecommender
from opencritic_app.news import GamingNewsService

app = FastAPI(title="OpenCritic ML API")

print("Loading data and models")
client = OpenCriticDataClient()
client._load_database()
df = client.to_dataframe()

df['primary_genre'] = df['genre'].str.split(',').str[0]

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
    recs = recommender.recommend(game.title, game.k)
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
    row = matches.iloc[0].to_dict()
    row['description_summary'] = news_service._summarize(row['description'], max_sentences=2)
    return row

@app.get("/analytics/stats")
def get_analytics_stats(
    developer: Optional[str] = None,
    genre: Optional[str] = None,
    min_score: float = 0.0
):
    filtered = df.copy()

    if developer:
        filtered = filtered[filtered['developer'].str.contains(developer, case=False, na=False)]
    if genre:
        filtered = filtered[filtered['primary_genre'].str.contains(genre, case=False, na=False)]
    if min_score > 0:
        filtered = filtered[filtered['score'] >= min_score]

    if filtered.empty:
        return {
            "yearly_scores": {},
            "genre_scores": {},
            "developer_scores": {},
            "score_distribution": {},
            "platform_scores": {},
            "series_scores": {}
        }

    # 1. Scores by release year
    df_year = filtered.copy()
    df_year['release_year'] = pd.to_datetime(df_year['release_date'], errors='coerce').dt.year
    yearly_avg = df_year.groupby('release_year')['score'].mean().dropna().sort_index()

    # 2. Scores by genre
    genre_avg = filtered.groupby('primary_genre')['score'].mean().sort_values(ascending=False)

    # 3. Top developers (min 3 games)
    dev_counts = filtered['developer'].value_counts()
    top_devs = dev_counts[dev_counts >= 3].index[:10]
    dev_avg = filtered[filtered['developer'].isin(top_devs)].groupby('developer')['score'].mean().sort_values(ascending=False)

    # 4. Score distribution
    score_bins = [0, 50, 60, 70, 80, 90, 101]
    score_labels = ['0-49', '50-59', '60-69', '70-79', '80-89', '90-100']
    filtered['score_bin'] = pd.cut(filtered['score'], bins=score_bins, labels=score_labels, right=False)
    score_dist = filtered['score_bin'].value_counts().sort_index()

    # 5. Platform performance (top 8)
    platform_avg = filtered.groupby('platform')['score'].mean().sort_values(ascending=False).head(8)

    # 6. Series / franchises
    def extract_series(title):
        words = title.split()
        if len(words) >= 2 and words[0].lower() in ['super', 'the', 'call', 'final', 'grand', 'resident', 'street', 'mortal', 'assassin', 'god', 'metal', 'borderlands']:
            return f"{words[0]} {words[1]}"
        return words[0] if words else title

    filtered['series'] = filtered['title'].apply(extract_series)
    series_counts = filtered['series'].value_counts()
    top_series = series_counts[series_counts >= 3].index[:8]
    series_avg = filtered[filtered['series'].isin(top_series)].groupby('series')['score'].mean().sort_values(ascending=False)

    return {
        "yearly_scores": {int(year): round(score, 1) for year, score in yearly_avg.items()},
        "genre_scores": {genre: round(score, 1) for genre, score in genre_avg.items()},
        "developer_scores": {dev: round(score, 1) for dev, score in dev_avg.items()},
        "score_distribution": {label: int(count) for label, count in score_dist.items()},
        "platform_scores": {platform: round(score, 1) for platform, score in platform_avg.items()},
        "series_scores": {series: round(score, 1) for series, score in series_avg.items()}
    }

@app.get("/games")
def list_games(limit: int = 100):
    return df.head(limit)[['title', 'score', 'genre', 'platform', 'developer']].to_dict(orient='records')

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    with open("static/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

app.mount("/", StaticFiles(directory="static", html=True), name="static")