from __future__ import annotations
from typing import Optional

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainingMetrics:
    mae: float
    r2: float
    samples: int


class ScorePredictor:
    """Predict OpenCritic scores from structured game metadata."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.metrics: TrainingMetrics | None = None

    @staticmethod
    def _feature_columns() -> Dict[str, List[str]]:
        return {
            "numeric": ["reviews", "percent_recommended"],
            "categorical": ["genre", "platform", "developer", "publisher"],
        }

    def fit(self, df: pd.DataFrame) -> TrainingMetrics:
        if len(df) < 30:
            raise ValueError("Need at least 30 games for reliable score prediction training.")
        train_df = df.dropna(subset=["score"]).copy()

        cols = self._feature_columns()
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), cols["numeric"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cols["categorical"]),
            ]
        )
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        
        self.pipeline = Pipeline([("prep", pre), ("rf", model)])

        x = train_df[cols["numeric"] + cols["categorical"]]
        y = train_df["score"].astype(float)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.pipeline.fit(x_train, y_train)
        preds = self.pipeline.predict(x_test)
        self.metrics = TrainingMetrics(
            mae=float(mean_absolute_error(y_test, preds)),
            r2=float(r2_score(y_test, preds)),
            samples=len(train_df),
        )
        return self.metrics

    def predict_score(self, game_features: Dict) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        row = pd.DataFrame([game_features])
        pred = float(self.pipeline.predict(row)[0])
        return round(np.clip(pred, 0, 100), 2)


class GameRecommender:
    """Content-based recommender using nearest neighbors over metadata."""

    def __init__(self) -> None:
        self.frame: pd.DataFrame | None = None
        self.vectorizer: ColumnTransformer | None = None
        self.nn_model: NearestNeighbors | None = None

    def fit(self, df: pd.DataFrame) -> None:
        if len(df) < 5:
            raise ValueError("Need at least 5 games for recommendations.")
        self.frame = df.reset_index(drop=True).copy()

        features = ["genre", "platform", "developer", "publisher", "percent_recommended", "score"]
        self.frame[features] = self.frame[features].fillna("Unknown")
        self.frame["percent_recommended"] = pd.to_numeric(self.frame["percent_recommended"], errors="coerce").fillna(0)
        self.frame["score"] = pd.to_numeric(self.frame["score"], errors="coerce").fillna(self.frame["score"].median())

        self.vectorizer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["genre", "platform", "developer", "publisher"]),
                ("num", StandardScaler(), ["percent_recommended", "score"]),
            ]
        )
        matrix = self.vectorizer.fit_transform(self.frame)

        self.nn_model = NearestNeighbors(metric="cosine", n_neighbors=min(15, len(self.frame)))
        self.nn_model.fit(matrix)

    def recommend(self, title: str, k: int = 5) -> List[Dict]:
        if self.frame is None or self.vectorizer is None or self.nn_model is None:
            raise RuntimeError("Recommender is not trained. Call fit() first.")
        matches = self.frame[self.frame["title"].str.lower() == title.lower()]
        if matches.empty:
            top = self.frame.sort_values(["score", "percent_recommended"], ascending=False).head(k)
            return top[["title", "score", "genre", "platform"]].to_dict(orient="records")

        idx = int(matches.index[0])
        matrix = self.vectorizer.transform(self.frame)
        distances, indices = self.nn_model.kneighbors(matrix[idx], n_neighbors=min(k + 1, len(self.frame)))

        results: List[Dict] = []
        for dist, rec_idx in zip(distances[0], indices[0]):
            if rec_idx == idx:
                continue
            row = self.frame.iloc[int(rec_idx)]
            results.append(
                {
                    "title": row["title"],
                    "score": float(row["score"]),
                    "genre": row["genre"],
                    "platform": row["platform"],
                    "similarity": round(1 - float(dist), 3),
                }
            )
            if len(results) >= k:
                break
        return results

