from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Union

from opencritic_app.data import OpenCriticDataClient
from opencritic_app.ml import GameRecommender, ScorePredictor
from opencritic_app.news import GamingNewsService


def _api_key_from_env() -> str:
    return (
        os.environ.get("OPENCRITIC_API_KEY", "")
        or os.environ.get("RAPIDAPI_KEY", "")
        or os.environ.get("RapidAPI_KEY", "")
    )


def save_json(path: str, payload: Union[Dict, List]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCritic ML application")
    parser.add_argument("--refresh", action="store_true", help="Fetch latest OpenCritic games")
    parser.add_argument(
        "--api-key",
        default=_api_key_from_env(),
        help="RapidAPI key (or set OPENCRITIC_API_KEY, RAPIDAPI_KEY, or RapidAPI_KEY in .env)",
    )
    parser.add_argument("--database", default="data/opencritic_database.json", help="Path to local game database")
    parser.add_argument("--pages", type=int, default=5, help="How many API pages to fetch")
    parser.add_argument(
        "--refresh-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API pages (reduces RapidAPI 429 rate limits)",
    )

    # NEW: sort order argument
    parser.add_argument(
        "--sort",
        default="date",
        choices=["date", "score", "num-reviews"],
        help="Sort order for fetching games: date (newest first), score (highest first), num-reviews (most reviews first)"
    )

    parser.add_argument("--train", action="store_true", help="Train OpenCritic score prediction model")
    parser.add_argument("--predict-title", default="", help="Game title (for display in prediction output)")
    parser.add_argument("--predict-genre", default="Action", help="Genre for prediction")
    parser.add_argument("--predict-platform", default="PC", help="Platform for prediction")
    parser.add_argument("--predict-developer", default="Unknown", help="Developer for prediction")
    parser.add_argument("--predict-publisher", default="Unknown", help="Publisher for prediction")
    parser.add_argument("--predict-reviews", type=int, default=25, help="Expected critic review count")
    parser.add_argument("--predict-recommended", type=float, default=75.0, help="Expected percent recommended")

    parser.add_argument("--recommend-for", default="", help="Get recommendations for a title")
    parser.add_argument("--recommend-k", type=int, default=5, help="Number of recommendations")

    parser.add_argument("--news", action="store_true", help="Fetch and summarize gaming news")
    parser.add_argument("--news-per-source", type=int, default=5, help="News items per source")
    parser.add_argument("--output", default="outputs/latest_results.json", help="Path to export run results JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results: Dict = {"steps": []}

    api_key = (args.api_key or "").strip() or None
    client = OpenCriticDataClient(database_file=args.database, api_key=api_key)

    if args.refresh:
        refresh_stats = client.refresh(
            pages=args.pages,
            page_delay_seconds=args.refresh_delay,
            sort_by=args.sort            # NEW: pass sort order
        )
        results["refresh"] = refresh_stats
        results["steps"].append("refresh")

    if args.news:
        news_service = GamingNewsService(enable_deduplication=True)
        news_items = news_service.fetch(per_source=args.news_per_source, deduplicate=True)
        results["news"] = [item.to_dict() for item in news_items]
        results["steps"].append("news")

    frame = client.to_dataframe()
    needs_games = bool(args.train or args.recommend_for)
    if frame.empty and needs_games:
        print("No data in database yet. Run with --refresh.")
        save_json(args.output, results)
        return

    if args.train:
        predictor = ScorePredictor()
        metrics = predictor.fit(frame)
        prediction = predictor.predict_score(
            {
                "reviews": args.predict_reviews,
                "percent_recommended": args.predict_recommended,
                "genre": args.predict_genre,
                "platform": args.predict_platform,
                "developer": args.predict_developer,
                "publisher": args.predict_publisher,
            }
        )
        results["training"] = {"mae": metrics.mae, "r2": metrics.r2, "samples": metrics.samples}
        results["prediction"] = {
            "title": args.predict_title or "Custom game",
            "predicted_opencritic_score": prediction,
        }
        results["steps"].append("train_predict")

    if args.recommend_for:
        rec = GameRecommender()
        rec.fit(frame)
        recs = rec.recommend(args.recommend_for, args.recommend_k)
        results["recommendations"] = {"for": args.recommend_for, "items": recs}
        results["steps"].append("recommend")

    save_json(args.output, results)
    print(f"Completed steps: {', '.join(results['steps']) if results['steps'] else 'none'}")
    print(f"Results saved to: {args.output}")
    if "prediction" in results:
        p = results["prediction"]
        print(f"Predicted score for {p['title']}: {p['predicted_opencritic_score']}")
    if "recommendations" in results:
        print(f"Recommendations generated: {len(results['recommendations']['items'])}")
    if "news" in results:
        print(f"News summaries collected: {len(results['news'])}")


if __name__ == "__main__":
    main()