# OpenCritic ML App

Python application that:

- ingests OpenCritic game data incrementally,
- trains a machine learning model to predict OpenCritic scores,
- gathers gaming news and builds short summaries,
- recommends similar games to users.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## OpenCritic API key (required for `--refresh`)

OpenCritic’s HTTP API is hosted on RapidAPI and returns **400** without a subscription key. Subscribe at [OpenCritic API on RapidAPI](https://rapidapi.com/opencritic-opencritic-default/api/opencritic-api), then either:

- create a `.env` file in the project root (see `.env.example`) with `OPENCRITIC_API_KEY`, `RAPIDAPI_KEY`, or `RapidAPI_KEY` — the app loads this on startup (tolerant parser: UTF-8 BOM, optional quotes, and a single opening `"` without a closing quote still work), or
- set the same variable names in your shell environment, or
- pass `--api-key YOUR_KEY` on the command line.

News (`--news`) does not need this key.

If `--refresh` returns **429 Too Many Requests**, RapidAPI is rate-limiting you: use a smaller `--pages` value or increase `--refresh-delay` (default **2** seconds between pages; retries with backoff are applied automatically).

## Example usage

Refresh data + train model + predict + recommend + collect news (replace `%OPENCRITIC_API_KEY%` on Windows or use env):

```bash
python app.py --refresh --pages 8 --train --predict-title "My New RPG" --predict-genre "RPG" --predict-platform "PC" --predict-developer "Indie Studio" --predict-publisher "Self Published" --predict-reviews 30 --predict-recommended 82 --recommend-for "Elden Ring" --recommend-k 6 --news --news-per-source 4 --output outputs/run.json
```

News-only:

```bash
python app.py --news --output outputs/news.json
```

Recommendations-only (after data already exists):

```bash
python app.py --recommend-for "The Witcher 3: Wild Hunt" --recommend-k 5
```

## Output

Every run exports a JSON payload (default `outputs/latest_results.json`) with:

- `refresh`: update stats
- `training`: model metrics (MAE, R2, sample count)
- `prediction`: predicted OpenCritic score
- `recommendations`: recommended games
- `news`: summarized gaming headlines

