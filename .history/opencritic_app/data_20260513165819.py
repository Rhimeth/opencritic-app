from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


def _parse_env_line(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if "=" not in s:
        return None
    key, rest = s.split("=", 1)
    key = key.strip()
    if not key:
        return None
    val = rest.strip()
    if val.startswith('"'):
        if len(val) >= 2 and val.endswith('"'):
            val = val[1:-1]
        else:
            val = val[1:]
    elif val.startswith("'"):
        if len(val) >= 2 and val.endswith("'"):
            val = val[1:-1]
        else:
            val = val[1:]
    return (key, val)


def _load_project_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-16")
        except OSError:
            return
    except OSError:
        return
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    for line in text.splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            k, v = parsed
            if v.strip() != "":
                os.environ.setdefault(k, v)


_load_project_dotenv()


@dataclass
class GameRecord:
    game_id: int
    title: str
    score: float
    reviews: int
    percent_recommended: float
    genre: str
    platform: str
    developer: str
    publisher: str
    release_date: str
    description: str
    scraped_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


class OpenCriticDataClient:
    RAPIDAPI_BASE = "https://opencritic-api.p.rapidapi.com/game"
    RAPIDAPI_HOST = "opencritic-api.p.rapidapi.com"

    def __init__(
        self,
        database_file: str = "data/opencritic_database.json",
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = self.RAPIDAPI_BASE
        key = (
            (api_key or "").strip()
            or os.environ.get("OPENCRITIC_API_KEY")
            or os.environ.get("RAPIDAPI_KEY")
            or os.environ.get("RapidAPI_KEY")
        )
        self.api_key = key.strip() if key else ""
        self.database_path = Path(database_file)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": "opencritic-ml-app/1.0",
            "Accept": "application/json",
            "X-RapidAPI-Host": self.RAPIDAPI_HOST,
        }
        if self.api_key:
            self.headers["X-RapidAPI-Key"] = self.api_key
        self.records: Dict[int, GameRecord] = self._load_database()

    def _load_database(self) -> Dict[int, GameRecord]:
        if not self.database_path.exists():
            return {}
        with self.database_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        loaded: Dict[int, GameRecord] = {}
        for row in rows:
            rec = GameRecord(**row)
            loaded[rec.game_id] = rec
        return loaded

    def _save_database(self) -> None:
        rows = [rec.to_dict() for rec in self.records.values()]
        with self.database_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _pick_score(payload: Dict) -> Optional[float]:
        score = payload.get("averageScore")
        if score in (None, -1):
            score = payload.get("topCriticScore")
        if score in (None, -1):
            return None
        return float(score)

    @staticmethod
    def _join_names(items: List[Dict], key: str = "name", limit: int = 3) -> str:
        names = [it.get(key, "").strip() for it in items if it.get(key)]
        return ", ".join(names[:limit]) if names else "Unknown"

    def _fetch_game_details(self, game_id: int) -> Optional[Dict]:
        """Fetch full game details including Companies from the individual endpoint."""
        url = f"{self.base_url}/{game_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Failed to fetch details for game {game_id}: {e}")
            return None

    def _parse_game(self, payload: Dict, is_full_detail: bool = False) -> Optional[GameRecord]:
        """Parse game data. If is_full_detail=True, we expect Companies field."""
        game_id = payload.get("id")
        if not game_id:
            return None
        score = self._pick_score(payload)
        if score is None:
            return None

        developer = "Unknown"
        publisher = "Unknown"
        if is_full_detail:
            companies = payload.get("Companies", [])
            for comp in companies:
                ctype = str(comp.get("type", "")).lower()
                name = comp.get("name", "Unknown")
                if "developer" in ctype and developer == "Unknown":
                    developer = name
                if "publisher" in ctype and publisher == "Unknown":
                    publisher = name
        else:
            developer = payload.get("developer") or payload.get("developers") or "Unknown"
            publisher = payload.get("publisher") or payload.get("publishers") or "Unknown"

        release = payload.get("firstReleaseDate")
        if isinstance(release, int):
            release = datetime.fromtimestamp(release / 1000).strftime("%Y-%m-%d")
        if not release:
            release = "Unknown"

        return GameRecord(
            game_id=int(game_id),
            title=payload.get("name", "Unknown"),
            score=score,
            reviews=int(payload.get("numReviews", 0) or 0),
            percent_recommended=float(payload.get("percentRecommended", 0) or 0),
            genre=self._join_names(payload.get("Genres", [])),
            platform=self._join_names(payload.get("Platforms", [])),
            developer=developer,
            publisher=publisher,
            release_date=release,
            description=(payload.get("description") or "No description.")[:800],
            scraped_at=datetime.now().isoformat(),
        )

    def refresh(
        self,
        pages: int = 5,
        page_size: int = 20,
        page_delay_seconds: float = 2.0,
        sort_by: str = "date",
        fetch_details: bool = True,
        detail_delay_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "OpenCritic data refresh requires a RapidAPI key. "
                "Subscribe at https://rapidapi.com/opencritic-opencritic-default/api/opencritic-api "
                "then set OPENCRITIC_API_KEY (or RAPIDAPI_KEY), or pass --api-key."
            )
        added = 0
        updated = 0
        for page in range(pages):
            if page > 0 and page_delay_seconds > 0:
                time.sleep(page_delay_seconds)
            params = {"skip": page * page_size, "limit": page_size, "sort": sort_by}
            payload = self._get_game_list_payload(params)
            games = payload if isinstance(payload, list) else payload.get("data", [])

            for game_payload in games:
                game_id = game_payload.get("id")
                if not game_id:
                    continue

                existing = self.records.get(game_id)
                if existing is None and fetch_details:
                    full_payload = self._fetch_game_details(game_id)
                    if full_payload:
                        record = self._parse_game(full_payload, is_full_detail=True)
                        if record:
                            self.records[record.game_id] = record
                            added += 1
                            if detail_delay_seconds > 0:
                                time.sleep(detail_delay_seconds)
                            continue
                    record = self._parse_game(game_payload, is_full_detail=False)
                    if record:
                        self.records[record.game_id] = record
                        added += 1
                else:
                    record = self._parse_game(game_payload, is_full_detail=False)
                    if record and (existing.score != record.score or existing.reviews != record.reviews):
                        if existing.developer != "Unknown":
                            record.developer = existing.developer
                        if existing.publisher != "Unknown":
                            record.publisher = existing.publisher
                        self.records[record.game_id] = record
                        updated += 1
                    elif not existing:
                        record = self._parse_game(game_payload, is_full_detail=False)
                        if record:
                            self.records[record.game_id] = record
                            added += 1

        self._save_database()
        return {"total": len(self.records), "added": added, "updated": updated}

    def _get_game_list_payload(self, params: Dict[str, Any]) -> Any:
        max_attempts = 8
        backoff = 2.0
        last: Optional[requests.Response] = None
        for attempt in range(max_attempts):
            last = requests.get(
                self.base_url, headers=self.headers, params=params, timeout=30
            )
            if last.status_code != 429:
                last.raise_for_status()
                return last.json()
            retry_after = last.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = min(60.0, backoff)
            else:
                wait = min(60.0, backoff)
            time.sleep(wait)
            backoff = min(60.0, backoff * 1.5)
        if last is not None:
            last.raise_for_status()
        raise RuntimeError("RapidAPI request failed after retries")

    def to_dataframe(self) -> pd.DataFrame:
        rows = [rec.to_dict() for rec in self.records.values()]
        return pd.DataFrame(rows)