from __future__ import annotations
from typing import Optional

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List

import requests

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# For abstractive summarization
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    pipeline = None


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: str
    summary: str

    def to_dict(self) -> Dict:
        return asdict(self)


class GamingNewsService:
    """Collect and summarize gaming news headlines from public feeds."""

    FEEDS = [
        ("IGN", "https://feeds.ign.com/ign/games-all"),
        ("GameSpot", "https://www.gamespot.com/feeds/mashup/"),
        ("Polygon", "https://www.polygon.com/rss/index.xml"),
        ("PC Gamer", "https://www.pcgamer.com/rss/"),
        ("Rock Paper Shotgun", "https://www.rockpapershotgun.com/feed"),
        ("Eurogamer", "https://www.eurogamer.net/?format=rss"),
        ("Destructoid", "https://www.destructoid.com/feed/"),
        ("Gematsu", "https://www.gematsu.com/feed"),
        ("Nintendo Life", "https://www.nintendolife.com/feeds/latest"),
        ("This Week In Videogames", "https://thisweekinvideogames.com/feed/"),
    ]

    def __init__(self, enable_deduplication: bool = True, similarity_threshold: float = 0.85):
        self.enable_deduplication = enable_deduplication and _HAS_SENTENCE_TRANSFORMERS
        self.similarity_threshold = similarity_threshold
        self._embedder = None
        self.summarizer = None
        self.summary_cache = {}
        self._load_summarizer()

        if self.enable_deduplication:
            self._load_embedder()

    def _load_summarizer(self):
        if self.summarizer is None and _HAS_TRANSFORMERS:
            try:
                self.summarizer = pipeline("summarization", model="t5-small", device=-1)  # -1 = CPU
                print("Abstractive summarizer loaded.")
            except Exception as e:
                print(f"Summarizer not available: {e}. Falling back to extractive.")

    def _summarize_abstractive(self, text: str, max_length: int = 80) -> str:
        if not self.summarizer or len(text) < 100:
            return self._summarize_extractive(text)
        # Check cache
        if text in self.summary_cache:
            return self.summary_cache[text]
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=20, do_sample=False)[0]['summary_text']
            self.summary_cache[text] = summary
            return summary
        except Exception:
            return self._summarize_extractive(text)

    @staticmethod
    def _summarize_extractive(text: str, max_sentences: int = 2) -> str:
        cleaned = GamingNewsService._clean_text(text)
        chunks = re.split(r"(?<=[.!?])\s+", cleaned)
        summary = " ".join(chunks[:max_sentences]).strip()
        return summary if summary else "No summary available."

    def _load_embedder(self):
        if self._embedder is None and _HAS_SENTENCE_TRANSFORMERS:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        if not self.enable_deduplication or len(items) <= 1:
            return items

        texts = [f"{item.title}. {item.summary}" for item in items]
        embeddings = self._embedder.encode(texts, show_progress_bar=False)

        from sklearn.metrics.pairwise import cosine_similarity
        kept_indices = []
        for i in range(len(items)):
            keep_this = True
            for j in kept_indices:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > self.similarity_threshold:
                    keep_this = False
                    break
            if keep_this:
                kept_indices.append(i)

        return [items[i] for i in kept_indices]

    def fetch(self, per_source: int = 5, deduplicate: Optional[bool] = None) -> List[NewsItem]:
        if deduplicate is None:
            deduplicate = self.enable_deduplication

        items: List[NewsItem] = []
        for source, url in self.FEEDS:
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                candidates = root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry")
                count = 0
                for node in candidates:
                    if count >= per_source:
                        break
                    title = (node.findtext("title") or node.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
                    link = (node.findtext("link") or "").strip()
                    if not link:
                        link_el = node.find("{http://www.w3.org/2005/Atom}link")
                        if link_el is not None:
                            link = link_el.attrib.get("href", "")
                    published = (
                        node.findtext("pubDate")
                        or node.findtext("{http://www.w3.org/2005/Atom}updated")
                        or "Unknown"
                    )
                    description = (
                        node.findtext("description")
                        or node.findtext("content")
                        or node.findtext("{http://www.w3.org/2005/Atom}summary")
                        or ""
                    )
                    if not title:
                        continue

                    summary = self._summarize_abstractive(description)

                    items.append(
                        NewsItem(
                            title=title,
                            link=link,
                            source=source,
                            published=published,
                            summary=summary,
                        )
                    )
                    count += 1
            except Exception as e:
                print(f"Error fetching {source}: {e}")
                continue

        if deduplicate:
            original_count = len(items)
            items = self._deduplicate(items)
            if original_count != len(items):
                print(f"Deduplication removed {original_count - len(items)} duplicate stories.")

        return items