from __future__ import annotations

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
    """Collect and summarize game industry headlines from public RSS feeds."""

    FEEDS = [
        ("IGN", "https://feeds.ign.com/ign/games-all"),
        ("GameSpot", "https://www.gamespot.com/feeds/mashup/"),
        ("Polygon", "https://www.polygon.com/rss/index.xml"),
    ]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _summarize(text: str, max_sentences: int = 2) -> str:
        cleaned = GamingNewsService._clean_text(text)
        chunks = re.split(r"(?<=[.!?])\s+", cleaned)
        summary = " ".join(chunks[:max_sentences]).strip()
        return summary if summary else "No summary available."

    def fetch(self, per_source: int = 5) -> List[NewsItem]:
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
                    items.append(
                        NewsItem(
                            title=title,
                            link=link,
                            source=source,
                            published=published,
                            summary=self._summarize(description),
                        )
                    )
                    count += 1
            except Exception:
                continue
        return items

