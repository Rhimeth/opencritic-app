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
    """Collect and summarize gaming news headlines from public feeds."""

    FEEDS = [
        ("IGN", "https://feeds.ign.com/ign/games-all"),
        ("GameSpot", "https://www.gamespot.com/feeds/mashup/"),
        ("Polygon", "https://www.polygon.com/rss/index.xml"),
    ]
    
    def __init__(self, enable_deduplication: bool = True, similarity_threshold: float = 0.85):
        self.enable_deduplication = enable_deduplication and _HAS_SENTENCE_TRANSFORMERS
        self.similarity_threshold = similarity_threshold
        self._embedder = None
        
        if self.enable_deduplication:
            self._load_embedder()
            
    def _load_embedder(self):
        if self._embedder is None and _HAS_SENTENCE_TRANSFORMERS:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

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
    
    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate stories based on similarity"""
        
        if not self.enable_deduplication or len(items) <= 1:
            return items
        
        # Text for each item
        texts = [f"{item.title}. {item.summary}" for item in items]
        
        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        
        # Compare pairwise and build clusters
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        kept_indices = []
        
        for i in range(len(items)):
            if i in kept_indices:
                continue
            keep_this = True
            
            for j in kept_indices:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                if sim > self.similarity_threshold:
                    keep_this = False
                    break
            
            if keep_this:
                kept_indices.append(i)
        
        # Return kept items
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
            
        # Deduplicate applied if requested
        if deduplicate:
            original_count = len(items)
            items = self._deduplicate(items)
            
            if original_count != len(items):
                print(f"Deduplication removed {original_count - len(items)} deduplicated stories.")
                
        return items

