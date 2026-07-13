"""Builds the chatbot's knowledge base from Navimind POIs (passed in by the
backend) instead of the Adham `categories.json` / `sort_data.json` fixtures.

Each POI already carries `name, code, type, floorLevel, category, description,
aliases[], productKeywords[]`, so a POI doubles as the "store" the chatbot
resolves to. We build, per building and cached by a content hash:

  * a deterministic alias/keyword index (fast-path, no embeddings/LLM), and
  * a sentence-transformers embedding index for semantic RAG search.

Resolution returns a POI dict directly (with id + floorLevel), so the caller can
emit a Navimind navigation action without any room-number mapping.
"""
from __future__ import annotations

import hashlib
import os
import threading

# Force transformers to ignore any globally-installed TensorFlow/Keras (this
# machine has TF for the positioning models; Keras 3 breaks transformers' TF
# path). We only need the PyTorch backend for the embedder.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np

from text_utils import norm, phrase_matches, tokenize, singular

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedder = None
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                print(f"[catalog] loading embedder {EMBED_MODEL_NAME} ...")
                _embedder = SentenceTransformer(EMBED_MODEL_NAME)
                print("[catalog] embedder ready.")
    return _embedder


def _poi_categories(poi: dict) -> list[str]:
    """All category + sub-category names linked to a POI (many-to-many)."""
    cats = poi.get("categories")
    if cats:
        return [str(c) for c in cats if c]
    one = _poi_category(poi)
    return [one] if one else []


def _poi_category(poi: dict) -> str:
    """A single representative category name (for display)."""
    cat = poi.get("category")
    if isinstance(cat, dict):
        return cat.get("name") or ""
    if cat:
        return str(cat)
    cats = poi.get("categories") or []
    return str(cats[0]) if cats else ""


def _rag_text(poi: dict) -> str:
    parts = [
        poi.get("name") or "",
        " ".join(_poi_categories(poi)),
        poi.get("type") or "",
        " ".join(poi.get("aliases") or []),
        " ".join(poi.get("productKeywords") or []),
        poi.get("description") or "",
    ]
    return " - ".join(p for p in parts if p).strip()


class BuildingCatalog:
    def __init__(self, pois: list[dict]):
        # Only active, resolvable POIs participate in retrieval.
        self.pois = [p for p in pois if p.get("active", True) and p.get("id")]
        self.by_id = {p["id"]: p for p in self.pois}

        # Deterministic indexes -------------------------------------------------
        # keyword token -> {poi_id: weight}; full phrase list checked as substrings.
        self.keyword_index: dict[str, dict[str, float]] = {}
        self.full_names: list[tuple[str, str]] = []      # (lowercased phrase, poi_id)
        self.arabic_aliases: list[tuple[str, str]] = []  # (normalized alias, poi_id)

        for p in self.pois:
            pid = p["id"]
            names = [p.get("name") or "", *_poi_categories(p)]
            names += list(p.get("aliases") or [])
            names += list(p.get("productKeywords") or [])
            for raw in names:
                if not raw:
                    continue
                low = raw.lower()
                if len(low) >= 4:
                    self.full_names.append((low, pid))
                # Arabic aliases matched with normalization + whole-word logic.
                if any("؀" <= c <= "ۿ" for c in raw):
                    self.arabic_aliases.append((norm(raw), pid))
                for tok in tokenize(raw):
                    self.keyword_index.setdefault(singular(tok), {}).setdefault(pid, 0.0)
                    self.keyword_index[singular(tok)][pid] += 1.0
        # longest / most specific phrases first
        self.full_names.sort(key=lambda kv: -len(kv[0]))

        # Lazy embedding index --------------------------------------------------
        self._emb = None
        self._emb_lock = threading.Lock()

    # -- deterministic fast-path ------------------------------------------------
    def alias_direct_match(self, text: str) -> dict | None:
        if not text:
            return None
        low = norm(text)
        for alias_norm, pid in self.arabic_aliases:
            if phrase_matches(alias_norm, low):
                return self.by_id.get(pid)

        low_en = str(text).lower()
        for phrase, pid in self.full_names:
            if phrase in low_en:
                return self.by_id.get(pid)

        scores: dict[str, float] = {}
        for tok in tokenize(text):
            tok = singular(tok)
            for pid, weight in self.keyword_index.get(tok, {}).items():
                scores[pid] = scores.get(pid, 0.0) + weight
        if scores:
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            return self.by_id.get(best)
        return None

    # -- semantic RAG -----------------------------------------------------------
    def _ensure_embeddings(self):
        if self._emb is not None or not self.pois:
            return
        with self._emb_lock:
            if self._emb is None:
                model = get_embedder()
                texts = [_rag_text(p) for p in self.pois]
                self._emb = model.encode(
                    texts, convert_to_numpy=True, normalize_embeddings=True
                )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.35):
        if not query or not self.pois:
            return []
        self._ensure_embeddings()
        model = get_embedder()
        q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self._emb @ q  # cosine (both normalized)
        order = np.argsort(-sims)[:top_k]
        out = []
        for idx in order:
            score = float(sims[int(idx)])
            if score >= min_score:
                out.append((self.pois[int(idx)], round(score, 3)))
        return out

    def category_siblings(self, poi: dict) -> list[dict]:
        """POIs that share at least one category with this POI."""
        cats = set(_poi_categories(poi))
        if not cats:
            return []
        return [p for p in self.pois if cats & set(_poi_categories(p))]

    def warm(self):
        """Pre-build embeddings so the first real request isn't slow."""
        self._ensure_embeddings()


# --- per-building cache (rebuilt only when the POI version changes) -----------
_cache: dict[str, tuple[str, BuildingCatalog]] = {}
_cache_lock = threading.Lock()


def get_catalog_by_version(building_id: str, version: str) -> BuildingCatalog | None:
    with _cache_lock:
        cached = _cache.get(building_id)
        if cached and cached[0] == version:
            return cached[1]
    return None


def build_and_cache_catalog(building_id: str, version: str, pois: list[dict]) -> BuildingCatalog:
    catalog = BuildingCatalog(pois)
    with _cache_lock:
        _cache[building_id] = (version, catalog)
    return catalog
