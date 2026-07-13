"""Product catalog + recommendations for the chatbot service.

Products live in the Navimind backend (seeded from the AI engineer's
sort_data.json). This module fetches them per building from the backend's
`/api/client/products` endpoint and caches them keyed by the building's product
version (Building.productUpdatedAt), mirroring how catalog.py caches POIs.

The recommendation ranking (top-N by rating + promo discounts) is ported from
final_chatbot's recommend_for_user / add_promotions.
"""
from __future__ import annotations

import os
import random
import threading

import httpx

from text_utils import tokenize, singular

BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:3000")
_FETCH_TIMEOUT = float(os.environ.get("PRODUCTS_FETCH_TIMEOUT_S", "15"))
_TOP_N = 5


class ProductCatalog:
    """Products for one building, indexed by store (poiId) and by category."""

    def __init__(self, products: list[dict]):
        self.products = products or []
        self.by_poi: dict[str, list[dict]] = {}
        self.by_category: dict[str, list[dict]] = {}
        for p in self.products:
            pid = p.get("poiId")
            if pid:
                self.by_poi.setdefault(pid, []).append(p)
            cat = (p.get("category") or "").lower()
            if cat:
                self.by_category.setdefault(cat, []).append(p)

    def top_for_poi(self, poi_id: str, n: int = _TOP_N, query: str | None = None) -> list[dict]:
        items = self.by_poi.get(poi_id, [])
        # A store can span several product categories (e.g. Computer Systems Hub
        # = laptops + peripherals). If the query names a specific one, narrow to
        # matching products so "a good laptop" doesn't return printers.
        narrowed = _filter_by_query(items, query)
        items = narrowed or items
        return sorted(items, key=lambda x: x.get("rating", 0), reverse=True)[:n]

    def top_overall(self, n: int = _TOP_N, interests: set[str] | None = None) -> list[dict]:
        # Only products tied to a real store (so we can offer navigation).
        items = [p for p in self.products if p.get("poiId")]
        interests = interests or set()
        # Interest-matching products first, then by rating — personalizes the
        # generic "recommend something" case by the user's onboarding interests.
        items.sort(
            key=lambda x: ((x.get("category") or "").lower() in interests, x.get("rating", 0)),
            reverse=True,
        )
        return items[:n]


def _filter_by_query(items: list[dict], query: str | None) -> list[dict]:
    """Keep products whose category/subCategory/name contains a query keyword.
    Returns [] when there's no query or nothing matches (caller falls back)."""
    if not query:
        return []
    tokens = {singular(t) for t in tokenize(query)}
    if not tokens:
        return []
    out = []
    for p in items:
        hay = f"{p.get('category','')} {p.get('subCategory','')} {p.get('name','')}".lower()
        if any(t in hay for t in tokens):
            out.append(p)
    return out


# --- per-building version cache (mirrors catalog.get_catalog_by_version) ------
_cache: dict[str, tuple[str, ProductCatalog]] = {}
_cache_lock = threading.Lock()


def _fetch_products(building_id: str) -> list[dict] | None:
    url = f"{BACKEND_BASE_URL}/api/client/products"
    try:
        resp = httpx.get(url, params={"buildingId": building_id}, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:  # noqa: BLE001 - network/daemon errors are non-fatal
        print(f"[products] fetch error for building {building_id}: {e}")
        return None
    data = body.get("data") if isinstance(body, dict) else body
    return data if isinstance(data, list) else []


def get_catalog(building_id: str, version: str | None) -> ProductCatalog | None:
    """Return the cached ProductCatalog for (building, version), fetching from
    the backend on a cache miss. Returns None if products can't be loaded."""
    if not building_id:
        return None
    key = building_id
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] == version:
            return cached[1]

    products = _fetch_products(building_id)
    if products is None:
        return None
    catalog = ProductCatalog(products)
    with _cache_lock:
        _cache[key] = (version or "", catalog)
    return catalog


# --- recommendation formatting (ported from final_chatbot) --------------------

def _with_promo(products: list[dict]) -> list[dict]:
    """Attach a random promo discount, like the original add_promotions."""
    out = []
    for p in products:
        original = int(p.get("price") or 0)
        discount = random.randint(10, 50)
        discounted = int(original * (1 - discount / 100))
        out.append({
            "name": p.get("name", "Unknown Product"),
            "brand": p.get("brand") or "",
            "original": original,
            "discounted": discounted,
            "discount": discount,
            "rating": p.get("rating", 0),
        })
    return out


def _format_reply(promos: list[dict], store_name: str, lang: str) -> str:
    lines = []
    for p in promos:
        brand = f"{p['brand']} — " if p["brand"] else ""
        star = f"⭐{p['rating']}"
        if p["original"] > 0:
            price = f"{p['discounted']} (was {p['original']}, {p['discount']}% off)"
        else:
            price = "—"
        lines.append(f"• {brand}{p['name']} — {price} {star}")
    body = "\n".join(lines)
    if lang == "ar":
        return (f"دي أحسن المنتجات في {store_name}:\n\n{body}\n\nتحب أوصّلك للمحل؟")
    return (f"Here are some top picks at {store_name}:\n\n{body}\n\nWant me to guide you to the store?")


def recommend(building_id: str, version: str | None, target_poi: dict | None,
              lang: str, query: str | None = None,
              interests: list[str] | None = None) -> tuple[str, str | None, int] | None:
    """Build a product recommendation.

    Returns (reply_text, suggest_poi_id, floor_level) or None if no products
    are available (caller then falls back to the backend store engine).
    """
    catalog = get_catalog(building_id, version)
    if catalog is None or not catalog.products:
        return None

    interest_set = {i.lower() for i in (interests or []) if i}

    if target_poi and target_poi.get("id"):
        # Explicit product/store named -> query-driven within that store.
        picks = catalog.top_for_poi(target_poi["id"], query=query)
        store_poi = target_poi
    else:
        # Generic "recommend something" -> personalize by the user's interests.
        picks = catalog.top_overall(interests=interest_set)
        store_poi = None

    if not picks:
        # e.g. the resolved store has no seeded products — try building-wide.
        picks = catalog.top_overall(interests=interest_set)
        store_poi = None
        if not picks:
            return None

    # Determine which store to offer navigation to.
    if store_poi is None:
        # Suggest the store of the top pick (resolved back through poiId).
        top_poi_id = picks[0].get("poiId")
        store_name = picks[0].get("category") or "the store"
        floor = 0
    else:
        top_poi_id = store_poi["id"]
        store_name = store_poi.get("name") or "the store"
        floor = store_poi.get("floorLevel", 0)

    reply = _format_reply(_with_promo(picks), store_name, lang)
    return reply, top_poi_id, floor
