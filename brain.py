"""The chatbot brain: intent understanding (Qwen) + deterministic pre-routing +
RAG resolution over Navimind POIs + the confirm-flow orchestrator.

Ported from newchatbot/final_version_of_chatbot.py's `chatbot_respond` and the
LLM/RAG helpers, adapted so that:
  * retrieval runs over Navimind POIs (a POI == a store), and
  * resolution returns a real poiId + floorLevel, so the caller emits a
    Navimind navigation `action` instead of Adham room numbers / A* text.
"""
from __future__ import annotations

import random

import llm
import products
from catalog import BuildingCatalog, _poi_category
from text_utils import (
    detect_language, detect_yes_no, looks_like_list_query, norm,
)

# Deterministic "recommend me something" detection (EN/AR/Egyptian). Resolved
# before the LLM: the old backend handled this intent via its recommendation
# service, and the 3B model proved unreliable here — with a pending navigation
# offer in context it misread "recommend me a place" as declining the offer.
_RECOMMEND_HINTS = [norm(h) for h in [
    "recommend", "recommendation", "suggest", "suggestion", "where should i go",
    "what should i visit", "good product", "a good", "best product", "whats good",
    "what's good", "اقترح", "اقتراح", "رشح", "رشحلي", "ارشدني",
    "انصحني", "نصحني", "تنصحني", "اروح فين", "فين اروح", "اماكن حلوه",
    "حاجه حلوه", "حاجة حلوة", "افضل", "احسن حاجه",
]]


def looks_like_recommend_query(text: str) -> bool:
    low = norm(text)
    return any(h in low for h in _RECOMMEND_HINTS)

BORDERLINE_MATCH_THRESHOLD = 0.30  # below this, not even worth asking the verifier
CONFIDENT_MATCH_THRESHOLD = 0.40   # 0.30-0.40: accepted ONLY on an explicit verifier "yes"
STRONG_MATCH_THRESHOLD = 0.55      # above this, an inconclusive verifier is trusted

VALID_INTENTS = {"product_query", "list_query", "navigate_confirm_yes",
                 "navigate_confirm_no", "mall_info", "chitchat", "out_of_scope",
                 "recommend"}

SYSTEM_PROMPT = """You are the brain of a mall assistant chatbot named "Navimind Assistant". You understand English, Modern Standard Arabic, and Egyptian Arabic slang.

STRICT RULE: You are NOT a general-purpose assistant. You never perform tasks unrelated to the mall (counting, math, trivia, jokes on demand, coding, translation requests, etc.).
- Any request for things NOT sold in Electronics, Furniture, or Household sections (e.g., food, chocolate, grocery, clothes, shoes, makeup, medicines) -> intent = "out_of_scope".
- Any request involving people, names, or "buying" humans (e.g., "أنا عايز أشتري محمد") -> intent = "out_of_scope".
- Any general-knowledge or trivia question with no connection to the mall — capitals, countries, presidents/politicians, history, geography, religion, sports, world events, science facts, celebrities, etc. (e.g., "what is the capital of Egypt", "من هو رئيس مصر", "كلمني عن كأس العالم") -> intent = "out_of_scope". Do NOT answer the factual question yourself, even if you know the answer — always classify it as "out_of_scope" and let the standard out-of-scope reply handle it.
- Requests for entertainment or content generation — jokes, riddles, poems, stories, songs, fun facts, "say something funny" (e.g., "tell me a joke", "قولي نكتة", "قولي نكته", "احكيلي حكاية", "make me laugh") -> intent = "out_of_scope". NEVER actually tell the joke/riddle/story yourself, even a short one; always classify it as "out_of_scope" and return an empty "reply". This is a hard rule.
- A vague request for a suggestion with NO concrete product or store named — "recommend something", "tell me a good product", "what's good here", "something nice", "رشحلي حاجة", "قولي حاجة حلوة", "عايز حاجة كويسة", "ايه احسن حاجة" -> intent = "recommend" (leave "search_query" null, "reply" empty). Do NOT invent a product and do NOT match it to a random store.
- If the user asks for your name or who you are -> intent = "chitchat", reply = "مرحباً! أنا مساعد Navimind، موجود هنا عشان أساعدك تلاقي أي محل أو منتج في المول." (if Arabic) or "I am the Navimind Assistant, here to help you find any store or product in the mall." (if English).
- If the user greets you (e.g., "hi", "hello", "مرحباً", "أهلاً") -> intent = "chitchat", reply a short, professional welcome such as "Hello! Welcome to Navimind. How can I help you today?" (if English) or "أهلاً بيك في Navimind! إزاي أقدر أساعدك النهاردة؟" (if Arabic).
- If the user asks how you are doing (e.g., "how are you", "كيف حالك", "ازيك") -> intent = "chitchat", reply professionally and briefly, e.g. "I'm doing great, thank you for asking! How can I help you find something in the mall today?" (if English) or "أنا تمام، شكراً لسؤالك! إزاي أقدر أساعدك تلاقي حاجة في المول النهاردة؟" (if Arabic).
- The words "store", "stores", "shop", "shops", "محل", "محلات" are NOT products and can NEVER be a "search_query". If the ONLY thing the user is asking about is the stores/shops themselves — listing them, or what stores the mall has — the intent is ALWAYS "mall_info", NEVER "product_query" or "list_query". This holds for every phrasing, including imperatives: "list all stores", "list all shops", "show me all the shops", "show me every store", "give me all the shops", "what stores are here", "what shops do you have", "قولي كل المحلات", "اعرض كل المحلات", "ايه المحلات الموجودة", "فيه محلات ايه هنا", "عندكم محلات ايه" -> intent = "mall_info". Leave "search_query" null and "reply" empty; the system fills the store list itself.
- IMPORTANT boundary: "mall_info" is ONLY for the whole-mall, no-specific-product case. If the message names a specific product, brand, or category (e.g., "what laptops do you have", "which shops sell phones", "ايه أنواع الموبايلات", "ماركات اللابتوبات"), that product/category IS the "search_query" and the intent is "product_query" or "list_query", NOT "mall_info".

STRICT LANGUAGE RULE FOR ALL "reply" TEXT:
- The "reply" field must be written ENTIRELY in the SAME language as the user's message (pure English if the user wrote in English, pure Arabic/Egyptian Arabic if the user wrote in Arabic). Never mix languages within a single reply, and never answer in a different language than the one the user used.
- All chitchat replies (greetings, "how are you", self-introduction, small talk, etc.) must sound professional, polite, and courteous — avoid overly casual filler words (e.g., do not write English words like "alright" inside an Arabic sentence, and do not switch languages mid-reply).

Pure filler words with no semantic content (e.g., "يسطا", "يعني", "بس", "خلاص") are NOT product requests. If no identifiable product/store is mentioned, intent = "chitchat".

Output ONLY a single valid JSON object:
{
  "intent": "product_query" | "list_query" | "navigate_confirm_yes" | "navigate_confirm_no" | "recommend" | "mall_info" | "chitchat" | "out_of_scope",
  "search_query": "<canonical English noun phrase for the product/category, or null>",
  "reply": "<short natural reply in the user's language, empty string for product_query/navigate_confirm_*/out_of_scope>"
}

CANONICALIZATION:
- "mobile"/"phone" -> "mobile phone"
- "laptop" -> "laptop"
- Brand only ("Samsung") -> "Samsung product"

Mall Knowledge (Electronics, Furniture, Household ONLY):
- Kitchen (453): appliances (fridge, oven, blender).
- Kitchen and Dining (457): utensils, cookware, dinnerware (NOT food/chocolate).
- Mobile & Tablets Hub (352): phones, tablets.
- Gaming (356): consoles, games.
- Furniture (450-452, 454).
- Smart Devices Hub (350): cameras, audio.

CRITICAL:
- Do NOT match food, people, or clothing to these categories.
- "Chocolate", "Banana", "Mohamed" are all OUT OF SCOPE.
- Keep "reply" empty for product_query and navigate_confirm_*.
"""

VERIFY_PROMPT = """You are verifying whether a shopper's request belongs to a store.

Request: "{query}"
Store: "{store}" (category: {category}; sells: {examples})

RULES:
1. If the request is FOOD, a PERSON's name, or CLOTHING and the store clearly doesn't sell it, match is false.
2. Would a shopper looking for "{query}" reasonably expect to find it at "{store}"?

Reply ONLY with: {{"match": true}} or {{"match": false}}
"""


# --- LLM understanding -----------------------------------------------------

def llm_understand(user_text: str, lang: str, pending: bool) -> dict:
    lang_note = ("The user's message is ENGLISH; the \"reply\" must be entirely in professional English.\n"
                 if lang == "en" else
                 "The user's message is ARABIC; the \"reply\" must be entirely in professional Arabic/Egyptian.\n")
    if pending:
        context = ("There IS a pending yes/no navigation offer. Classify as navigate_confirm_yes/no ONLY when the ENTIRE message is a bare standalone confirmation word (yes/no/ايوه/لأ/تمام/ماشي) and nothing else. "
                   "If the message contains a question, a new request, or any other content — even a short one like \"does Adham understand?\" / \"هو ادهم بيفهم؟\" — it is NOT a confirmation: classify it by its own meaning (chitchat, product_query, out_of_scope, etc.), never as navigate_confirm_yes/no.\n")
    else:
        context = ("There is no pending navigation offer; do not classify this as navigate_confirm_yes/no "
                   "unless it is itself a standalone confirmation word (yes/no/ايوه/لأ/تمام/ماشي) with no other content.\n")
    user = context + lang_note + f"User message: {user_text}"
    data = llm.chat_json(SYSTEM_PROMPT, user)
    return _sanitize(data, user_text)


def _sanitize(data: dict | None, user_text: str) -> dict:
    if not isinstance(data, dict):
        return {"intent": "chitchat", "search_query": None,
                "reply": "معلش، ممكن توضح طلبك أكتر؟ / Sorry, could you rephrase that?"}
    data.setdefault("intent", "chitchat")
    data.setdefault("search_query", None)
    data.setdefault("reply", "")
    if data["intent"] not in VALID_INTENTS:
        data["intent"] = "product_query" if data.get("search_query") else "chitchat"
    return data


def llm_verify_match(query: str, poi: dict, examples: list[str]):
    """LLM sanity-check of a RAG match. Returns True/False, or None if the model
    couldn't answer cleanly (caller then falls back to a stricter score gate)."""
    prompt = VERIFY_PROMPT.format(
        query=query, store=poi.get("name", ""), category=_poi_category(poi) or "N/A",
        examples=", ".join(examples[:6]) if examples else "N/A",
    )
    data = llm.chat_json("You verify product-to-store matches. Output only JSON.", prompt, max_tokens=20)
    if isinstance(data, dict) and "match" in data:
        return bool(data["match"])
    return None


def mall_info(user_text: str, catalog: BuildingCatalog, lang: str) -> str:
    cats = sorted({_poi_category(p) for p in catalog.pois if _poi_category(p)})
    stores = [p.get("name", "") for p in catalog.pois][:25]
    info = (f"This building has these store categories: {', '.join(cats) or 'N/A'}. "
            f"Some stores: {', '.join(s for s in stores if s)}.")
    system = ("You are a mall assistant. Answer ONLY from the facts below, briefly, in the "
              f"user's language. If it can't be answered from these facts, say so politely.\n{info}")
    reply = llm.chat_text(system, user_text)
    if not reply:
        reply = ("للأسف مش قادر أجاوب على ده دلوقتي." if lang == "ar"
                 else "Sorry, I can't answer that right now.")
    return reply


# --- POI resolution --------------------------------------------------------

def resolve_poi(catalog: BuildingCatalog, search_query: str | None, raw_text: str):
    """Return (poi, confidence) or (None, 0). Deterministic alias fast-path
    first, then RAG + LLM verify."""
    direct = None
    if search_query:
        direct = catalog.alias_direct_match(search_query)
    if not direct:
        direct = catalog.alias_direct_match(raw_text)
    if direct:
        return direct, 1.0

    query = search_query or raw_text
    hits = catalog.search(query, top_k=3, min_score=BORDERLINE_MATCH_THRESHOLD)
    if not hits:
        return None, 0.0
    poi, score = hits[0]
    examples = (poi.get("productKeywords") or []) + (poi.get("aliases") or [])
    verified = llm_verify_match(query, poi, examples)
    if verified is False:
        return None, 0.0
    if score < CONFIDENT_MATCH_THRESHOLD and verified is not True:
        # Borderline embedding score: only an explicit verifier "yes" may
        # rescue it (handles vague phrasing like "something to fry eggs in"
        # without letting coincidental similarity through).
        return None, 0.0
    if verified is None and score < STRONG_MATCH_THRESHOLD:
        return None, 0.0
    return poi, score


# --- reply templates -------------------------------------------------------

def _floor(poi: dict):
    return poi.get("floorLevel", 0)


def _place_phrase(poi: dict, lang: str) -> str:
    """Human-friendly location, e.g. 'Gaming Store on floor 3 (room 356)'."""
    name = poi.get("name", "")
    code = poi.get("code") or ""
    floor = poi.get("floorLevel")
    if lang == "ar":
        bits = []
        if floor is not None:
            bits.append(f"الدور {floor}")
        if code:
            bits.append(f"غرفة {code}")
        return f"{name}" + (f" ({' - '.join(bits)})" if bits else "")
    bits = []
    if floor is not None:
        bits.append(f"floor {floor}")
    if code:
        bits.append(f"room {code}")
    return f"{name}" + (f" ({', '.join(bits)})" if bits else "")


def _suggest_reply(poi: dict, lang: str) -> str:
    place = _place_phrase(poi, lang)
    cat = _poi_category(poi)
    if lang == "ar":
        cat_bit = f" في قسم {cat}" if cat else ""
        return f"لقيت {place}{cat_bit} 👍 تحب أوصّلك هناك؟"
    cat_bit = f" in the {cat} section" if cat else ""
    return f"Found it — {place}{cat_bit}. 👍 Want me to guide you there?"


def _nav_reply(poi: dict, lang: str) -> str:
    place = _place_phrase(poi, lang)
    if lang == "ar":
        return f"🗺️ تمام! بوصّلك لـ {place} دلوقتي. اتبع الطريق على الخريطة."
    return f"🗺️ On it! Guiding you to {place} now — just follow the route on the map."


_NOT_FOUND = {
    "ar": ("للأسف مش لاقي المنتج ده في المكان، أو مش تابع لأي محل موجود. "
           "جرّب تقولي منتج أو محل تاني وهدورلك عليه."),
    "en": ("Sorry, I couldn't find that here, or it doesn't map to any store. "
           "Try another product or store name and I'll look it up."),
}
_OUT_OF_SCOPE = {
    "ar": [
        "السؤال ده خارج نطاق مساعدتي، بس اسأل عن أي محل أو منتج وأنا جاهز.",
        "ده مش من اختصاصي، بس لو محتاج تلاقي منتج أو تروح لمحل أنا موجود.",
    ],
    "en": [
        "That's outside what I can help with — ask me about any store or product and I'll jump right in.",
        "I can't help with that one, but I'm great at finding stores and products here. Want to try?",
    ],
}
_CHITCHAT = {
    "ar": ["أنا تمام، شكراً! قولي عايز تشتري ايه أو تروح فين في المكان.",
           "جاهز أساعدك — اسألني عن أي محل أو منتج."],
    "en": ["I'm doing great, thanks! Tell me what you're looking for or which store you want to reach.",
           "Ready to help — ask me about any store or product here."],
}


def _is_echo(reply: str, message: str) -> bool:
    """True when the LLM 'reply' is just the user's message parroted back
    (ignoring case, punctuation, and surrounding quotes)."""
    import re
    def squash(s: str) -> str:
        return re.sub(r"[\W_]+", "", str(s).lower())
    r, m = squash(reply), squash(message)
    if not r or not m:
        return False
    # Echo = the reply adds nothing beyond the user's text. (A reply that
    # merely CONTAINS the message, like "I'm fine, how are you?", is fine.)
    return r == m or r in m


# --- orchestrator ----------------------------------------------------------

def respond(message: str, catalog: BuildingCatalog, pending_poi_id: str | None,
            lang: str | None = None, building_id: str | None = None,
            products_version: str | None = None,
            interests: list[str] | None = None) -> dict:
    """Returns {reply, lang, action?}. `action` = {type, poiId, floorLevel}.
    A "suggest" action doubles as the pending confirmation offer: the caller
    echoes its poiId back as `pending_poi_id` on the next turn."""
    lang = lang or detect_language(message)
    awaiting = bool(pending_poi_id and catalog.by_id.get(pending_poi_id))

    def out(reply, action=None, handoff=None, clear_pending=False):
        r = {"reply": reply, "lang": lang}
        if action:
            r["action"] = action
        if handoff:
            r["handoff"] = handoff
        if clear_pending:
            r["clearPending"] = True
        return r

    # 1. Deterministic pre-routing (root fix: the small LLM is unreliable at
    #    intent once a "pending offer" hint is in context; resolve what we can
    #    with certainty here and only let novel phrasing reach the LLM).
    direct_poi = catalog.alias_direct_match(message)
    confirmation = detect_yes_no(message)

    if awaiting and confirmation and not direct_poi:
        intent = "navigate_confirm_yes" if confirmation == "yes" else "navigate_confirm_no"
        understanding = {"intent": intent, "search_query": None, "reply": ""}
    elif looks_like_recommend_query(message):
        # Checked before the direct-POI fast-path: "recommend a good phone"
        # names a product ("phone"), but the recommend cue means the user wants
        # product suggestions, not to be routed straight to the store.
        understanding = {"intent": "recommend", "search_query": None, "reply": ""}
    elif direct_poi:
        understanding = {
            "intent": "list_query" if looks_like_list_query(message) else "product_query",
            "search_query": message, "reply": "",
        }
    else:
        understanding = llm_understand(message, lang, pending=awaiting)

    intent = understanding.get("intent")
    llm_reply = understanding.get("reply") or ""

    # Hard guard: never act on a confirmation unless we're actually awaiting one.
    if intent in ("navigate_confirm_yes", "navigate_confirm_no") and not awaiting:
        intent = "product_query" if understanding.get("search_query") else "chitchat"

    # Guard against the "no + new request" trap: if the LLM labels a message a
    # rejection but the message actually carries a new request (it produced a
    # search_query, or the message is clearly more than a bare "no"), treat it
    # as that request instead of swallowing it with "okay, anything else?".
    if intent == "navigate_confirm_no" and understanding.get("search_query"):
        intent = "product_query"

    # 2. Confirmation handling
    if intent == "navigate_confirm_yes":
        poi = catalog.by_id.get(pending_poi_id)
        if not poi:
            return out(_CHITCHAT[lang][0])
        return out(_nav_reply(poi, lang),
                   {"type": "navigate", "poiId": poi["id"], "floorLevel": _floor(poi)})

    if intent == "navigate_confirm_no":
        return out("تمام، لو احتجت حاجة تانية أنا موجود." if lang == "ar"
                   else "Okay, let me know if you need anything else.",
                   clear_pending=True)

    # Recommendation request -> recommend real PRODUCTS from the backend catalog
    # (ported from final_chatbot). If a product/category is named, recommend
    # within that store; otherwise recommend top-rated products building-wide.
    # If products can't be loaded, fall back to the backend store engine.
    if intent == "recommend":
        # Resolve which store/category the recommendation targets. A confident
        # match (e.g. "recommend a good phone" -> phones) recommends within that
        # store; otherwise recommend top-rated products building-wide.
        poi, conf = resolve_poi(catalog, understanding.get("search_query"), message)
        target_poi = poi if conf >= CONFIDENT_MATCH_THRESHOLD else None
        rec = products.recommend(building_id, products_version, target_poi, lang,
                                 query=understanding.get("search_query") or message,
                                 interests=interests)
        if rec:
            reply, top_poi_id, floor = rec
            action = ({"type": "suggest", "poiId": top_poi_id, "floorLevel": floor}
                      if top_poi_id else None)
            return out(reply, action)
        return out("", handoff="recommend")

    # 3. List query -> list stores in the matched category, offer to navigate.
    if intent == "list_query":
        poi, _ = resolve_poi(catalog, understanding.get("search_query"), message)
        if poi:
            siblings = catalog.category_siblings(poi) or [poi]
            names = "، ".join(p.get("name", "") for p in siblings) if lang == "ar" \
                else ", ".join(p.get("name", "") for p in siblings)
            cat = _poi_category(poi)
            if lang == "ar":
                reply = f"في قسم {cat} عندنا: {names}. تحب أوصّلك لـ {_place_phrase(poi, lang)}؟"
            else:
                reply = f"In {cat} we have: {names}. Want me to guide you to {_place_phrase(poi, lang)}?"
            return out(reply, {"type": "suggest", "poiId": poi["id"], "floorLevel": _floor(poi)})
        return out(_NOT_FOUND[lang])

    # 4. Product query -> resolve to a store, offer to navigate.
    if intent == "product_query":
        poi, _ = resolve_poi(catalog, understanding.get("search_query"), message)
        if poi:
            return out(_suggest_reply(poi, lang),
                       {"type": "suggest", "poiId": poi["id"], "floorLevel": _floor(poi)})
        return out(_NOT_FOUND[lang])

    # 5. Mall info
    if intent == "mall_info":
        return out(mall_info(message, catalog, lang))

    # 6. Out of scope
    if intent == "out_of_scope":
        return out(random.choice(_OUT_OF_SCOPE[lang]))

    # 7. Chitchat (use the LLM's reply if it gave one).
    # Echo guard: small models sometimes put the user's own message in the
    # "reply" field (e.g. user says "Yepppp", model replies "Yepppp"). A reply
    # that is essentially the user's message is worse than a canned line, so
    # discard it.
    if llm_reply and _is_echo(llm_reply, message):
        llm_reply = ""
    return out(llm_reply or random.choice(_CHITCHAT[lang]))
