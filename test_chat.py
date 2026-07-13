"""Test harness for the Navimind chatbot-service.

Two modes:

  python test_chat.py suite            # automated scenario suite (pass/fail)
  python test_chat.py chat             # interactive REPL (simulates the app)

Both default to in-process calls (no server needed). Add --http to test a
running server instead:

  python test_chat.py suite --http http://127.0.0.1:8000
  python test_chat.py chat  --http http://127.0.0.1:8000

POIs: uses the built-in sample mall below, or pull the real ones from the
backend with --backend (backend must be running and have data):

  python test_chat.py chat --backend http://127.0.0.1:3000 --building <buildingId>

The REPL mirrors the app exactly: when a reply carries action.type == "suggest"
it echoes that poiId back as pendingPoiId on your next message, so you can test
the "yes/no" confirm flow like a real conversation.

Note: scenarios marked [llm] need Ollama + qwen2.5:3b running; without them the
suite still runs the deterministic scenarios and skips/flags the LLM ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# --- sample mall (used unless --backend is given) ---------------------------
SAMPLE_POIS = [
    dict(id="p_computers", name="Computer Systems Hub", code="351", type="STORE",
         floorLevel=3, category="Computers", description="Laptops, desktops and PC accessories.",
         aliases=["laptop", "pc", "computer", "لابتوب", "كمبيوتر"],
         productKeywords=["laptop", "desktop", "charger", "keyboard", "mouse"], active=True),
    dict(id="p_mobiles", name="Mobile & Tablets Hub", code="352", type="STORE",
         floorLevel=3, category="Mobiles", description="Phones and tablets.",
         aliases=["mobile", "phone", "موبايل", "تليفون", "هاتف"],
         productKeywords=["iphone", "samsung", "tablet", "smartphone"], active=True),
    dict(id="p_gaming", name="Gaming Store", code="356", type="STORE",
         floorLevel=3, category="Gaming", description="Consoles and video games.",
         aliases=["gaming", "بلايستيشن", "اكس بوكس"],
         productKeywords=["playstation", "xbox", "console", "games"], active=True),
    dict(id="p_kitchen", name="Kitchen & Dining", code="457", type="STORE",
         floorLevel=4, category="Household", description="Cookware, utensils, dinnerware.",
         aliases=["kitchen", "cookware", "مطبخ"],
         productKeywords=["pots", "pans", "utensils", "ملاعق", "اطباق"], active=True),
]


# --- transports --------------------------------------------------------------
class InProcess:
    name = "in-process"

    def __init__(self):
        import app  # noqa: PLC0415
        self.app = app

    def chat(self, message, pois, pending=None, building="test-building"):
        req = self.app.ChatRequest(
            message=message, buildingId=building, version="test-version",
            pois=[self.app.PoiIn(**p) for p in pois], pendingPoiId=pending,
        )
        resp = self.app.chat(req)
        return json.loads(resp.model_dump_json())

    def health(self):
        return self.app.health()


class Http:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.name = f"http {self.base}"

    def _post(self, path, payload):
        import os
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("CHATBOT_SERVICE_TOKEN")
        if token:
            headers["X-Chatbot-Token"] = token
        req = urllib.request.Request(
            f"{self.base}{path}", data=data,
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat(self, message, pois, pending=None, building="test-building"):
        return self._post("/chat", {
            "message": message, "buildingId": building, "version": "test-version",
            "pendingPoiId": pending, "pois": pois,
        })

    def health(self):
        import os
        headers = {}
        token = os.environ.get("CHATBOT_SERVICE_TOKEN")
        if token:
            headers["X-Chatbot-Token"] = token
        req = urllib.request.Request(f"{self.base}/health", headers=headers, timeout=10)
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode("utf-8"))


def fetch_backend_pois(backend_url, building_id):
    url = f"{backend_url.rstrip('/')}/api/client/pois?buildingId={building_id}"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    raw = data if isinstance(data, list) else data.get("data") or data.get("pois") or []
    pois = []
    for p in raw:
        cat = p.get("category")
        cats = [c.get("name") for c in (p.get("categories") or []) if c.get("name")]
        pois.append(dict(
            id=p["id"], name=p.get("name", ""), code=p.get("code"),
            type=str(p.get("type") or ""), floorLevel=p.get("floorLevel", 0),
            category=(cat.get("name") if isinstance(cat, dict) else cat),
            categories=cats,
            description=p.get("description"), aliases=p.get("aliases") or [],
            productKeywords=p.get("productKeywords") or [], active=p.get("active", True),
        ))
    print(f"[backend] loaded {len(pois)} POIs for building {building_id}")
    return pois


# --- automated suite ----------------------------------------------------------
def run_suite(t, pois):
    llm_up = False
    try:
        h = t.health()
        llm_up = bool(h.get("modelLoaded"))
        print(f"health: {h}")
    except Exception as e:  # noqa: BLE001
        print(f"health check failed: {e}")
    print(f"transport: {t.name} | LLM available: {llm_up}\n")

    # (label, message, pending, expect_fn, needs_llm)
    def expect_suggest(pid):
        return lambda r: (r.get("action") or {}).get("type") == "suggest" and (r.get("action") or {}).get("poiId") == pid

    def expect_navigate(pid):
        return lambda r: (r.get("action") or {}).get("type") == "navigate" and (r.get("action") or {}).get("poiId") == pid

    def expect_no_action(r):
        return not r.get("action")

    def expect_mall_listing(r):
        # A real mall_info listing names actual stores and is NOT the
        # "couldn't find that" not-found template.
        rep = (r.get("reply") or "")
        if not r.get("action") and "couldn't find" not in rep and "مش لاقي" not in rep:
            return any(s in rep for s in ("Hub", "Store", "Kitchen", "Gaming"))
        return False

    def expect_refusal(r):
        # out_of_scope: no action, non-empty reply, and does NOT contain a joke
        # punchline / the word "joke" answered back.
        rep = (r.get("reply") or "").lower()
        return (not r.get("action")) and bool(rep.strip()) and "atoms" not in rep

    scenarios = [
        ("EN product query -> suggest", "where can I buy a laptop?", None,
         expect_suggest("p_computers"), False),
        ("AR product query -> suggest", "عايز اشتري موبايل", None,
         expect_suggest("p_mobiles"), False),
        ("AR normalized alias (تليفون)", "فين التليفونات", None,
         expect_suggest("p_mobiles"), False),
        ("confirm YES -> navigate", "yes", "p_computers",
         expect_navigate("p_computers"), False),
        ("confirm 'Yepppp' (stretched) -> navigate", "Yepppp", "p_computers",
         expect_navigate("p_computers"), False),
        ("confirm 'okkk' -> navigate", "okkk", "p_mobiles",
         expect_navigate("p_mobiles"), False),
        ("no echo: reply never equals user msg [llm]", "Yepppp", None,
         lambda r: r.get("reply", "").strip().lower() != "yepppp", True),
        ("AR confirm YES (ايوه) -> navigate", "ايوه", "p_gaming",
         expect_navigate("p_gaming"), False),
        ("confirm NO -> no action", "لا", "p_computers",
         expect_no_action, False),
        ("anti-hijack: new query beats stale pending", "I want a playstation", "p_computers",
         expect_suggest("p_gaming"), False),
        ("recommend with pending -> handoff (NOT 'okay bye')", "recommend me a place", "p_gaming",
         lambda r: r.get("handoff") == "recommend", False),
        ("AR recommend -> handoff", "اقترح عليا مكان اروحه", None,
         lambda r: r.get("handoff") == "recommend", False),
        ("AR 'list all shops' -> mall_info listing [llm]", "قولي كل المحلات", None,
         expect_mall_listing, True),
        ("EN 'list all shops' -> mall_info listing [llm]", "show me all the shops", None,
         expect_mall_listing, True),
        ("EN imperative 'list all stores' -> mall_info [llm]", "list all stores", None,
         expect_mall_listing, True),
        ("EN joke -> refuse, no joke told [llm]", "tell me a joke", None,
         expect_refusal, True),
        ("AR joke -> refuse [llm]", "قولي نكتة", None,
         expect_refusal, True),
        ("pending + real question not hijacked as 'no' [llm]", "هو ادهم بيفهم؟", "p_mobiles",
         lambda r: not r.get("clearPending"), True),
        ("greeting -> chitchat, no action [llm]", "hello, how are you?", None,
         expect_no_action, True),
        ("out-of-scope -> refusal, no action [llm]", "what is the capital of Egypt?", None,
         expect_no_action, True),
        ("AR out-of-scope [llm]", "من هو رئيس مصر؟", None,
         expect_no_action, True),
        ("semantic RAG (no exact keyword) [llm]", "I need something to fry eggs in", None,
         expect_suggest("p_kitchen"), True),
        ("verifier blocks food -> not found [llm]", "I want to buy chocolate", None,
         expect_no_action, True),
    ]

    passed = failed = skipped = 0
    for label, msg, pending, check, needs_llm in scenarios:
        if needs_llm and not llm_up:
            print(f"  SKIP  {label} (LLM not available)")
            skipped += 1
            continue
        try:
            r = t.chat(msg, pois, pending=pending)
            ok = check(r)
        except Exception as e:  # noqa: BLE001
            r, ok = {"error": str(e)}, False
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  {label}")
        if not ok:
            print(f"        msg={msg!r} pending={pending}")
            print(f"        got: {json.dumps(r, ensure_ascii=False)[:300]}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


# --- interactive REPL -----------------------------------------------------------
def run_repl(t, pois):
    print(f"transport: {t.name} | {len(pois)} POIs | type 'quit' to exit")
    print("The REPL echoes suggest-actions back as pendingPoiId, like the real app.\n")
    pending = None
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() in {"quit", "exit"}:
            break
        try:
            r = t.chat(msg, pois, pending=pending)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {e}")
            continue
        print(f"bot> {r.get('reply', '')}")
        action = r.get("action")
        if action:
            print(f"     [action: {action['type']} -> {action['poiId']} (floor {action['floorLevel']})]")
        # mirror the app: a 'suggest' becomes the pending offer for next turn
        pending = action["poiId"] if action and action["type"] == "suggest" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["suite", "chat"])
    ap.add_argument("--http", metavar="URL", help="test a running server instead of in-process")
    ap.add_argument("--backend", metavar="URL", help="fetch real POIs from the Navimind backend")
    ap.add_argument("--building", metavar="ID", help="buildingId for --backend")
    args = ap.parse_args()

    pois = SAMPLE_POIS
    if args.backend:
        if not args.building:
            ap.error("--backend requires --building <buildingId>")
        pois = fetch_backend_pois(args.backend, args.building)

    t = Http(args.http) if args.http else InProcess()
    if args.mode == "suite":
        sys.exit(0 if run_suite(t, pois) else 1)
    run_repl(t, pois)


if __name__ == "__main__":
    main()
