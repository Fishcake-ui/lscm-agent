"""
parse_order.py — Order parsing agent for Abir's F&B distribution (Pune)
Anthropic API direct + Claude Haiku 4.5 with tool use.

Auth: set Anthropic API key before running.
  PowerShell:  $env:ANTHROPIC_API_KEY = "<your sk-ant-... key>"
  cmd.exe:     set ANTHROPIC_API_KEY=<your key>

Install:
  pip install anthropic

Run:
  python parse_order.py                          # run all 7 test messages
  python parse_order.py "Sheraton kal 5kg pnr"   # single message
  python parse_order.py --sonnet                 # use Sonnet 4.6 instead of Haiku

Customer master: customers.json must be in the same directory.
"""

import anthropic
import json
import os
import re
import sys
import time
from difflib import get_close_matches

# ---------------- CONFIG ----------------
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"
MAX_ITERATIONS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 1024

# Anthropic API pricing (USD per 1M tokens) — for cost log only
PRICING = {
    MODEL_HAIKU:  {"in": 1.00, "out": 5.00},
    MODEL_SONNET: {"in": 3.00, "out": 15.00},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMERS_PATH = os.path.join(SCRIPT_DIR, "customers.json")

# ---------------- CUSTOMER MASTER (loaded from JSON) ----------------
def _load_customers():
    if not os.path.exists(CUSTOMERS_PATH):
        print(f"WARNING: {CUSTOMERS_PATH} not found. Using empty master.")
        return []
    with open(CUSTOMERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("customers", [])

CUSTOMER_MASTER = _load_customers()


def lookup_customer(query: str) -> dict:
    """Fuzzy match customer name across both company masters. Top 3 matches."""
    q = (query or "").lower().strip()
    if not q:
        return {"query": query, "matches": [], "count": 0}

    matches = []
    seen = set()
    # 1. substring match on name + aliases
    for c in CUSTOMER_MASTER:
        names = [c["name"].lower()] + c.get("aliases", [])
        for n in names:
            if q == n:
                score = 1.0
            elif q in n or n in q:
                score = 0.88
            else:
                continue
            if c["id"] not in seen:
                seen.add(c["id"])
                matches.append({**_public_fields(c), "match_score": score})
            break

    # 2. fallback fuzzy across all aliases
    if not matches:
        all_aliases = [a for c in CUSTOMER_MASTER for a in [c["name"].lower()] + c.get("aliases", [])]
        close = get_close_matches(q, all_aliases, n=5, cutoff=0.55)
        for n in close:
            for c in CUSTOMER_MASTER:
                if n in [c["name"].lower()] + c.get("aliases", []) and c["id"] not in seen:
                    seen.add(c["id"])
                    matches.append({**_public_fields(c), "match_score": 0.65})

    matches.sort(key=lambda m: m["match_score"], reverse=True)
    return {"query": query, "matches": matches[:3], "count": len(matches)}


def _public_fields(c: dict) -> dict:
    """Strip large/internal fields before sending to the agent."""
    return {
        "id": c["id"],
        "name": c["name"],
        "company": c.get("company"),
        "area": c.get("area"),
        "pincode": c.get("pincode"),
        "match_status": c.get("match_status"),
    }


# ---------------- TOOL SPEC (Anthropic API format) ----------------
TOOLS = [
    {
        "name": "lookup_customer",
        "description": (
            f"Look up a customer (hotel / restaurant / catering account) by name or "
            f"partial name across the unified customer master ({len(CUSTOMER_MASTER)} accounts "
            f"spanning Abir Foods, James Smith/Capella, and shared/Both accounts). "
            f"Call this FIRST on every order message before extracting items. "
            f"Pass the customer name fragment VERBATIM from the message — preserve "
            f"apostrophes, capitalisation, spacing. Handles abbreviations like "
            f"'JW' -> JW Marriott. Returns top 3 matches with id, name, company, "
            f"area, pincode, match_status, and a match_score 0..1. "
            f"company is 'Abir Foods' | 'James Smith' | 'Both' (account exists in both Tally DBs)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Customer name fragment from the message, VERBATIM. Examples: 'Sheraton', 'JW marriott', \"Mila's NIBM\", 'Hilton hinjewadi'."
                }
            },
            "required": ["query"]
        }
    }
]


def run_tool(name: str, args: dict) -> dict:
    if name == "lookup_customer":
        return lookup_customer(**args)
    return {"error": f"unknown tool: {name}"}


# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """You are an order-parsing agent for Abir's F&B distribution business in Pune, India.

# IDENTITY
You serve two companies operating from the same warehouse:
- Abir Foods — 75 hotel/restaurant accounts
- James Smith / Capella — 258 accounts
- 38 accounts overlap (served by Both companies — these have company="Both" in the lookup result)
Customers are hotels, restaurants, caterers, and clubs. All on 30-day credit.

# YOUR JOB
Parse incoming WhatsApp order messages into structured JSON. Salesmen relay these messages; customers send them. Both write in Hinglish (English + Hindi/Marathi mix). You do NOT execute orders. You only parse and classify. A separate system handles confirmation, routing, and invoicing.

# CORE PRINCIPLES
1. ACCURACY OVER SPEED. A wrongly-parsed order costs money. An order flagged for human review costs a few seconds.
2. CONFIDENCE OVER CERTAINTY. When unsure, set confidence=medium/low and flag for review. Never invent.
3. VERBATIM QUERIES. Pass strings to tools exactly as they appear in the message — preserve apostrophes, capitalisation, spacing. Do not "clean up" the input.
4. ONE TOOL CALL PER QUESTION. Don't loop on the same tool with variations.
5. JSON ONLY. Your final response is a single JSON object — no prose, no markdown fences, no explanations outside the "notes" field.

# HINGLISH GLOSSARY
Time:
- "aaj" = today | "kal" = tomorrow (in F&B context always future, not yesterday) | "parso" = day after tomorrow
- "morning" = before 11am | "afternoon" = 12-4pm | "evening" = 4-7pm | "raat"/"night" = after 7pm
- TIME-WITHOUT-DAY HEURISTIC: if a time is given but no day (e.g. "by 11am", "till 5pm"), default delivery_when to "tomorrow [time]". F&B orders placed during the day deliver next morning. Only assume "today" if the message explicitly says "aaj"/"today"/"now"/"abhi".

Products (shorthand):
- pnr = paneer | chkn = chicken | mtn = mutton
- "dairy" = category (milk/curd/butter/paneer/cheese — ambiguous, flag for review)
- "veg"/"non-veg" = category (always ambiguous, flag for review)
- "case" = 12 units (standard pack)
- "regular"/"usual"/"standing order" = means historical pattern — always flag, items=[]

Quantities & amounts:
- kg = kilograms | ltr/litre = volume | pcs/nos/piece = individual units
- "1k" = 1000 | "10k" = 10000 | "1 lakh" = 100000
- "8000 ka" / "₹8000 worth" / "8k ka" = amount mentioned

# PROCESS (do these in order, every time)

STEP 1: IDENTIFY CUSTOMER
Extract the customer name fragment from the message. Pass it VERBATIM to lookup_customer (do not strip apostrophes, do not change case beyond what's needed for the tool). Always call lookup_customer first. Always. Even if you "know" the customer.

STEP 2: EVALUATE MATCH
- match_score >= 0.85 → confident match, use it
- 0.70 <= match_score < 0.85 → usable, but downgrade confidence to medium
- match_score < 0.70 OR zero matches → DO NOT GUESS. Set customer_match=null, needs_human_review=true
- If multiple matches >= 0.85 and they look distinct (different IDs, different areas), the message is AMBIGUOUS. Pick the best candidate but downgrade confidence to medium AND flag for review.

STEP 3: EXTRACT ITEMS
For each product mentioned, capture: product (normalised name), quantity (number), unit (kg|case|litre|piece).
- Decode shorthand using the glossary
- If quantity is missing for a product, set quantity=null
- If a category word appears instead of a SKU (dairy, veg, non-veg), capture it as-is — and flag

STEP 4: EXTRACT TIMING
Capture delivery_when as plain text ("tomorrow morning", "monday 11am", "today by 5pm").
- Apply the TIME-WITHOUT-DAY HEURISTIC (default to tomorrow)
- If no timing at all, set "not specified"

STEP 5: EXTRACT AMOUNT
If a rupee figure is mentioned, capture as integer (8000 ka → 8000, 50k → 50000). Otherwise null.

STEP 6: CLASSIFY INTENT
- new_order — customer wants to place a new order
- cancel — customer wants to cancel a prior order
- confirmation_request — customer asks us to confirm something ("confirmation?", "confirm karo", trailing question mark on an order)
- clarification_needed — message is too vague to act on at all

STEP 7: SCORE CONFIDENCE
- high — clear customer match (>=0.85, single), all items have qty+unit, specific timing, unambiguous intent
- medium — one borderline field (match 0.7-0.85, multiple plausible matches, or one item ambiguous, or fuzzy timing)
- low — multiple borderline fields, or any field missing entirely

STEP 8: HUMAN REVIEW FLAG
Set needs_human_review = true if ANY of these is true. Check every one — do not skip:
a) customer_match is null OR match_score < 0.70
b) amount_mentioned > 30000
c) intent is anything other than new_order (cancels and confirmations always need a human)
d) items list is empty
e) any item product is a category (dairy, veg, non-veg) instead of a specific SKU
f) delivery_when contains "not specified" or is unclear
g) confidence is "low"
h) any single line item exceeds the sanity ceiling: >30kg meat, >20kg paneer/cheese, >5 case dairy
i) more than one customer match returned with score >= 0.85 from different IDs (ambiguous customer)

# GUARDRAILS (hard rules — never violate)
- NEVER invent a customer. If no match, customer_match=null and flag.
- NEVER auto-confirm orders > ₹30,000 — always flag.
- NEVER infer "regular order"/"usual" items. Always flag with items=[].
- NEVER add items not mentioned in the message.
- NEVER guess a delivery date when none is given. Use "not specified" and flag.
- If the message is incomprehensible, set intent=clarification_needed, items=[], needs_human_review=true, and explain in notes.

# OUTPUT FORMAT
Return ONLY this JSON object. No markdown fences. No prose before or after.

{
  "raw_message": "<original message verbatim>",
  "customer_query": "<the string you passed to lookup_customer>",
  "customer_match": {"id":"...","name":"...","company":"...","area":"...","pincode":"...","match_score":0.0} OR null,
  "items": [{"product":"...","quantity":<number or null>,"unit":"kg|case|litre|piece"}],
  "delivery_when": "<plain text>",
  "amount_mentioned": <integer or null>,
  "intent": "new_order|cancel|confirmation_request|clarification_needed",
  "confidence": "high|medium|low",
  "needs_human_review": <bool>,
  "notes": "<one short line — why flagged, what's ambiguous, what to watch>"
}"""


# ---------------- AGENT LOOP ----------------
def parse_order(message: str, model: str = MODEL_HAIKU, verbose: bool = False) -> dict:
    """Run the agent on one message. Returns parsed JSON dict (or error dict)."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    messages = [{"role": "user", "content": message}]

    total_in, total_out = 0, 0
    t0 = time.time()

    for it in range(MAX_ITERATIONS):
        if verbose:
            print(f"  iter {it+1}...", end="", flush=True)

        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        # append assistant turn (content blocks as-is)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = _extract_json(text, message)
            parsed["_meta"] = _meta(model, total_in, total_out, time.time() - t0, it + 1)
            if verbose:
                print(" done")
            return parsed

        if resp.stop_reason == "tool_use":
            tool_results = []
            for b in resp.content:
                if b.type == "tool_use":
                    if verbose:
                        print(f"\n    -> {b.name}({json.dumps(b.input)})", end="")
                    out = run_tool(b.name, b.input)
                    if verbose:
                        nmatch = out.get("count", "?") if isinstance(out, dict) else "?"
                        print(f"  [{nmatch} match]", end="")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": json.dumps(out, ensure_ascii=False),
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        if verbose:
            print(f" stopped: {resp.stop_reason}")
        return {"error": f"unexpected stop_reason: {resp.stop_reason}", "raw_message": message,
                "_meta": _meta(model, total_in, total_out, time.time() - t0, it + 1)}

    return {"error": "max_iterations_exceeded", "raw_message": message,
            "_meta": _meta(model, total_in, total_out, time.time() - t0, MAX_ITERATIONS)}


def _meta(model, t_in, t_out, secs, iters):
    p = PRICING.get(model, {"in": 0, "out": 0})
    cost_usd = (t_in / 1_000_000) * p["in"] + (t_out / 1_000_000) * p["out"]
    return {
        "model": model,
        "tokens_in": t_in,
        "tokens_out": t_out,
        "iterations": iters,
        "latency_s": round(secs, 2),
        "cost_usd": round(cost_usd, 6),
        "cost_inr_approx": round(cost_usd * 84, 4),
    }


def _extract_json(text: str, original: str) -> dict:
    """Pull JSON object out of model output (handles ```json fences and plain)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"json_parse_failed: {e}", "raw_text": text[:500], "raw_message": original}


# ---------------- TEST HARNESS ----------------
TEST_MESSAGES = [
    "Sheraton bhai kal 5kg paneer 10kg chicken bhejna hai 8000 ka",
    "JW marriott monday morning 3 case dairy",
    "Pls 2kg pnr 5kg chkn for Hilton hinjewadi by 11am",
    "Tomorrow Mila's NIBM regular order",
    "Forest Club Karjat 50kg chicken wedding hai",
    "Cancel yesterday's order for Sheraton",
    "Ek case dairy aur 2kg cheese for Westin kal morning - confirmation?",
]


def run_tests(model: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("PowerShell:  $env:ANTHROPIC_API_KEY = '<your sk-ant-... key>'")
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Customer master: {len(CUSTOMER_MASTER)} accounts loaded")
    print(f"Messages: {len(TEST_MESSAGES)}")
    print("=" * 78)

    results = []
    grand_cost = 0.0
    for i, msg in enumerate(TEST_MESSAGES, 1):
        print(f"\n[{i}/{len(TEST_MESSAGES)}] {msg}")
        try:
            r = parse_order(msg, model=model, verbose=True)
            results.append({"message": msg, "result": r})
            print(json.dumps(r, indent=2, ensure_ascii=False))
            grand_cost += r.get("_meta", {}).get("cost_usd", 0.0)
        except anthropic.APIError as e:
            print(f"  API ERROR: {type(e).__name__}: {e}")
            results.append({"message": msg, "error": f"{type(e).__name__}: {e}"})
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            results.append({"message": msg, "error": f"{type(e).__name__}: {e}"})

    out_path = os.path.join(SCRIPT_DIR, "parse_order_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 78)
    print(f"Total cost: ${grand_cost:.6f}  (~₹{grand_cost*84:.4f})")
    print(f"Saved: {out_path}")


def run_single(message: str, model: str):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)
    print(f"Customer master: {len(CUSTOMER_MASTER)} accounts loaded")
    r = parse_order(message, model=model, verbose=True)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    args = sys.argv[1:]
    model = MODEL_SONNET if "--sonnet" in args else MODEL_HAIKU
    args = [a for a in args if a != "--sonnet"]

    if args:
        run_single(" ".join(args), model)
    else:
        run_tests(model)
