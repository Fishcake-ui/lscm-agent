"""
main.py - FastAPI entrypoint for the LSCM order agent.
Render-deployable: binds 0.0.0.0:$PORT, exposes /health for platform checks.

Routes:
GET /                   - service info
GET /health             - Render health probe
POST /parse             - parse one order message (test / debug)
POST /webhook/whatsapp  - WhatsApp BSP webhook (receives inbound, dispatches to parse_order, replies via BSP)

Local run:
python main.py
# or, with reload during dev:
# uvicorn main:app --reload --port 10000
"""

import hashlib
import hmac
import httpx
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lscm")

# Importing parse_order triggers the customer-master load (Sheets or local).
# The loader logs its source + count, so by the time the app boots we already
# know whether we're running on real data.
from parse_order import (  # noqa: E402 (intentional: log basicConfig first)
    CUSTOMER_MASTER,
    MODEL_HAIKU,
    MODEL_SONNET,
    parse_order,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    source = "google_sheet" if (sheet_id and os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")) else "local"
    log.info(
        "lscm-agent startup: customer_master=%d entries, source=%s, sheet_id=%s",
        len(CUSTOMER_MASTER), source, sheet_id or "(none)",
    )
    yield
    log.info("lscm-agent shutdown")

app = FastAPI(title="LSCM Order Agent", version="0.1.0", lifespan=lifespan)

class ParseRequest(BaseModel):
    message: str
    model: Optional[str] = None  # "haiku" (default) | "sonnet"

@app.get("/")
def root() -> dict[str, Any]:
    """Service info. Confirms env wiring without exposing secret values."""
    return {
        "service": "lscm-agent",
        "version": app.version,
        "customer_master_count": len(CUSTOMER_MASTER),
        "env": {
            "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google_geocoding_key_set": bool(os.getenv("GOOGLE_GEOCODING_API_KEY")),
            "google_sheets_creds_set": bool(os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")),
            "google_sheet_id": os.getenv("GOOGLE_SHEET_ID"),
        },
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/parse")
def parse(req: ParseRequest) -> dict[str, Any]:
    """Run the parsing agent on one WhatsApp-style order message."""
    model = MODEL_SONNET if (req.model or "").lower() == "sonnet" else MODEL_HAIKU
    log.info("parse: model=%s msg=%r", model, req.message[:160])
    try:
        return parse_order(req.message, model=model, verbose=False)
    except Exception as e:
        log.exception("parse failed")
        return {
            "error": f"{type(e).__name__}: {e}",
            "raw_message": req.message,
        }

# ================================================================
# WhatsApp BSP Webhook
# ================================================================
# Environment variables expected:
#   WHATSAPP_VERIFY_TOKEN   - the token you set in Meta developer console
#   WHATSAPP_APP_SECRET     - Meta app secret for SHA-256 HMAC payload verification
#   WHATSAPP_API_TOKEN      - Cloud API / BSP bearer token for sending replies
#   WHATSAPP_PHONE_NUMBER_ID - Phone number ID for the Cloud API send endpoint
#
# The webhook handles two HTTP methods:
#   GET  /webhook/whatsapp  - Meta hub.challenge verification handshake
#   POST /webhook/whatsapp  - Inbound message delivery

@app.get("/webhook/whatsapp")
async def webhook_whatsapp_verify(
    hub_mode: Optional[str] = None,
    hub_verify_token: Optional[str] = None,
    hub_challenge: Optional[str] = None,
) -> Any:
    """
    Meta webhook verification handshake.
    Meta sends GET with hub.mode=subscribe, hub.verify_token, hub.challenge.
    We must return hub.challenge as plain text if the token matches.
    """
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token == verify_token and hub_challenge:
        log.info("webhook/whatsapp: verification handshake OK")
        return int(hub_challenge)
    log.warning("webhook/whatsapp: verification failed mode=%s token_match=%s",
                hub_mode, hub_verify_token == verify_token)
    raise HTTPException(status_code=403, detail="verification failed")

@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request) -> dict[str, str]:
    """
    WhatsApp Cloud API / BSP inbound webhook.
    
    Flow:
      1. Validate HMAC-SHA256 signature from X-Hub-Signature-256 header.
      2. Parse the payload to extract inbound text messages.
      3. For each inbound text message, run parse_order (Haiku).
      4. Send a brief WhatsApp reply via the Cloud API with parse result summary.
      5. ACK 200 immediately so Meta does not retry.
    """
    # --- 1. Read raw body (needed for HMAC check) ---
    raw_body = await request.body()

    # --- 2. Signature verification ---
    app_secret = os.getenv("WHATSAPP_APP_SECRET", "")
    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if app_secret:
        expected = "sha256=" + hmac.new(
            app_secret.encode(), raw_body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig_header, expected):
            log.warning("webhook/whatsapp: HMAC mismatch, rejecting payload")
            raise HTTPException(status_code=401, detail="invalid signature")
    else:
        log.debug("webhook/whatsapp: WHATSAPP_APP_SECRET not set, skipping HMAC check")

    # --- 3. Parse JSON payload ---
    try:
        payload = json.loads(raw_body)
    except Exception:
        log.warning("webhook/whatsapp: non-JSON body received")
        return {"status": "received"}

    log.info("webhook/whatsapp raw payload: %r", str(payload)[:400])

    # --- 4. Extract inbound text messages ---
    entries = payload.get("entry", [])
    for entry in entries:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            for msg in messages:
                if msg.get("type") != "text":
                    log.info("webhook/whatsapp: skipping non-text message type=%s", msg.get("type"))
                    continue

                from_number = msg.get("from", "")
                text_body = msg.get("text", {}).get("body", "")
                msg_id = msg.get("id", "")

                log.info("webhook/whatsapp: inbound from=%s msg_id=%s body=%r",
                    from_number, msg_id, text_body[:160])

                # --- 5. Run order parser ---
                try:
                    parsed = parse_order(text_body, model=MODEL_HAIKU, verbose=False)
                except Exception as exc:
                    log.exception("webhook/whatsapp: parse_order failed")
                    parsed = {"error": str(exc)}

                # --- 6. Build reply text ---
                reply_text = _build_reply(parsed)

                # --- 7. Send reply via WhatsApp Cloud API ---
                await _send_whatsapp_reply(from_number, reply_text)

    return {"status": "received"}


def _build_reply(parsed: dict) -> str:
    """
    Build a short WhatsApp reply from a parsed order dict.
    Keeps it under ~200 chars so it's readable in a chat bubble.
    """
    if "error" in parsed:
        err = parsed.get("error", "unknown")[:80]
        return f"[LSCM] Parse error: {err}"

    customer = parsed.get("customer_match")
    cname = customer["name"] if customer else "Unknown customer"
    intent = parsed.get("intent", "?")
    confidence = parsed.get("confidence", "?")
    review = parsed.get("needs_human_review", False)
    items = parsed.get("items", [])
    item_summary = ", ".join(
        f'{i.get("quantity", "?")}{i.get("unit", "")} {i.get("product", "?")}'
        for i in items[:3]
    ) or "no items"

    flag = " [REVIEW]" if review else ""
    return (
        f"[LSCM] {cname} | {intent} | {item_summary} | "
        f"conf:{confidence}{flag}"
    )[:300]


async def _send_whatsapp_reply(to: str, text: str) -> None:
    """
    POST a text message reply via WhatsApp Cloud API.
    Silently logs errors - never raises, so the webhook always returns 200.
    """
    api_token = os.getenv("WHATSAPP_API_TOKEN", "")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")

    if not api_token or not phone_number_id:
        log.info("_send_whatsapp_reply: WHATSAPP_API_TOKEN or PHONE_NUMBER_ID not set, skipping send")
        return

    url = f"https://graph.facebook.com/v20.0/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            log.info("_send_whatsapp_reply: sent to=%s status=200", to)
        else:
            log.warning("_send_whatsapp_reply: to=%s status=%d body=%r",
                to, resp.status_code, resp.text[:200])
    except Exception as exc:
        log.exception("_send_whatsapp_reply: failed to=%s err=%s", to, exc)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log.info("starting lscm-agent on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
