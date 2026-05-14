"""
main.py - FastAPI entrypoint for the LSCM order agent.
Render-deployable: binds 0.0.0.0:$PORT, exposes /health for platform checks.

Routes:
  GET  /                      - service info
  GET  /health                - Render health probe
  POST /parse                 - parse one order message (test / debug)
  POST /webhook/whatsapp      - WhatsApp BSP webhook (placeholder; logs + ACKs)

Local run:
  python main.py
  # or, with reload during dev:
  # uvicorn main:app --reload --port 10000
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lscm")

# Importing parse_order triggers the customer-master load (Sheets or local).
# The loader logs its source + count, so by the time the app boots we already
# know whether we're running on real data.
from parse_order import (  # noqa: E402  (intentional: log basicConfig first)
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


@app.post("/webhook/whatsapp")
async def webhook_whatsapp(request: Request) -> dict[str, str]:
    """
    WhatsApp BSP webhook. Currently a placeholder: logs payload and ACKs 200
    so the BSP does not retry. Real handling (extract message, dispatch to
    parse_order, persist to Sheet, reply via BSP API) lands in a later phase.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {"_raw": (await request.body()).decode("utf-8", errors="replace")}
    log.info("webhook/whatsapp received: %r", payload)
    return {"status": "received"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log.info("starting lscm-agent on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
