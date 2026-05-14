# order_log.py
"""Append parsed orders to the 'Order Queue' tab of the master Sheet."""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("order_log")
SCOPE = "https://www.googleapis.com/auth/spreadsheets"
TAB_NAME = "Order Queue"


def append_order(msg_id: str, from_number: str, raw_message: str, parsed: dict) -> None:
    """Append one row. Fail-soft: log on error, do not raise."""
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
    if not (sheet_id and creds_json):
        log.warning("order_log: Sheet env vars not set, skipping persistence")
        return
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_info(
            json.loads(creds_json), scopes=[SCOPE]
        )
        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        cust = parsed.get("customer_match") or {}
        row = [
            datetime.now(timezone.utc).isoformat(),
            msg_id,
            from_number,
            raw_message[:500],
            cust.get("id", ""),
            cust.get("name", ""),
            cust.get("company", ""),
            cust.get("area", ""),
            cust.get("pincode", ""),
            parsed.get("intent", ""),
            parsed.get("confidence", ""),
            "TRUE" if parsed.get("needs_human_review") else "FALSE",
            json.dumps(parsed.get("items", []), ensure_ascii=False),
            parsed.get("delivery_when", ""),
            parsed.get("amount_mentioned", ""),
            (parsed.get("notes", "") or "")[:300],
        ]
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range=f"'{TAB_NAME}'!A:P",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
        log.info("order_log: appended order msg_id=%s customer=%s", msg_id, cust.get("name", "?"))
    except Exception as e:
        log.exception("order_log: append failed (%s)", e)
