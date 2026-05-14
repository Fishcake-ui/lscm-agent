"""
sheets_loader.py - Load the customer master from a Google Sheet at startup.

Source resolution order:
  1. Google Sheets, when GOOGLE_SHEET_ID and GOOGLE_SHEETS_CREDENTIALS_JSON are set.
     GOOGLE_SHEETS_CREDENTIALS_JSON contains the full service-account JSON as a
     string (not a file path) so it works on Render without file uploads.
  2. Local customers.json, as a fallback for offline dev or when the Sheet is
     unreachable.

The Sheet must be shared (Viewer or higher) with the service account's
client_email, e.g. lscm-sheets-agent@<project>.iam.gserviceaccount.com.

Sheet shape assumed:
  - Data lives in the FIRST tab of the spreadsheet (whatever its name is).
  - Row 1 = column headers.
  - Rows 2..N = customer records.

Header-to-canonical-field mapping (case-insensitive, first match wins) lives in
HEADER_ALIASES below; the actual mapping resolved against the live Sheet is
logged at startup so it's easy to verify what columns were picked up.
"""

import json
import logging
import os
import re
from typing import Any, Optional

log = logging.getLogger("sheets_loader")

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_PATH = os.path.join(SCRIPT_DIR, "customers.json")

# Canonical field -> ordered list of header strings to try (case-insensitive,
# whitespace-trimmed). First header found in the sheet wins for that field.
HEADER_ALIASES: dict[str, list[str]] = {
    "customer_id":         ["customer id", "cust id", "id"],
    "name":                ["delivery destination", "customer name", "name", "account name"],
    "address":             ["address", "delivery address", "full address"],
    "phone":               ["phone", "phone number", "mobile", "contact"],
    "whatsapp_number":     ["whatsapp", "whatsapp number", "wa number", "wa"],
    "credit_limit":        ["credit limit", "limit", "credit"],
    "outstanding_balance": ["outstanding balance", "outstanding", "balance", "due"],
    "area":                ["area / locality", "area", "locality"],
    "pincode":             ["pincode", "pin", "postal code", "zip"],
    "match_status":        ["match status", "status"],
    "abir_tally":          ["abir tally name", "abir name"],
    "js_tally":            ["james smith tally name", "js name", "james smith name"],
}

# Alias-string generation (mirrors build_customers.py so fuzzy match keeps working).
_NOISE = re.compile(
    r"\b(pvt|private|limited|ltd|llp|inc|co|company|the|hotel|hotels|"
    r"restaurant|restaurants|services|hospitality|enterprises|industries)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[\(\)\[\]\.,&/\-_]+")


def _gen_aliases(dest: Optional[str], abir_tally: Optional[str], js_tally: Optional[str]) -> list[str]:
    raw = [s for s in (dest, abir_tally, js_tally) if s]
    out: set[str] = set()
    for s in raw:
        s = str(s).strip()
        out.add(s.lower())
        head = s.split(",")[0].strip()
        out.add(head.lower())
        cleaned = _PUNCT.sub(" ", _NOISE.sub(" ", head))
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        if cleaned:
            out.add(cleaned)
        words = [w for w in cleaned.split() if len(w) > 1]
        if len(words) >= 2:
            out.add(" ".join(words[:2]))
        if words:
            out.add(words[0])
    out.discard((dest or "").lower())
    return sorted(a for a in out if a and len(a) >= 2 and not a.isdigit())


def _build_header_index(headers: list[str]) -> dict[str, int]:
    """Resolve canonical_field -> column_index using HEADER_ALIASES."""
    norm = [(h or "").strip().lower() for h in headers]
    mapping: dict[str, int] = {}
    for canonical, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in norm:
                mapping[canonical] = norm.index(a)
                break
    return mapping


def _cell(row: list, headers: list[str], idx: dict[str, int], canonical: str) -> Any:
    i = idx.get(canonical)
    if i is None or i >= len(row):
        return None
    v = row[i]
    return v if v not in ("", None) else None


def _normalize_row(row: list, headers: list[str], idx: dict[str, int], counters: dict) -> Optional[dict]:
    """Build one customer dict from a sheet row. Returns None if row has no name."""
    name = _cell(row, headers, idx, "name")
    if not name:
        return None

    status = _cell(row, headers, idx, "match_status") or ""
    if "Abir Only" in status:
        counters["ab"] += 1
        gen_id = f"AF-{counters['ab']:03d}"
        company = "Abir Foods"
    elif "James Smith Only" in status:
        counters["js"] += 1
        gen_id = f"JS-{counters['js']:03d}"
        company = "James Smith"
    else:
        counters["both"] += 1
        gen_id = f"BOTH-{counters['both']:03d}"
        company = "Both"

    cust_id = _cell(row, headers, idx, "customer_id") or gen_id
    phone = _cell(row, headers, idx, "phone")
    # India: a hotel's contact number is virtually always WhatsApp-enabled. If
    # the sheet has a dedicated WhatsApp column, use it; otherwise default to phone.
    wa = _cell(row, headers, idx, "whatsapp_number") or phone

    # Preserve raw row keyed by header for debugging / future-proofing.
    raw: dict[str, Any] = {}
    for i, h in enumerate(headers):
        if h and i < len(row):
            raw[h] = row[i]

    pincode_raw = _cell(row, headers, idx, "pincode")

    return {
        # Canonical fields (the contract expected by sheets_loader callers)
        "customer_id":         str(cust_id),
        "name":                str(name),
        "address":             str(_cell(row, headers, idx, "address") or ""),
        "phone":               phone,
        "whatsapp_number":     wa,
        "credit_limit":        _cell(row, headers, idx, "credit_limit"),
        "outstanding_balance": _cell(row, headers, idx, "outstanding_balance"),

        # Fields parse_order.lookup_customer reads (keep these in sync)
        "id":            str(cust_id),
        "aliases":       _gen_aliases(name, _cell(row, headers, idx, "abir_tally"),
                                      _cell(row, headers, idx, "js_tally")),
        "company":       company,
        "area":          str(_cell(row, headers, idx, "area") or ""),
        "pincode":       str(pincode_raw) if pincode_raw else "",
        "match_status":  status,
        "abir_tally":    _cell(row, headers, idx, "abir_tally"),
        "js_tally":      _cell(row, headers, idx, "js_tally"),
        "credit_days":   30,

        # Raw row, header-keyed, for inspection.
        "_raw": raw,
    }


def _load_from_sheet(sheet_id: str, creds_json: str) -> list[dict]:
    """Pull rows from the first tab of the Google Sheet and normalize them."""
    # Imported lazily so the module remains importable without google libs.
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=[SHEETS_SCOPE]
    )
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)

    meta = service.spreadsheets().get(spreadsheetId=sheet_id, includeGridData=False).execute()
    sheets = meta.get("sheets") or []
    if not sheets:
        log.error("Spreadsheet %s has no tabs.", sheet_id)
        return []
    first_tab = sheets[0]["properties"]["title"]
    log.info("Reading first tab: %r", first_tab)

    resp = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=sheet_id, range=f"'{first_tab}'!A1:Z")
        .execute()
    )
    rows: list[list] = resp.get("values", [])
    if len(rows) < 2:
        log.warning("Sheet %r has fewer than 2 rows (need header + data).", first_tab)
        return []

    headers = rows[0]
    idx = _build_header_index(headers)
    log.info(
        "Header mapping resolved: %s",
        {canonical: headers[col] for canonical, col in idx.items()},
    )
    if "name" not in idx:
        log.error("No customer-name column found in headers: %s", headers)
        return []

    counters = {"ab": 0, "js": 0, "both": 0}
    customers: list[dict] = []
    for r in rows[1:]:
        c = _normalize_row(r, headers, idx, counters)
        if c:
            customers.append(c)
    return customers


def _load_from_local(path: str) -> list[dict]:
    if not os.path.exists(path):
        log.warning("Local customers file not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("customers", []) if isinstance(data, dict) else []
    # Keep the canonical-field contract identical to the Sheets path, even
    # though local customers.json was built by build_customers.py and uses 'id'.
    for c in raw:
        c.setdefault("customer_id", c.get("id"))
        c.setdefault("whatsapp_number", c.get("phone"))
        c.setdefault("credit_limit", None)
        c.setdefault("outstanding_balance", None)
    return raw


def load_customers(local_path: str = DEFAULT_LOCAL_PATH) -> list[dict]:
    """
    Load the customer master. Tries Google Sheets first when env vars are set,
    falls back to the local customers.json on any failure.

    Always logs the source and the count.
    """
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")

    if sheet_id and creds_json:
        try:
            customers = _load_from_sheet(sheet_id, creds_json)
            if customers:
                log.info("Loaded %d customers from Google Sheet %s", len(customers), sheet_id)
                return customers
            log.warning("Google Sheet returned 0 usable rows; falling back to local file.")
        except Exception as e:
            log.exception("Sheets load failed (%s: %s); falling back to local file.",
                          type(e).__name__, e)
    else:
        missing = [
            k for k, v in (
                ("GOOGLE_SHEET_ID", sheet_id),
                ("GOOGLE_SHEETS_CREDENTIALS_JSON", creds_json),
            ) if not v
        ]
        log.info("Sheets env vars not set (%s); using local fallback.", ", ".join(missing))

    customers = _load_from_local(local_path)
    log.info("Loaded %d customers from local file %s", len(customers), local_path)
    return customers


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cs = load_customers()
    print(f"\nTotal customers: {len(cs)}")
    if cs:
        sample = cs[0]
        print("First customer (canonical fields):")
        for k in ("customer_id", "name", "address", "phone",
                  "whatsapp_number", "credit_limit", "outstanding_balance"):
            print(f"  {k}: {sample.get(k)!r}")
