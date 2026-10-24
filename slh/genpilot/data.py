"""Read-only helpers over the public retail database (``db.json``)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RETAIL = REPO / "vendor/tau2-bench/data/tau2/domains/retail"
DB_PATH = RETAIL / "db.json"
OFFICIAL_TASKS = RETAIL / "tasks.json"
GENPILOT = REPO / "genpilot"
ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def json_sha(value: Any) -> str:
    return sha256_bytes(canonical(value))


def load_db(path: Path = DB_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def addr_key(address: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(address.get(f, "")) for f in ADDRESS_FIELDS)


def addr_args(address: dict[str, Any]) -> dict[str, str]:
    return {f: str(address.get(f, "")) for f in ADDRESS_FIELDS}


def addr_text(address: dict[str, Any]) -> str:
    line = address["address1"] + (f", {address['address2']}" if address.get("address2") else "")
    return f"{line}, {address['city']}, {address['state']} {address['zip']}, {address['country']}"


def user_orders(db: dict[str, Any], user: dict[str, Any]) -> list[dict[str, Any]]:
    return [db["orders"][o] for o in user["orders"] if o in db["orders"]]


def product_names(order: dict[str, Any]) -> list[str]:
    return [item["name"] for item in order["items"]]


def original_payment(order: dict[str, Any]) -> str | None:
    payments = [p for p in order["payment_history"] if p["transaction_type"] == "payment"]
    return payments[0]["payment_method_id"] if payments else None


def pm_desc(pm: dict[str, Any]) -> str:
    source = pm["source"]
    if source == "credit_card":
        return f"{str(pm.get('brand', 'credit')).title()} credit card ending in {pm['last_four']}"
    if source == "paypal":
        return "PayPal account"
    if source == "gift_card":
        return "gift card"
    raise ValueError(f"unknown payment source {source!r}")


def pm_by_desc(user: dict[str, Any], desc: str) -> list[str]:
    """Payment method ids of ``user`` that the description names."""
    return sorted(pid for pid, pm in user["payment_methods"].items() if pm_desc(pm) == desc)


def pm_unique(user: dict[str, Any], pm_id: str) -> bool:
    return pm_by_desc(user, pm_desc(user["payment_methods"][pm_id])) == [pm_id]


def matches(options: dict[str, str], constraints: dict[str, str]) -> bool:
    return all(options.get(k) == v for k, v in constraints.items())


def product_by_id(db: dict[str, Any], product_id: str) -> dict[str, Any]:
    return db["products"][product_id]


def money(value: float) -> str:
    return f"${value:,.2f}"


def ceil_dollar(value: float) -> int:
    return math.ceil(value - 1e-9)


def mentioned_users(db: dict[str, Any], text: str) -> set[str]:
    """User ids named in ``text`` by id, email, full name or id stem."""
    found = set()
    for uid, user in db["users"].items():
        name = f"{user['name']['first_name']} {user['name']['last_name']}"
        if uid in text or user["email"] in text or name in text or \
                uid.rsplit("_", 1)[0] in text:
            found.add(uid)
    return found


def public_scenario_text(path: Path = OFFICIAL_TASKS) -> str:
    """The official tasks' PUBLIC user scenarios only (never their evaluation
    criteria): the generator's exclusion list is built from this text."""
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(json.dumps(t.get("user_scenario"), sort_keys=True) for t in tasks)
