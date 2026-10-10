"""Bundle-local typed pseudonyms for public conversation evidence.

Collect before projecting: an identity discovered in a later lookup is also
replaced in the earlier request. Never serialize the reverse identity map.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

ALIAS = re.compile(r"\b(?:ORDER|ITEM|USER|PAYMENT|PRODUCT|EMAIL|PHONE|ADDRESS|NAME|TRACKING|ID)_[A-Z]+(?:\.(?:first|last))?\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\w)|"
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?\d{5}[ .-]\d{5}(?!\w)")
_TOKEN = re.compile(r"(?<![#\w])#\w+(?![#\w])|\b(?:credit_card|gift_card|paypal)_\w+\b|\b[a-z]+_[a-z]+_\d+\b")
_TYPED = re.compile(
    r"\b(order|item|product|user|customer|payment)(?:[ _-]ids?)?\s*[:=#]?\s+"
    r"(#[\w]+|\d{4,}|(?=[\w.-]*[A-Za-z])(?=[\w.-]*[\d_])[\w.-]+)(?!\w)",
    re.IGNORECASE)


def field_type(key: str) -> str | None:
    key = key.casefold()
    if key in {"email", "email_address"}:
        return "EMAIL"
    if "phone" in key:
        return "PHONE"
    if key in {"address", "address1", "address2", "street", "city", "state", "zip", "zip_code"}:
        return "ADDRESS"
    if key in {"product_name", "item_name", "variant_name"}:
        return None
    if key in {"first_name", "last_name"}:
        return "NAME"
    for word, kind in (("payment", "PAYMENT"), ("order", "ORDER"),
                       ("product", "PRODUCT"), ("item", "ITEM"), ("variant", "ITEM"),
                       ("user", "USER"), ("customer", "USER"), ("tracking", "TRACKING")):
        if word in key and (key.endswith(("id", "ids")) or
                            key in {"orders", "payment_methods", "variants"}):
            return kind
    return "ID" if key.endswith(("_id", "_ids")) else None


def child_context(key: str, context: str | None) -> str | None:
    if key in {"items", "products", "variants", "catalogue", "catalog"}:
        return "catalogue"
    if key in {"user", "users", "customer", "customers", "profile"}:
        return "person"
    return context


def public_name_field(key: str, record: dict, context: str | None = None) -> bool:
    return key in {"product_name", "item_name", "variant_name"} or (
        key == "name" and (context == "catalogue" or
                           any(k in record for k in ("product_id", "item_id", "variants"))))


def person_name_fields(record: dict, context: str | None = None) -> dict[str, str]:
    """Only declared person fields, never capitalisation or surrounding prose."""
    if context == "catalogue" or any(k in record for k in ("product_id", "item_id", "variants")):
        return {}
    person = context == "person" or any(k in record for k in (
        "user_id", "customer_id", "first_name", "last_name"))
    fields = {k: v for k, v in record.items() if isinstance(v, str) and v.strip() and not ALIAS.fullmatch(v) and (
        k in {"first_name", "last_name"} or person and k in {"first", "last", "name"})}
    return fields


def name_parts(fields: dict[str, str]) -> tuple[str, str, str]:
    first = fields.get("first_name", fields.get("first", ""))
    last = fields.get("last_name", fields.get("last", ""))
    full = fields.get("name", "") or (first + " " + last if first and last else "")
    if full and not first and not last:
        parts = full.split()
        first, last = parts[0], parts[-1] if len(parts) > 1 else ""
    return first, last, full


def _letters(number: int) -> str:
    out = ""
    while number:
        number, digit = divmod(number - 1, 26)
        out = chr(65 + digit) + out
    return out


class AliasTable:
    """One table per episode, shared by its dialogue, calls, results and thoughts.

    Schema field names are immutable. Dictionary keys inside identifier-indexed
    maps (variants/payment_methods) are *values*, and receive the same alias as
    the corresponding identifier field. List order and duplicates are retained.
    """

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._pattern: re.Pattern | None = None
        self._public_names: set[str] = set()
        self._name_records: dict[tuple[str, str, str], str] = {}
        self._name_owners: dict[str, str] = {}

    def register(self, value: Any, kind: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return
        raw = str(value)
        if not raw or raw in {"###STOP###", "###TRANSFER###"} or ALIAS.fullmatch(raw) or raw in self._values or raw in self._public_names:
            return
        self._counts[kind] += 1
        self._values[raw] = f"{kind}_{_letters(self._counts[kind])}"
        self._pattern = None
        if kind == "ORDER" and raw.startswith("#"):
            self._values.setdefault(raw[1:], self._values[raw])

    def _person_alias(self, fields: dict[str, str]) -> str | None:
        if not fields:
            return None
        key = name_parts(fields)
        first, last, full = key
        if key in self._name_records:
            return self._name_records[key]
        alias = None
        # A later full record can extend an established first/last mapping, but
        # never reuse another person's prefix merely because a surname matches.
        for raw in (full, first, last):
            existing = self._values.get(raw, "").split(".")[0]
            if re.fullmatch(r"NAME_[A-Z]+", existing) and (
                    not full or existing not in self._name_owners or self._name_owners[existing] == full):
                alias = existing
                break
        if alias is None:
            self._counts["NAME"] += 1
            alias = f"NAME_{_letters(self._counts['NAME'])}"
        self._name_records[key] = alias
        if full:
            self._name_owners.setdefault(alias, full)
        for raw, replacement in ((full, alias), (first, alias + ".first"), (last, alias + ".last")):
            if raw and raw not in self._public_names:
                self._values.setdefault(raw, replacement)
        self._pattern = None
        return alias

    def prime(self, value: Any, kind: str | None = None, *, indexed: bool = False,
              context: str | None = None) -> None:
        """Collect full person records before their possibly ambiguous parts."""
        partials = []
        self._prime(value, kind, indexed=indexed, context=context, partials=partials)
        for fields in partials:
            self._person_alias(fields)

    def _prime(self, value: Any, kind: str | None = None, *, indexed: bool = False,
               context: str | None = None, partials: list[dict[str, str]]) -> None:
        if isinstance(value, str) and kind is None:
            try:
                parsed = json.loads(value)
            except ValueError:
                return
            if isinstance(parsed, (dict, list)):
                self._prime(parsed, context=context, partials=partials)
            return
        if isinstance(value, dict):
            fields = person_name_fields(value, context)
            if name_parts(fields)[2]:
                self._person_alias(fields)
            elif fields:
                partials.append(fields)
            for key, child in value.items():
                if public_name_field(key, value, context):
                    if isinstance(child, str):
                        self._public_names.add(child)
                        self._pattern = None
                    continue
                if key == "name" and "arguments" in value:
                    continue
                if key == "content" and value.get("role") in {"user", "assistant"}:
                    continue  # Person discovery uses tool records, not dialogue JSON.
                if indexed and kind:
                    self.register(key, kind)
                nested_context = child_context(key, context)
                if key == "name" and isinstance(child, dict) and any(
                        k in child for k in ("first", "last", "first_name", "last_name")):
                    nested_context = "person"
                child_kind = None if indexed else field_type(key)
                if child_kind == "NAME":
                    child_kind = None  # Person fields were registered together above.
                self._prime(child, child_kind or (kind if kind == "ADDRESS" else None),
                            indexed=key in {"variants", "payment_methods"}, context=nested_context,
                            partials=partials)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._prime(child, kind, context=context, partials=partials)
        elif kind:
            self.register(value, kind)

    def collect_text(self, text: str) -> None:
        try:
            structured = json.loads(text)
        except ValueError:
            structured = None
        if isinstance(structured, (dict, list)):
            self._collect(structured)
            return
        for pattern, kind in ((_EMAIL, "EMAIL"), (_PHONE, "PHONE")):
            for match in pattern.finditer(text):
                self.register(match.group(), kind)
        for match in _TOKEN.finditer(text):
            raw = match.group()
            kind = "ORDER" if raw.startswith("#") else (
                "PAYMENT" if raw.startswith(("credit_card_", "gift_card_", "paypal_")) else "USER")
            self.register(raw, kind)
        for match in _TYPED.finditer(text):
            raw = match.group(2).rstrip(".")
            # Only identifier-shaped values; do not alias 'order status'.
            if raw in self._values or re.search(r"\d{4,}|[A-Za-z]\w*[_#]\w+|#\w+|[A-Za-z]+\d+", raw):
                kind = {"customer": "USER"}.get(match.group(1).lower(), match.group(1).upper())
                self.register(raw, kind)

    def collect(self, value: Any, kind: str | None = None, *, indexed: bool = False) -> None:
        self.prime(value, kind, indexed=indexed)
        self._collect(value, kind, indexed=indexed)

    def _collect(self, value: Any, kind: str | None = None, *, indexed: bool = False,
                 context: str | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if public_name_field(key, value, context) or key == "name":
                    continue  # Names are discovered only in the structured priming pass.
                if indexed and kind:
                    self.register(key, kind)
                child_kind = None if indexed else field_type(key)
                if child_kind == "NAME":
                    continue
                self._collect(child, child_kind or (kind if kind == "ADDRESS" else None),
                              indexed=key in {"variants", "payment_methods"},
                              context=child_context(key, context))
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._collect(child, kind, context=context)
        elif kind:
            self.register(value, kind)
        elif isinstance(value, str):
            self.collect_text(value)

    def text(self, text: str) -> str:
        values = self._values.keys() | self._public_names
        if not self._values:
            return text
        # Simultaneous substitution prevents cascading, overlapping or substring aliases.
        if self._pattern is None:
            self._pattern = re.compile(r"(?<!\w)(?:" + "|".join(
                re.escape(v) for v in sorted(values, key=lambda v: (-len(v), v))) + r")(?!\w)")
        return self._pattern.sub(lambda m: (m.group() if m.group() in self._public_names
                                            else self._values[m.group()]), text)

    def project(self, value: Any, kind: str | None = None, *, indexed: bool = False,
                context: str | None = None) -> Any:
        if isinstance(value, dict):
            fields = person_name_fields(value, context)
            person_alias = self._name_records.get(name_parts(fields))
            result = {}
            for key, child in value.items():
                output_key = self._values.get(key, key) if indexed else key
                if public_name_field(key, value, context) or key == "name" and "arguments" in value:
                    result[output_key] = child
                elif key in fields and person_alias:
                    suffix = ".first" if key in {"first", "first_name"} else (
                        ".last" if key in {"last", "last_name"} else "")
                    result[output_key] = person_alias + suffix
                else:
                    nested_context = child_context(key, context)
                    if key == "name" and isinstance(child, dict) and any(
                            k in child for k in ("first", "last", "first_name", "last_name")):
                        nested_context = "person"
                    result[output_key] = self.project(child, None if indexed else field_type(key),
                        indexed=key in {"variants", "payment_methods"}, context=nested_context)
            return result
        if isinstance(value, (list, tuple)):
            return [self.project(child, kind, context=context) for child in value]
        if isinstance(value, str):
            try:
                structured = json.loads(value)
            except ValueError:
                structured = None
            if isinstance(structured, (dict, list)):
                return json.dumps(self.project(structured, context=context), ensure_ascii=True, separators=(",", ":"))
            return self.text(value)
        if kind and not isinstance(value, bool) and isinstance(value, (int, float)):
            return self._values.get(str(value), value)
        return value


def alias_bundle(value: Any) -> Any:
    table = AliasTable()
    table.collect(value)
    return table.project(value)
