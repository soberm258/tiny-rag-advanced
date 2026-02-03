from __future__ import annotations

from typing import Any


def to_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or "")
    return str(item or "")


def item_key(item: Any) -> str:
    if isinstance(item, dict):
        item_id = item.get("id")
        if item_id:
            return f"id:{item_id}"
        meta = item.get("meta") or {}
        if isinstance(meta, dict):
            doc_id = meta.get("doc_id")
            if doc_id:
                return f"doc_id:{doc_id}"
        return f"text:{to_text(item)}"
    return f"text:{to_text(item)}"

