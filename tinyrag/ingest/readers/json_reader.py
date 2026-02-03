from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from .common import read_text_file


def read_json_file(path: Path) -> Any:
    return json.loads(read_text_file(path))


def read_jsonl_file(path: Path) -> List[Any]:
    out: List[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_texts_from_json_obj(obj: Any, *, text_key: str) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        out: List[str] = []
        for item in obj:
            out.extend(extract_texts_from_json_obj(item, text_key=text_key))
        return out
    if isinstance(obj, dict):
        val = obj.get(text_key)
        if isinstance(val, str):
            return [val]
    return []

