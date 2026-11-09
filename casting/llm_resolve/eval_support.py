# -*- coding: utf-8 -*-
"""评测辅助：载入标注、按 needle 定位段落。"""
from __future__ import annotations

import json
from pathlib import Path


def load_annotations(path):
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {"cases": data}


def resolve_annotation_needle(paragraphs, needle, expected_index=None):
    """优先用标注下标；若下标漂移，则按 needle 线性查找。"""
    if not needle:
        return expected_index if expected_index is not None else -1
    if (expected_index is not None
            and 0 <= expected_index < len(paragraphs)
            and needle in (paragraphs[expected_index] or "")):
        return expected_index
    for i, para in enumerate(paragraphs):
        if para and needle in para:
            return i
    return -1
