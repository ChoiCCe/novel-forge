"""剧本片段的数据模型与 JSON Schema。

独立成模块，供 script / coverage 共用，避免二者之间形成循环导入。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    type: str        # narration | dialogue
    speaker: str
    emotion: str
    text: str
    repaired: bool = False

    def to_dict(self) -> dict:
        data = {"type": self.type, "speaker": self.speaker, "emotion": self.emotion, "text": self.text}
        if self.repaired:
            data["repaired"] = True
        return data


SEGMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["narration", "dialogue"]},
                    "speaker": {"type": "string"},
                    "emotion": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["type", "text"],
            },
        }
    },
    "required": ["segments"],
}
