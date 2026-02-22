import json
from typing import Any, Dict

OUTLINES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mass_fractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "substance_name": {"type": "string"},
                    "mass_fraction": {"type": "string"}  # ← Упрощено до строки
                },
                "required": ["substance_name", "mass_fraction"],
                "additionalProperties": False
            },
        },
        "other_params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter_name": {"type": "string"},
                    "mass": {"type": "string"},      # ← Упрощено
                    "volume": {"type": "string"},    # ← Упрощено
                    "quantity": {"type": "string"},  # ← Упрощено
                    "value": {"type": "string"},     # ← Упрощено
                    "unit": {"type": "string"}       # ← Упрощено
                },
                "required": ["parameter_name"],
                "additionalProperties": False
            },
        },
    },
    "required": ["mass_fractions", "other_params"],
    "additionalProperties": False
}


def get_outlines_schema_str() -> str:
    """Возвращает JSON-схему строкой."""
    return json.dumps(OUTLINES_SCHEMA, ensure_ascii=False, indent=2)