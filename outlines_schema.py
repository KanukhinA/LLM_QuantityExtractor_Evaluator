"""
JSON-схема для outlines (структурированная генерация).
Отдельный файл — один источник истины для передачи схемы в outlines как строки JSON.
Латиница в ключах (совместимость с токенизатором); при выводе ключи переводятся в кириллицу через structured_schemas.latin_to_cyrillic_output.
"""
import json
from typing import Any, Dict

# Схема совпадает по структуре с FertilizerExtractionOutputLatin (mass_fractions, other_params).
OUTLINES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mass_fractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "substance_name": {"type": "string"},
                    "mass_fraction": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                        ]
                    },
                },
                "required": ["substance_name", "mass_fraction"],
            },
        },
        "other_params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter_name": {"type": "string"},
                    "mass": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                            {"type": "null"},
                        ]
                    },
                    "volume": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                            {"type": "null"},
                        ]
                    },
                    "quantity": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                            {"type": "null"},
                        ]
                    },
                    "value": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "null"},
                        ]
                    },
                    "unit": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["parameter_name"],
            },
        },
    },
    "required": ["mass_fractions", "other_params"],
}


def get_outlines_schema_str() -> str:
    """Возвращает JSON-схему строкой с корректной кодировкой (ensure_ascii=False)."""
    return json.dumps(OUTLINES_SCHEMA, ensure_ascii=False, indent=2)
