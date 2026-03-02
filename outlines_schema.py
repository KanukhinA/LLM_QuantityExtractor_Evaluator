"""
JSON-схема для outlines (структурированная генерация).
Отдельный файл — один источник истины для передачи схемы в outlines как строки JSON.
Латиница в ключах (совместимость с токенизатором); при выводе ключи переводятся в кириллицу через structured_schemas.latin_to_cyrillic_output.
"""
import json
from typing import Any, Dict

# Схема совпадает по структуре с FertilizerExtractionOutput (mass_fractions, other_params; в Pydantic — alias «массовая доля», «прочее»).
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

# Схема с кириллическими ключами (для промптов DETAILED_INSTR_*_OUTLINES_RUS)
OUTLINES_SCHEMA_RUS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "массовая доля": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "вещество": {"type": "string"},
                    "массовая доля": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                        ]
                    },
                },
                "required": ["вещество", "массовая доля"],
            },
        },
        "прочее": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "параметр": {"type": "string"},
                    "масса": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                            {"type": "null"},
                        ]
                    },
                    "объем": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                            {"type": "null"},
                        ]
                    },
                    "количество": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": ["number", "null"]}},
                            {"type": "null"},
                        ]
                    },
                    "значение": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "null"},
                        ]
                    },
                    "единица": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["параметр"],
            },
        },
    },
    "required": ["массовая доля", "прочее"],
}


def get_outlines_schema_str(prompt_template_name: str = None) -> str:
    """
    Возвращает JSON-схему строкой с корректной кодировкой (ensure_ascii=False).
    Для промптов с суффиксом _RUS (напр. DETAILED_INSTR_ZEROSHOT_BASELINE_OUTLINES_RUS)
    возвращает схему с кириллическими ключами.
    """
    if prompt_template_name and prompt_template_name.endswith("_RUS"):
        return json.dumps(OUTLINES_SCHEMA_RUS, ensure_ascii=False, indent=2)
    return json.dumps(OUTLINES_SCHEMA, ensure_ascii=False, indent=2)


def get_outlines_schema_rus_str() -> str:
    """Схема с кириллическими ключами (для режима --guidance по умолчанию)."""
    return json.dumps(OUTLINES_SCHEMA_RUS, ensure_ascii=False, indent=2)
