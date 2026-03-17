"""
Pydantic схемы для structured output
Латиница в именах полей — для совместимости с outlines (regex/токенизация).
Кириллица в alias и description — для вывода и валидации.
"""
import re
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


LATIN_TO_CYRILLIC_KEYS = {
    "mass_fractions": "массовая доля",
    "other_params": "прочее",
    "substance_name": "вещество",
    "mass_fraction": "массовая доля",
    "parameter_name": "параметр",
    "mass": "масса",
    "volume": "объем",
    "quantity": "количество",
    "value": "значение",
    "unit": "единица",
}


def latin_keys_to_cyrillic_in_json_str(s: str) -> str:
    """
    Заменяет латинские ключи JSON на кириллические в строке.
    Используется когда parse_json_safe не смог распарсить невалидный/обрезанный JSON —
    в этом случае заменяем ключи на уровне строки.
    Порядок замен: длинные ключи первыми, чтобы не заменить часть короткого.
    """
    if not s or not isinstance(s, str):
        return s
    # Сортируем по длине убывания: mass_fractions -> mass_fraction -> substance_name и т.д.
    sorted_items = sorted(LATIN_TO_CYRILLIC_KEYS.items(), key=lambda x: -len(x[0]))
    for latin_key, cyrillic_val in sorted_items:
        # Заменяем только ключи в JSON: "key": (с кавычками и двоеточием)
        pattern = r'"' + re.escape(latin_key) + r'"\s*:'
        replacement = f'"{cyrillic_val}":'
        s = re.sub(pattern, replacement, s)
    return s


def latin_to_cyrillic_output(obj: Any) -> Any:
    """Преобразует вывод outlines (латиница) в формат с кириллическими ключами для пайплайна."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            new_key = LATIN_TO_CYRILLIC_KEYS.get(k, k)
            result[new_key] = latin_to_cyrillic_output(v)
        return result
    elif isinstance(obj, list):
        return [latin_to_cyrillic_output(item) for item in obj]
    return obj


class MassDolyaItem(BaseModel):
    """Элемент массовой доли"""
    substance_name: str = Field(
        alias="вещество",
        description="Название вещества (например, N, P2O5, K2O)"
    )
    mass_fraction: Union[float, List[Optional[float]]] = Field(
        alias="массовая доля",
        description="Массовая доля в процентах. Число или список [min, max]"
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("mass_fraction", mode="before")
    @classmethod
    def _coerce_mass_fraction(cls, v):
        """
        Допускаем 0 и приводим строковые значения вида '0', '0.0', '0,0', '0%' к числу.
        В таможенных декларациях массовая доля 0 может встречаться.
        """
        def _one(x):
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                s = x.strip()
                if s.endswith("%"):
                    s = s[:-1].strip()
                s = s.replace(",", ".")
                return float(s)
            return x

        if isinstance(v, list):
            return [_one(x) for x in v]
        return _one(v)


class ProcheeItem(BaseModel):
    """Элемент прочих параметров"""
    parameter_name: str = Field(
        alias="параметр",
        description="Название параметра (масса нетто, объем, количество и т.д.)"
    )
    mass: Optional[Union[float, List[Optional[float]]]] = Field(
        default=None,
        alias="масса",
        description="Масса в кг"
    )
    volume: Optional[Union[float, List[Optional[float]]]] = Field(
        default=None,
        alias="объем",
        description="Объем в л или мл"
    )
    quantity: Optional[Union[float, List[Optional[float]]]] = Field(
        default=None,
        alias="количество",
        description="Количество в шт"
    )
    value: Optional[Union[str, float]] = Field(
        default=None,
        alias="значение",
        description="Текстовое или числовое значение"
    )
    unit: Optional[str] = Field(
        default=None,
        alias="единица",
        description="Единица измерения"
    )

    model_config = ConfigDict(populate_by_name=True)


class FertilizerExtractionOutput(BaseModel):
    """Структурированный вывод для извлечения данных об удобрениях"""
    mass_fractions: List[MassDolyaItem] = Field(
        alias="массовая доля",
        default_factory=list,
        description="Список веществ с их массовыми долями"
    )
    other_params: List[ProcheeItem] = Field(
        alias="прочее",
        default_factory=list,
        description="Список прочих параметров"
    )

    model_config = ConfigDict(populate_by_name=True)
