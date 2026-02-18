"""
Pydantic схемы для structured output
Латиница в именах полей — для совместимости с outlines (regex/токенизация).
Кириллица в alias и description — для вывода и валидации.
"""
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict


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


# Схема только с латиницей — для outlines (без alias, JSON schema с Latin keys)
class MassDolyaItemLatin(BaseModel):
    substance_name: str = Field(description="Название вещества (N, P2O5, K2O)")
    mass_fraction: Union[float, List[Optional[float]]] = Field(
        description="Массовая доля в процентах. Число или [min, max]"
    )


class ProcheeItemLatin(BaseModel):
    parameter_name: str = Field(description="Название параметра")
    mass: Optional[Union[float, List[Optional[float]]]] = Field(default=None, description="Масса в кг")
    volume: Optional[Union[float, List[Optional[float]]]] = Field(default=None, description="Объем в л или мл")
    quantity: Optional[Union[float, List[Optional[float]]]] = Field(default=None, description="Количество в шт")
    value: Optional[Union[str, float]] = Field(default=None, description="Текстовое или числовое значение")
    unit: Optional[str] = Field(default=None, description="Единица измерения")


class FertilizerExtractionOutputLatin(BaseModel):
    """Только латинские ключи — для outlines (совместимость с токенизатором)."""
    mass_fractions: List[MassDolyaItemLatin] = Field(
        default_factory=list,
        description="Список веществ с их массовыми долями"
    )
    other_params: List[ProcheeItemLatin] = Field(
        default_factory=list,
        description="Список прочих параметров"
    )
