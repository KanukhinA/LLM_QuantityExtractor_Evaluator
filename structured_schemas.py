"""
Pydantic схемы для structured output
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field


class MassDolyaItem(BaseModel):
    """Элемент массовой доли"""
    вещество: str = Field(description="Название вещества (например, N, P2O5, K2O)")
    массовая_доля: Union[float, List[Optional[float]]] = Field(
        alias="массовая доля",
        description="Массовая доля в процентах. Может быть числом или списком [min, max] для диапазонов"
    )
    
    class Config:
        populate_by_name = True  # Позволяет использовать как "массовая доля", так и "массовая_доля"


class ProcheeItem(BaseModel):
    """Элемент прочих параметров"""
    параметр: str = Field(description="Название параметра (например: масса нетто единицы, масса брутто, объем нетто единицы, количество поддонов, количество мешков, количество единиц, количество вагонов, количество биг-бэгов, стандарт, марка, концентрация N, концентрация P2O5 и т.д.)")
    масса: Optional[Union[float, List[Optional[float]]]] = Field(
        default=None,
        description="Масса в кг (для параметров: масса нетто единицы, масса брутто, масса нетто, масса п/э мешков, масса стрейч плёнки, масса брутто единицы)"
    )
    объем: Optional[Union[float, List[Optional[float]]]] = Field(
        default=None,
        description="Объем в л или мл (для параметров: объем нетто единицы)"
    )
    количество: Optional[Union[float, List[Optional[float]]]] = Field(
        default=None,
        description="Количество в шт (для параметров: количество поддонов, количество мешков, количество единиц, количество вагонов, количество биг-бэгов)"
    )
    значение: Optional[Union[str, float]] = Field(
        default=None,
        description="Текстовое значение для стандарта/марки (например: 'ТУ 2184-037-32496445-02', 'N7-P20-K30-S3') или числовое значение для концентраций (например: 9.12 для концентрация N в г/л)"
    )
    единица: Optional[str] = Field(
        default=None,
        description="Единица измерения: кг (для массы), л или мл (для объема), шт (для количества), г/л или мг/л (для концентраций), или отсутствует для стандарта и марки"
    )


class FertilizerExtractionOutput(BaseModel):
    """Структурированный вывод для извлечения данных об удобрениях"""
    массовая_доля: List[MassDolyaItem] = Field(
        alias="массовая доля",
        default_factory=list,
        description="Список веществ с их массовыми долями"
    )
    прочее: List[ProcheeItem] = Field(
        default_factory=list,
        description="Список прочих параметров (масса, объем, количество, стандарт, марка и т.д.)"
    )
    
    class Config:
        populate_by_name = True  # Позволяет использовать как "массовая доля", так и "массовая_доля"

