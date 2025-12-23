"""
Core модуль с ООП архитектурой для оценки моделей
"""
from .base import BaseGenerator
from .generators import StandardGenerator

__all__ = [
    "BaseGenerator",
    "StandardGenerator",
]

