"""
Загрузчик конфигурации (API ключи)
"""
import os


def load_api_keys():
    """
    Загружает API ключи с приоритетом:
    1. Из config_secrets.py (если файл существует)
    2. Из переменных окружения
    
    Returns:
        tuple: (HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY)
    """
    HF_TOKEN = None
    GEMINI_API_KEY = None
    OPENAI_API_KEY = None
    
    # Приоритет 1: Пытаемся загрузить из config_secrets.py
    try:
        from config_secrets import (
            HF_TOKEN as _hf_token, 
            GEMINI_API_KEY as _gemini_key,
            OPENAI_API_KEY as _openai_key
        )
        if _hf_token and _hf_token.strip():
            HF_TOKEN = _hf_token.strip()
        if _gemini_key and _gemini_key.strip():
            GEMINI_API_KEY = _gemini_key.strip()
        if _openai_key and _openai_key.strip():
            OPENAI_API_KEY = _openai_key.strip()
    except ImportError:
        # Файл config_secrets.py не найден - это нормально, используем переменные окружения
        pass
    except Exception as e:
        # Если файл есть, но есть ошибка при импорте - выводим предупреждение
        print(f"Предупреждение: не удалось загрузить config_secrets.py: {e}")
        print("Будет использована переменная окружения, если она установлена.")
    
    # Приоритет 2: Если не загрузили из файла, пробуем переменные окружения
    if not HF_TOKEN:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if HF_TOKEN:
            HF_TOKEN = HF_TOKEN.strip()
    
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            GEMINI_API_KEY = GEMINI_API_KEY.strip()
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            OPENAI_API_KEY = OPENAI_API_KEY.strip()
    
    # Проверка обязательного HF_TOKEN
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN не установлен!\n"
            "Создайте файл config_secrets.py на основе config_secrets.py.example и заполните свои ключи,\n"
            "или установите переменную окружения:\n"
            "  Windows: set HF_TOKEN=your_token_here\n"
            "  Linux/Mac: export HF_TOKEN=your_token_here"
        )
    
    # Нормализуем GEMINI_API_KEY (пустая строка = None)
    if GEMINI_API_KEY == "":
        GEMINI_API_KEY = None
    
    # Нормализуем OPENAI_API_KEY (пустая строка = None)
    if OPENAI_API_KEY == "":
        OPENAI_API_KEY = None
    
    return HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY

