"""
Функции для работы с API моделями (Gemma 3 через Google Generative AI API)
"""
import os
import time
import re
from typing import Tuple, Any, Optional
from config import GEMINI_API_KEY

# Импорт для API моделей
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


# ============================================================================
# API модели (Gemma 3 через Google Generative AI API)
# ============================================================================

def load_gemma_3_4b_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Загрузка gemma-3-4b-it через API (возвращает клиент API вместо модели)"""
    if not GENAI_AVAILABLE:
        raise ImportError("Библиотека google-genai не установлена. Установите: pip install google-genai")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    
    print(f"   Инициализация API клиента для gemma-3-4b-it...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"   ✓ API клиент инициализирован")
        # Для API моделей возвращаем клиент вместо модели/tokenizer
        return client, None
    except Exception as e:
        print(f"   ❌ Ошибка инициализации API клиента: {e}")
        raise


def load_gemma_3_12b_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Загрузка gemma-3-12b-it через API (возвращает клиент API вместо модели)"""
    if not GENAI_AVAILABLE:
        raise ImportError("Библиотека google-genai не установлена. Установите: pip install google-genai")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    
    print(f"   Инициализация API клиента для gemma-3-12b-it...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"   ✓ API клиент инициализирован")
        return client, None
    except Exception as e:
        print(f"   ❌ Ошибка инициализации API клиента: {e}")
        raise


def load_gemma_3_27b_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Загрузка gemma-3-27b-it через API (возвращает клиент API вместо модели)"""
    if not GENAI_AVAILABLE:
        raise ImportError("Библиотека google-genai не установлена. Установите: pip install google-genai")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    
    print(f"   Инициализация API клиента для gemma-3-27b-it...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"   ✓ API клиент инициализирован")
        return client, None
    except Exception as e:
        print(f"   ❌ Ошибка инициализации API клиента: {e}")
        raise


def generate_gemma_api(
    client, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 1024, 
    model_name: str = "gemma-3-4b-it",
    repetition_penalty: float = None
) -> str:
    """
    Функция генерации для Gemma 3 моделей через API с retry логикой до 10 попыток
    
    Args:
        client: клиент API (genai.Client)
        tokenizer: не используется, но нужен для совместимости сигнатуры (может быть None)
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        model_name: имя модели API (gemma-3-4b, gemma-3-12b, gemma-3-27b)
        repetition_penalty: штраф за повторения (если None, не используется, для API не поддерживается напрямую)
    
    Returns:
        сгенерированный текст
    """
    # Параметры для разных попыток (до 10 попыток с разными параметрами)
    retry_configs = [
        {"temperature": 0.0, "top_p": None, "top_k": None},  # Попытка 1: детерминированная
        {"temperature": 0.1, "top_p": None, "top_k": None},  # Попытка 2: немного случайности
        {"temperature": 0.0, "top_p": 0.95, "top_k": 40},    # Попытка 3: с top_p/top_k
        {"temperature": 0.2, "top_p": 0.9, "top_k": 50},    # Попытка 4: более высокая температура
        {"temperature": 0.0, "top_p": 0.99, "top_k": None}, # Попытка 5: только top_p
        {"temperature": 0.15, "top_p": None, "top_k": 40},  # Попытка 6: только top_k
        {"temperature": 0.0, "top_p": 0.9, "top_k": None},  # Попытка 7: низкая температура с top_p
        {"temperature": 0.1, "top_p": 0.95, "top_k": 50},   # Попытка 8: комбинация параметров
        {"temperature": 0.05, "top_p": None, "top_k": None}, # Попытка 9: очень низкая температура
        {"temperature": 0.0, "top_p": 1.0, "top_k": None},  # Попытка 10: детерминированная с top_p=1.0
    ]
    
    last_error = None
    
    for attempt in range(10):
        config = retry_configs[attempt]
        
        try:
            # Формируем параметры запроса
            generation_config = {
                "max_output_tokens": max_new_tokens,
                "temperature": config["temperature"],
            }
            
            # Добавляем опциональные параметры
            if config["top_p"] is not None:
                generation_config["top_p"] = config["top_p"]
            if config["top_k"] is not None:
                generation_config["top_k"] = config["top_k"]
            
            # Выполняем запрос к API
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generation_config
            )
            
            # Извлекаем текст ответа
            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                text = response.candidates[0].content.parts[0].text
            else:
                text = str(response)
            
            # Если получили текст, возвращаем его
            if text and text.strip():
                return text.strip()
            else:
                # Пустой ответ - пробуем следующую попытку
                last_error = "Пустой ответ от API"
                continue
                
        except Exception as e:
            last_error = str(e)
            print(last_error)
            error_str = str(e).lower()
            
            # Проверяем, является ли это ошибкой 404 (модель не найдена)
            # В таком случае не имеет смысла повторять запросы
            if "404" in error_str or "not found" in error_str or "model not found" in error_str:
                raise Exception(f"Модель не найдена (404): {last_error}")
            
            # Проверяем, является ли это ошибкой 429 (rate limit / resource exhausted)
            # Для free tier нужно увеличить задержку между запросами
            is_rate_limit = (
                "429" in error_str or 
                "resource_exhausted" in error_str or 
                "rate limit" in error_str or
                "quota exceeded" in error_str
            )
            
            # Если это последняя попытка, пробрасываем исключение
            if attempt == 9:
                raise Exception(f"Ошибка после 10 попыток. Последняя ошибка: {last_error}")
            
            # Для rate limit ошибок пытаемся извлечь требуемое время ожидания из ошибки
            if is_rate_limit:
                delay = None
                
                # Пытаемся найти время ожидания в тексте ошибки
                # Ищем паттерны типа "retry after 30s", "wait 60 seconds", "retry_after: 45", "Please retry in 12.12324s" и т.д.
                # Паттерны для поиска времени ожидания (в секундах) - поддерживают целые и десятичные числа
                patterns = [
                    r'retry[_\s]?after[_\s]?:?\s*(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)',  # "retry after 30s", "retry_after: 45 seconds"
                    r'wait[_\s]+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)',                    # "wait 60 seconds"
                    r'retry[_\s]?in[_\s]+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)',          # "retry in 30 seconds", "Please retry in 12.12324s"
                    r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s+.*?(?:retry|wait)',           # "30 seconds before retry"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, last_error, re.IGNORECASE)
                    if match:
                        try:
                            extracted_delay = float(match.group(1))
                            # Проверяем разумность значения (максимум 3600 секунд = 1 час)
                            # Время ожидания обычно не превышает часа
                            if 1 <= extracted_delay <= 3600:
                                delay = extracted_delay
                                break
                        except (ValueError, IndexError):
                            continue
                
                # Если не удалось извлечь время из ошибки, используем экспоненциальную задержку
                if delay is None:
                    # Экспоненциальная задержка с большим базовым временем для free tier
                    # Начинаем с 5 секунд и увеличиваем экспоненциально
                    delay = min(5.0 * (2 ** attempt), 60.0)  # Максимум 60 секунд
                    print(f"   ⚠️ Rate limit (429). Время ожидания не указано. Используем {delay:.1f} секунд перед попыткой {attempt + 2}/10...")
                else:
                    # Добавляем небольшую буферную задержку к указанному времени (10%)
                    delay = delay * 1.1
                    print(f"   ⚠️ Rate limit (429). API требует ожидания {delay:.1f} секунд. Ожидание перед попыткой {attempt + 2}/10...")
                
                time.sleep(delay)
            else:
                # Для других ошибок используем меньшую задержку
                delay = 0.5 * (attempt + 1)
                time.sleep(delay)
    
    # Если дошли сюда, все попытки провалились
    raise Exception(f"Не удалось получить ответ после 10 попыток. Последняя ошибка: {last_error}")

