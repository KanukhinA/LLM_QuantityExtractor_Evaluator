"""
Функции для работы с API моделями (Gemma 3 через Google Generative AI API и OpenRouter API)
"""
import os
import time
import re
from typing import Tuple, Any, Optional
from config import GEMINI_API_KEY, OPENAI_API_KEY

# Импорт для API моделей
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

# Импорт для OpenRouter API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


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
    repetition_penalty: float = None,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None,
    structured_output: bool = False,
    response_schema: Any = None
) -> str:
    """
    Функция генерации для Gemma 3 моделей через API.
    
    По умолчанию использует детерминированную генерацию (temperature=0.0, do_sample=False).
    Для включения sampling передайте temperature > 0 (используется в few_shot_extractor).
    
    Args:
        client: клиент API (genai.Client)
        tokenizer: не используется, но нужен для совместимости сигнатуры (может быть None)
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        model_name: имя модели API (gemma-3-4b, gemma-3-12b, gemma-3-27b)
        repetition_penalty: штраф за повторения (если None, не используется, для API не поддерживается напрямую)
        temperature: температура для sampling (если None, используется 0.0 - детерминированная генерация)
        top_k: top_k для sampling (если указан, используется)
        top_p: top_p для sampling (если указан, используется)
    
    Returns:
        сгенерированный текст
    """
    import time
    import re
    
    # По умолчанию используем детерминированную генерацию (do_sample=False)
    if temperature is None:
        temperature = 0.0
    
    num_retries = 10
    last_error = None
    
    for attempt in range(num_retries):
        try:
            # Формируем параметры запроса
            generation_config = {
                "max_output_tokens": max_new_tokens,
                "temperature": temperature,
            }
            
            # Добавляем опциональные параметры sampling
            if top_p is not None:
                generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
            
            # Добавляем structured output, если указан
            if structured_output and response_schema is not None:
                try:
                    # Конвертируем Pydantic схему в JSON Schema для Gemini API
                    if hasattr(response_schema, 'model_json_schema'):
                        json_schema = response_schema.model_json_schema()
                        generation_config["response_schema"] = json_schema
                        generation_config["response_mime_type"] = "application/json"
                except Exception as e:
                    print(f"   ⚠️ Предупреждение: не удалось добавить structured output: {e}")
                    print(f"   Продолжаем без structured output...")
            
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
            if "404" in error_str or "not found" in error_str or "model not found" in error_str:
                raise Exception(f"Модель не найдена (404): {last_error}")
            
            # Проверяем, является ли это ошибкой 429 (rate limit / resource exhausted)
            is_rate_limit = (
                "429" in error_str or 
                "resource_exhausted" in error_str or 
                "rate limit" in error_str or
                "quota exceeded" in error_str
            )
            
            # Если это последняя попытка, пробрасываем исключение
            if attempt == num_retries - 1:
                raise Exception(f"Ошибка после {num_retries} попыток. Последняя ошибка: {last_error}")
            
            # Для rate limit ошибок пытаемся извлечь требуемое время ожидания из ошибки
            if is_rate_limit:
                delay = None
                
                # Пытаемся найти время ожидания в тексте ошибки
                patterns = [
                    r'retry[_\s]?after[_\s]?:?\s*(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)',
                    r'wait[_\s]+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)',
                    r'retry[_\s]?in[_\s]+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)',
                    r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)\s+.*?(?:retry|wait)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, last_error, re.IGNORECASE)
                    if match:
                        try:
                            extracted_delay = float(match.group(1))
                            if 1 <= extracted_delay <= 3600:
                                delay = extracted_delay
                                break
                        except (ValueError, IndexError):
                            continue
                
                # Если не удалось извлечь время из ошибки, используем экспоненциальную задержку
                if delay is None:
                    delay = min(5.0 * (2 ** attempt), 60.0)
                    print(f"   ⚠️ Rate limit (429). Время ожидания не указано. Используем {delay:.1f} секунд перед попыткой {attempt + 2}/{num_retries}...")
                else:
                    delay = delay * 1.1
                    print(f"   ⚠️ Rate limit (429). API требует ожидания {delay:.1f} секунд. Ожидание перед попыткой {attempt + 2}/{num_retries}...")
                
                time.sleep(delay)
            else:
                # Для других ошибок используем меньшую задержку
                delay = 0.5 * (attempt + 1)
                time.sleep(delay)
    
    # Если дошли сюда, все попытки провалились
    raise Exception(f"Не удалось получить ответ после {num_retries} попыток. Последняя ошибка: {last_error}")


# ============================================================================
# API модели через OpenRouter
# ============================================================================

def load_openrouter_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Универсальная загрузка для API-моделей через OpenRouter (или другой OpenAI-совместимый endpoint)."""
    if not OPENAI_AVAILABLE:
        raise ImportError("Библиотека openai не установлена. Установите: pip install openai")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    if not OPENAI_API_KEY.startswith("sk-") and not OPENAI_API_KEY.startswith("sk-or-"):
        print("   Предупреждение: API ключ не начинается с 'sk-' или 'sk-or-'.")
    print(f"   Инициализация API клиента (base_url={base_url})...")
    try:
        client = OpenAI(base_url=base_url, api_key=OPENAI_API_KEY)
        print("   API клиент инициализирован")
        return client, None
    except Exception as e:
        print(f"   Ошибка инициализации API клиента: {e}")
        raise


def load_deepseek_r1t_chimera_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Загрузка deepseek-r1t-chimera через OpenRouter API (возвращает клиент API вместо модели)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("Библиотека openai не установлена. Установите: pip install openai")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    
    # Проверяем формат ключа
    if not OPENAI_API_KEY.startswith("sk-") and not OPENAI_API_KEY.startswith("sk-or-"):
        print(f"   ⚠️ Предупреждение: API ключ не начинается с 'sk-' или 'sk-or-'. Убедитесь, что это правильный ключ OpenRouter.")
    
    print(f"   Инициализация OpenRouter API клиента для deepseek-r1t-chimera...")
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENAI_API_KEY,
        )
        print(f"   ✓ OpenRouter API клиент инициализирован")
        return client, None
    except Exception as e:
        print(f"   ❌ Ошибка инициализации OpenRouter API клиента: {e}")
        raise


def load_mistral_small_3_1_24b_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Загрузка mistral-small-3.1-24b-instruct через OpenRouter API (возвращает клиент API вместо модели)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("Библиотека openai не установлена. Установите: pip install openai")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    
    # Проверяем формат ключа
    if not OPENAI_API_KEY.startswith("sk-") and not OPENAI_API_KEY.startswith("sk-or-"):
        print(f"   ⚠️ Предупреждение: API ключ не начинается с 'sk-' или 'sk-or-'. Убедитесь, что это правильный ключ OpenRouter.")
    
    print(f"   Инициализация OpenRouter API клиента для mistral-small-3.1-24b-instruct...")
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENAI_API_KEY,
        )
        print(f"   ✓ OpenRouter API клиент инициализирован")
        return client, None
    except Exception as e:
        print(f"   ❌ Ошибка инициализации OpenRouter API клиента: {e}")
        raise


def load_qwen_3_32b_api() -> Tuple[Optional[Any], Optional[Any]]:
    """Загрузка qwen/qwen3-32b через OpenRouter API (возвращает клиент API вместо модели)"""
    if not OPENAI_AVAILABLE:
        raise ImportError("Библиотека openai не установлена. Установите: pip install openai")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не установлен. Установите переменную окружения или в config_secrets.py")
    
    # Проверяем формат ключа
    if not OPENAI_API_KEY.startswith("sk-") and not OPENAI_API_KEY.startswith("sk-or-"):
        print(f"   ⚠️ Предупреждение: API ключ не начинается с 'sk-' или 'sk-or-'. Убедитесь, что это правильный ключ OpenRouter.")
    
    print(f"   Инициализация OpenRouter API клиента для qwen/qwen3-32b...")
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENAI_API_KEY,
        )
        print(f"   ✓ OpenRouter API клиент инициализирован")
        return client, None
    except Exception as e:
        print(f"   ❌ Ошибка инициализации OpenRouter API клиента: {e}")
        raise


def generate_openrouter_api(
    client, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 1024, 
    model_name: str = "tngtech/deepseek-r1t-chimera:free",
    repetition_penalty: float = None,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None,
    structured_output: bool = False,
    response_schema: Any = None
) -> str:
    """
    Функция генерации для моделей через OpenRouter API.
    
    По умолчанию использует детерминированную генерацию (temperature=0.0, do_sample=False).
    Для включения sampling передайте temperature > 0 (используется в few_shot_extractor).
    
    Args:
        client: клиент API (OpenAI)
        tokenizer: не используется, но нужен для совместимости сигнатуры (должен быть None для API моделей)
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        model_name: имя модели API
        repetition_penalty: штраф за повторения (если None, не используется)
        temperature: температура для sampling (если None, используется 0.0 - детерминированная генерация)
        top_k: top_k для sampling (если указан, используется)
        top_p: top_p для sampling (если указан, используется)
    
    Returns:
        сгенерированный текст
    """
    import time
    import warnings
    
    # Защита: убеждаемся, что tokenizer не используется для API моделей
    if tokenizer is not None:
        warnings.warn("tokenizer передан в generate_openrouter_api, но не используется для API моделей. Убедитесь, что для API моделей tokenizer=None.")
    
    # По умолчанию используем детерминированную генерацию (do_sample=False)
    if temperature is None:
        temperature = 0.0
    
    num_retries = 10
    last_error = None
    
    for attempt in range(num_retries):
        try:
            # Формируем параметры запроса
            messages = [{"role": "user", "content": prompt}]
            
            generation_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
            
            # Добавляем опциональные параметры sampling
            if top_p is not None:
                generation_params["top_p"] = top_p
            if top_k is not None:
                generation_params["top_k"] = top_k
            if repetition_penalty is not None:
                generation_params["frequency_penalty"] = repetition_penalty
            
            # Выполняем запрос к API
            try:
                response = client.chat.completions.create(
                    **generation_params,
                    extra_headers={
                        "HTTP-Referer": "https://github.com",  # Optional
                        "X-Title": "SmallLLMEvaluator",  # Optional
                    }
                )
            except Exception as api_error:
                # Специальная обработка ошибки 401 (аутентификация)
                error_str = str(api_error).lower()
                if "401" in error_str or "unauthorized" in error_str or "cookie auth" in error_str or "no cookie" in error_str:
                    print(f"   ❌ Ошибка аутентификации (401): Проверьте правильность OPENAI_API_KEY")
                    print(f"   💡 Убедитесь, что:")
                    print(f"      - Ключ получен с https://openrouter.ai/keys")
                    print(f"      - Ключ начинается с 'sk-or-' (OpenRouter) или 'sk-'")
                    print(f"      - Ключ установлен в config_secrets.py или переменной окружения OPENAI_API_KEY")
                    if OPENAI_API_KEY:
                        key_preview = OPENAI_API_KEY[:10] + "..." if len(OPENAI_API_KEY) > 10 else OPENAI_API_KEY
                        print(f"   🔑 Используемый ключ (первые 10 символов): {key_preview}")
                    else:
                        print(f"   🔑 Ключ не установлен или пустой!")
                raise
            
            # Извлекаем текст ответа
            if response.choices and len(response.choices) > 0:
                text = response.choices[0].message.content
            else:
                text = ""
            
            # Если получили текст, возвращаем его
            if text and text.strip():
                return text.strip()
            else:
                # Пустой ответ - пробуем следующую попытку
                last_error = "Пустой ответ от API"
                continue
                
        except Exception as e:
            last_error = str(e)
            print(last_error)  # Выводим ошибку в консоль полностью (как для других API моделей)
            error_str = str(e).lower()
            
            # Проверяем, является ли это ошибкой 404 (модель не найдена)
            if "404" in error_str or "not found" in error_str or "model not found" in error_str:
                raise Exception(f"Модель не найдена (404): {last_error}")
            
            # Проверяем, является ли это ошибкой 401 (аутентификация)
            if "401" in error_str or "unauthorized" in error_str or "cookie auth" in error_str or "no cookie" in error_str:
                raise Exception(f"Ошибка аутентификации (401): {last_error}. Проверьте правильность OPENAI_API_KEY.")
            
            # Проверяем, является ли это ошибкой 429 (rate limit)
            is_rate_limit = (
                "429" in error_str or 
                "rate limit" in error_str or
                "quota exceeded" in error_str or
                "too many requests" in error_str
            )
            
            # Если это последняя попытка, пробрасываем исключение
            if attempt == 9:
                raise Exception(f"Ошибка после 10 попыток. Последняя ошибка: {last_error}")
            
            # Для rate limit ошибок используем экспоненциальную задержку
            if is_rate_limit:
                delay = min(5.0 * (2 ** attempt), 60.0)  # Максимум 60 секунд
                print(f"   ⚠️ Rate limit (429). Ожидание {delay:.1f} секунд перед попыткой {attempt + 2}/10...")
                time.sleep(delay)
            else:
                # Для других ошибок используем меньшую задержку
                delay = 0.5 * (attempt + 1)
                time.sleep(delay)
    
    # Если дошли сюда, все попытки провалились
    raise Exception(f"Не удалось получить ответ после 10 попыток. Последняя ошибка: {last_error}")

