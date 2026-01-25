"""
Анализатор ошибок через Gemini API
"""
import os
import json
import random
import time
from typing import Dict, Any, List, Tuple
try:
    from google import genai
except ImportError:
    genai = None


def check_gemini_api(gemini_api_key: str = None) -> Tuple[bool, str]:
    """
    Проверяет работоспособность Gemini API
    
    Args:
        gemini_api_key: API ключ для Gemini (если None, берется из переменной окружения)
    
    Returns:
        (is_working, message): кортеж (работает ли API, сообщение о статусе)
    """
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        return False, "GEMINI_API_KEY не установлен. Установите переменную окружения или передайте ключ напрямую."
    
    if genai is None:
        return False, "Библиотека google-genai не установлена. Установите: pip install google-genai"
    
    try:
        # Инициализируем клиент с API ключом
        client = genai.Client(api_key=gemini_api_key)
        
        # Пробуем сделать простой запрос для проверки
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Привет"
        )
        
        # Проверяем, что получили ответ
        if hasattr(response, 'text') or (hasattr(response, 'candidates') and len(response.candidates) > 0):
            return True, "✅ Gemini API работает корректно"
        else:
            return False, "⚠️ Gemini API вернул неожиданный формат ответа"
    
    except Exception as e:
        return False, f"❌ Ошибка при проверке Gemini API: {str(e)}"


def analyze_errors_with_gemini(
    model_name: str,
    parsing_errors: List[Dict[str, Any]],
    quality_metrics: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    prompt_full_text: str = None,
    gemini_api_key: str = None
) -> Dict[str, Any]:
    """
    Анализирует ошибки модели через Gemini API и дает рекомендации
    
    Args:
        model_name: название протестированной модели
        parsing_errors: список словарей с ошибками: [{"text_index": int, "text": str, "error": str, "response": str}, ...]
        quality_metrics: метрики качества ответов
        hyperparameters: гиперпараметры модели
        prompt_full_text: полный текст использованного промпта
        gemini_api_key: API ключ для Gemini (если None, берется из переменной окружения)
    
    Returns:
        словарь с анализом и рекомендациями
    """
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        return {
            "status": "error",
            "message": "GEMINI_API_KEY не установлен. Установите переменную окружения или передайте ключ напрямую."
        }
    
    if genai is None:
        return {
            "status": "error",
            "message": "Библиотека google-genai не установлена. Установите: pip install google-genai"
        }
    
    # Формируем промпт для анализа (один раз, вне цикла попыток)
    prompt_section = ""
    if prompt_full_text:
        # Передаем полный промпт без обрезки
        prompt_section = f"""
Использованный промпт (полный, {len(prompt_full_text)} символов):
{prompt_full_text}
"""
    
    # Формируем примеры ошибок с исходными текстами
    errors_section = ""
    if parsing_errors:
        # parsing_errors - список записей с полями {text_index, text, response, errors}
        # Берем до 3 примеров текстов с ошибками
        example_errors = parsing_errors[:3]
        
        if example_errors:
            errors_section = "\nПримеры ошибок с исходными текстами:\n"
            for idx, error_entry in enumerate(example_errors, 1):
                errors_section += f"\n--- Пример {idx} ---\n"
                errors_section += f"Исходный текст:\n{error_entry.get('text', '')}\n\n"
                errors_list = error_entry.get('errors', [])
                if errors_list:
                    errors_section += f"Ошибки:\n"
                    for err in errors_list:
                        errors_section += f"- {err}\n"
                if error_entry.get('response'):
                    errors_section += f"\nОтвет модели: {error_entry.get('response', '')}\n"
        
        # Формируем список из до 10 ошибок (только описания ошибок)
        error_descriptions = []
        for error_entry in parsing_errors:
            text_idx = error_entry.get('text_index', 0)
            for err in error_entry.get('errors', []):
                error_descriptions.append(f"Текст #{text_idx}: {err}")
                if len(error_descriptions) >= 10:
                    break
            if len(error_descriptions) >= 10:
                break
        
        if error_descriptions:
            errors_section += f"\n\nСписок ошибок (показано {len(error_descriptions)}):\n"
            for error_desc in error_descriptions:
                errors_section += f"- {error_desc}\n"
    else:
        errors_section = "\nОшибок парсинга не обнаружено.\n"
    
    prompt = f"""
Ты — эксперт по оценке качества работы языковых моделей. Проанализируй результаты тестирования модели и дай рекомендации по улучшению.

Модель: {model_name}

Гиперпараметры:
{json.dumps(hyperparameters, indent=2, ensure_ascii=False)}
{prompt_section}
Метрики качества:
- Массовая доля: Accuracy: {quality_metrics.get('массовая доля', {}).get('accuracy', 0):.2%}, Precision: {quality_metrics.get('массовая доля', {}).get('precision', 0):.2%}, Recall: {quality_metrics.get('массовая доля', {}).get('recall', 0):.2%}, F1: {quality_metrics.get('массовая доля', {}).get('f1', 0):.2%}
- Прочее: Accuracy: {quality_metrics.get('прочее', {}).get('accuracy', 0):.2%}, Precision: {quality_metrics.get('прочее', {}).get('precision', 0):.2%}, Recall: {quality_metrics.get('прочее', {}).get('recall', 0):.2%}, F1: {quality_metrics.get('прочее', {}).get('f1', 0):.2%}
{errors_section}
Проанализируй:
1. Характерные ошибки модели
2. Причины ошибок парсинга JSON
3. Причины ошибок в извлечении данных
4. Рекомендации по улучшению промпта (если промпт предоставлен)
5. Рекомендации по настройке гиперпараметров
6. Общие рекомендации по улучшению качества

Ответь структурированно на русском языке.
"""
    
    # Количество попыток
    num_retries = 3
    last_error = None
    
    for attempt in range(num_retries):
        try:
            # Инициализируем клиент с API ключом (на каждой попытке, на случай проблем с подключением)
            client = genai.Client(api_key=gemini_api_key)
            
            # Используем новый API
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            # Извлекаем текст ответа
            # Пробуем разные способы извлечения текста
            if hasattr(response, 'text'):
                analysis_text = response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                # Альтернативный способ через candidates
                analysis_text = response.candidates[0].content.parts[0].text
            else:
                # Если ничего не помогло, преобразуем в строку
                analysis_text = str(response)
            
            return {
                "status": "success",
                "analysis": analysis_text,
                "model_used": "gemini-2.5-flash"
            }
        
        except Exception as e:
            last_error = str(e)
            error_str = last_error.lower()
            
            # Проверяем, является ли это ошибкой подключения (disconnected, connection, timeout и т.д.)
            is_connection_error = (
                "server disconnected" in error_str or
                "connection" in error_str or
                "timeout" in error_str or
                "network" in error_str or
                "socket" in error_str
            )
            
            # Если это последняя попытка, возвращаем ошибку
            if attempt == num_retries - 1:
                return {
                    "status": "error",
                    "message": f"Ошибка при обращении к Gemini API после {num_retries} попыток. Последняя ошибка: {last_error}"
                }
            
            # Если это ошибка подключения, делаем повторную попытку с задержкой
            if is_connection_error:
                delay = 2.0 * (attempt + 1)  # Экспоненциальная задержка: 2, 4, 6 секунд
                print(f"   ⚠️ Ошибка подключения к Gemini API (попытка {attempt + 1}/{num_retries}): {last_error}")
                print(f"   ⏳ Повторная попытка через {delay:.1f} секунд...")
                time.sleep(delay)
            else:
                # Для других ошибок также делаем повторную попытку, но с меньшей задержкой
                delay = 1.0 * (attempt + 1)
                print(f"   ⚠️ Ошибка при обращении к Gemini API (попытка {attempt + 1}/{num_retries}): {last_error}")
                print(f"   ⏳ Повторная попытка через {delay:.1f} секунд...")
                time.sleep(delay)

