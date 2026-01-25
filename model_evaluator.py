"""
Основной класс для оценки моделей LLM
"""
import torch
import gc
import time
import pandas as pd
import json
import copy
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import os

from utils import build_prompt3, parse_json_safe, is_valid_json, extract_json_from_response
from metrics import calculate_quality_metrics
from gpu_info import get_gpu_info, get_gpu_memory_usage
from multi_agent_graph import process_with_multi_agent
import re


def sanitize_filename(name: str) -> str:
    """
    Санитизирует имя для использования в имени файла.
    Заменяет все недопустимые символы на подчеркивания.
    
    Args:
        name: исходное имя
        
    Returns:
        безопасное имя для файла
    """
    # Недопустимые символы для имен файлов в Windows и Linux: < > : " / \ | ? *
    # Также заменяем пробелы и другие проблемные символы
    invalid_chars = r'[<>:"/\\|?*\s]'
    sanitized = re.sub(invalid_chars, '_', name)
    # Удаляем множественные подчеркивания
    sanitized = re.sub(r'_+', '_', sanitized)
    # Удаляем подчеркивания в начале и конце
    sanitized = sanitized.strip('_')
    return sanitized
try:
    from gemini_analyzer import analyze_errors_with_gemini
except ImportError:
    analyze_errors_with_gemini = None


class ModelEvaluator:
    """
    Класс для оценки LLM моделей на датасете
    """
    
    def __init__(self, 
                 dataset_path: str,
                 ground_truth_path: Optional[str] = None,
                 output_dir: str = "results"):
        """
        Args:
            dataset_path: путь к датасету (Excel файл)
            ground_truth_path: путь к файлу с истинными значениями (опционально)
            output_dir: директория для сохранения результатов
        """
        self.dataset_path = dataset_path
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Загружаем датасет
        print(f"📂 Загрузка датасета из: {dataset_path}")
        
        # Проверяем существование файла перед загрузкой
        if not os.path.exists(dataset_path):
            abs_path = os.path.abspath(dataset_path)
            current_dir = os.getcwd()
            error_msg = (
                f"❌ Ошибка: файл датасета не найден!\n"
                f"   Путь: {dataset_path}\n"
                f"   Абсолютный путь: {abs_path}\n"
                f"   Текущая рабочая директория: {current_dir}\n"
                f"   Убедитесь, что файл существует и путь указан правильно."
            )
            raise FileNotFoundError(error_msg)
        
        self.df_full = pd.read_excel(dataset_path)
        print(f"   ✅ Датасет загружен: {len(self.df_full)} строк, {len(self.df_full.columns)} колонок")
        print(f"   📋 Колонки: {', '.join(self.df_full.columns.tolist()[:5])}{'...' if len(self.df_full.columns) > 5 else ''}")
        
        # Удаляем колонки, которые не нужны для текстов
        self.df = self.df_full.drop(["json", "Unnamed: 0"], axis=1, errors='ignore')
        self.texts = self.df["text"].tolist()
        print(f"   ✅ Извлечено {len(self.texts)} текстов для обработки\n")
        
        # Загружаем ground truth из того же файла (колонка json_parsed)
        self.ground_truths = None
        if "json_parsed" in self.df_full.columns:
            try:
                # json_parsed уже является словарем, но может быть строкой
                self.ground_truths = []
                for j in self.df_full["json_parsed"]:
                    if isinstance(j, dict):
                        self.ground_truths.append(j)
                    elif isinstance(j, str):
                        self.ground_truths.append(parse_json_safe(j))
                    else:
                        self.ground_truths.append({})
                non_empty = sum(1 for gt in self.ground_truths if gt)
                print(f"   ✅ Загружено {len(self.ground_truths)} ground truth значений из колонки json_parsed")
                print(f"      (Непустых: {non_empty}, Пустых: {len(self.ground_truths) - non_empty})\n")
            except Exception as e:
                print(f"   ⚠️ Не удалось загрузить ground truth из json_parsed: {e}\n")
        elif ground_truth_path and os.path.exists(ground_truth_path):
            # Fallback: загрузка из отдельного файла (старый способ)
            try:
                print(f"   📂 Загрузка ground truth из отдельного файла: {ground_truth_path}")
                gt_df = pd.read_excel(ground_truth_path)
                if "json" in gt_df.columns:
                    self.ground_truths = [parse_json_safe(str(j)) for j in gt_df["json"]]
                    print(f"   ✅ Загружено {len(self.ground_truths)} ground truth значений из отдельного файла\n")
            except Exception as e:
                print(f"   ⚠️ Не удалось загрузить ground truth: {e}\n")
        else:
            print(f"   ⚠️ Ground truth не найден (колонка json_parsed отсутствует)\n")
    
    def clear_memory(self):
        """Очистка GPU памяти"""
        print("♻️ Очистка памяти PyTorch...")
        global model, tokenizer
        try:
            del model
        except NameError:
            pass
        try:
            del tokenizer
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Память очищена")
    
    def evaluate_model(self,
                      model_name: str,
                      load_model_func: Callable,
                      generate_func: Callable,
                      hyperparameters: Dict[str, Any],
                      prompt_template: str = None,
                      max_new_tokens: int = 1024,
                      num_retries: int = 2,
                      verbose: bool = False,
                      use_gemini_analysis: bool = False,
                      gemini_api_key: str = None) -> Dict[str, Any]:
        """
        Оценивает модель на датасете
        
        Args:
            model_name: название модели
            load_model_func: функция для загрузки модели (должна возвращать (model, tokenizer))
            generate_func: функция генерации (model, tokenizer, prompt) -> response_text
            hyperparameters: словарь с гиперпараметрами (может содержать multi_agent_mode)
            prompt_template: шаблон промпта (если None, используется build_prompt3)
            max_new_tokens: максимальное количество новых токенов
            num_retries: количество попыток при ошибке
        
        Returns:
            словарь с результатами оценки
        """
        # Определяем режим работы из гиперпараметров
        multi_agent_mode = hyperparameters.get("multi_agent_mode", None)
        use_multi_agent = multi_agent_mode is not None and multi_agent_mode != ""
        
        # Определяем, является ли модель API-моделью
        is_api_model = hyperparameters.get("api_model", False)
        if not is_api_model:
            # Также проверяем по типу возвращаемых значений после загрузки
            # Для API моделей tokenizer будет None
            pass  # Проверим после загрузки
        
        # Устанавливаем num_retries для API моделей (10 попыток)
        if is_api_model:
            num_retries = 10
        
        # Определяем название режима для вывода
        if multi_agent_mode:
            mode_name = f"Мультиагентный ({multi_agent_mode})"
        else:
            mode_name = "Одноагентный"
        
        print(f"\n{'='*80}")
        print(f"🚀 НАЧАЛО ОЦЕНКИ МОДЕЛИ")
        print(f"{'='*80}")
        print(f"📌 Модель: {model_name}")
        print(f"📌 Датасет: {len(self.texts)} текстов")
        print(f"📌 Режим: {mode_name}")
        print(f"📌 Гиперпараметры:")
        for key, value in hyperparameters.items():
            print(f"   • {key}: {value}")
        print(f"{'='*80}\n")
        
        # Проверяем, является ли это API моделью (до загрузки)
        is_api_model = hyperparameters.get("api_model", False)
        
        # Информация о GPU/API до загрузки модели
        if is_api_model:
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ:")
            print(f"   • Тип: API (Google Generative AI)")
            print(f"   • Модель будет использоваться через API")
            print()
            gpu_info_before = {"api": True}
        else:
            gpu_info_before = get_gpu_info()
            print(f"📊 ИНФОРМАЦИЯ О GPU (до загрузки модели):")
            print(f"   • CUDA доступна: {gpu_info_before.get('cuda_available', False)}")
            if gpu_info_before.get('cuda_available'):
                print(f"   • Название GPU: {gpu_info_before.get('gpu_name', 'N/A')}")
                print(f"   • Версия CUDA: {gpu_info_before.get('cuda_version', 'N/A')}")
                print(f"   • Общая память: {gpu_info_before.get('gpu_memory_total_gb', 0):.2f} GB")
                print(f"   • Использовано памяти: {gpu_info_before.get('gpu_memory_allocated_gb', 0):.2f} GB")
            print()
        
        # Загружаем модель
        print(f"📦 ЗАГРУЗКА МОДЕЛИ...")
        start_load = time.time()
        try:
            model, tokenizer = load_model_func()
            load_time = time.time() - start_load
            print(f"✅ Модель успешно загружена за {load_time:.2f} секунд ({load_time/60:.2f} минут)")
        except Exception as e:
            import traceback
            error_details = str(e)
            full_traceback = traceback.format_exc()
            
            print(f"\n{'='*80}")
            print(f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ")
            print(f"{'='*80}")
            print(f"Ошибка: {error_details}")
            print(f"\nПолный traceback:")
            print(f"{'─'*80}")
            print(full_traceback)
            print(f"{'─'*80}")
            print(f"Детали ошибки также сохранены в отчёте")
            print(f"{'='*80}\n")
            
            # Очищаем память после ошибки загрузки
            self.clear_memory()
            
            return {
                "status": "error",
                "error": f"Ошибка загрузки модели: {error_details}",
                "error_traceback": full_traceback
            }
        
        # Информация о GPU/API после загрузки
        if is_api_model:
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ:")
            print(f"   • Тип: API (Google Generative AI)")
            print(f"   • Модель доступна через API")
            print()
            gpu_info_after = {"api": True}
            memory_after_load = {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
        else:
            gpu_info_after = get_gpu_info()
            memory_after_load = get_gpu_memory_usage()
            print(f"📊 ИНФОРМАЦИЯ О GPU (после загрузки модели):")
            print(f"   • Использовано памяти: {memory_after_load['allocated']:.2f} GB")
            print(f"   • Зарезервировано памяти: {memory_after_load['reserved']:.2f} GB")
            print(f"   • Доступно памяти: {memory_after_load['total'] - memory_after_load['allocated']:.2f} GB")
            print()
        
        # Используем промпт
        if prompt_template is None:
            prompt_template = build_prompt3
        
        # Оценка на датасете
        results = []
        parsing_errors = []  # Список словарей с ошибками: {"text_index": int, "text": str, "error": str, "response": str}
        times = []
        memory_samples = []  # Для сбора измерений памяти во время инференса (только для локальных моделей)
        total_start_time = time.time()
        
        # Переводим в eval режим только локальные модели
        if not is_api_model and hasattr(model, 'eval'):
            model.eval()
        
        print(f"🔄 ОБРАБОТКА ДАТАСЕТА")
        print(f"{'='*80}")
        print(f"Всего текстов: {len(self.texts)}")
        print(f"{'='*80}\n")
        
        # Создаем обертку для генератора для мультиагентного подхода
        if use_multi_agent:
            if is_api_model:
                # Для API моделей используем APIGenerator
                from core.generators import APIGenerator
                model_name = hyperparameters.get("model_name", "gemma-3-12b-it")
                generator = APIGenerator(model, tokenizer, model_name=model_name)
            else:
                # Для локальных моделей используем StandardGenerator
                from core.generators import StandardGenerator
                generator = StandardGenerator(model, tokenizer)
        
        interrupted = False
        last_processed_index = -1
        
        try:
            for i, text in enumerate(self.texts):
                response_text = ""
                error_msg = None
                
                if use_multi_agent:
                    # Мультиагентный подход
                    try:
                        # Выводим сообщение только при verbose режиме
                        if verbose:
                            print(f"   🔄 Мультиагентная обработка текста {i+1}/{len(self.texts)}:")
                        start_time = time.time()
                        result = process_with_multi_agent(
                            text=text,
                            generator=generator,
                            max_new_tokens=max_new_tokens,
                            multi_agent_mode=multi_agent_mode
                        )
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        
                        # Измеряем память во время инференса
                        if not is_api_model:
                            memory_sample = get_gpu_memory_usage()
                            memory_samples.append(memory_sample["allocated"])
                        
                        response_text = result.get("response", "")
                        json_part = result.get("json", "")
                        parsed_json = result.get("json_parsed", {})
                        is_valid = result.get("is_valid", False)
                        error_msg = result.get("error")
                        
                        if error_msg:
                            parsing_errors.append({
                                "text_index": i,
                                "text": text,
                                "error": f"Ошибка в мультиагентном подходе: {error_msg}",
                                "response": response_text[:500] if response_text else ""
                            })
                        
                        if not is_valid and json_part:
                            # Для API моделей сохраняем полный JSON при verbose
                            json_display = json_part if (is_api_model and verbose) else json_part[:200]
                            parsing_errors.append({
                                "text_index": i,
                                "text": text,
                                "error": f"Невалидный JSON. Ответ: {json_display}",
                                "response": json_part[:500]
                            })
                        
                        results.append({
                            "text": text,
                            "json": json_part,
                            "json_parsed": parsed_json,
                            "is_valid": is_valid
                        })
                    except Exception as e:
                        error_msg = str(e)
                        import traceback
                        traceback_str = traceback.format_exc()
                        # Для API моделей сохраняем полный traceback
                        traceback_display = traceback_str if is_api_model else traceback_str[:200]
                        parsing_errors.append({
                            "text_index": i,
                            "text": text,
                            "error": f"Критическая ошибка в мультиагентном подходе: {error_msg}. Traceback: {traceback_display}",
                            "response": ""
                        })
                        results.append({
                            "text": text,
                            "json": "",
                            "json_parsed": {},
                            "is_valid": False
                        })
                else:
                    # Одноагентный подход (оригинальный)
                    prompt = prompt_template(text)
                    
                    # Попытки генерации
                    for attempt in range(num_retries):
                        try:
                            start_time = time.time()
                            # Передаем repetition_penalty из гиперпараметров, если есть
                            repetition_penalty = hyperparameters.get("repetition_penalty")
                            # Для API моделей передаем model_name из hyperparameters
                            if is_api_model and "model_name" in hyperparameters:
                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens, model_name=hyperparameters["model_name"])
                            elif repetition_penalty is not None:
                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens, repetition_penalty=repetition_penalty)
                            elif "enable_thinking" in hyperparameters:
                                # Для Qwen3 передаем enable_thinking из hyperparameters
                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens, enable_thinking=hyperparameters.get("enable_thinking", True))
                            else:
                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens)
                            elapsed = time.time() - start_time
                            times.append(elapsed)
                            
                            # Выводим исходный текст и полный ответ в консоль (только при verbose)
                            if verbose:
                                print(f"   📝 Исходный текст для анализа:")
                                print(f"   {'─'*76}")
                                for line in text.split('\n'):
                                    print(f"   {line}")
                                print(f"   {'─'*76}")
                                model_type_label = "API модели" if is_api_model else "модели"
                                print(f"   📋 Полный ответ {model_type_label}:")
                                print(f"   {'─'*76}")
                                for line in response_text.split('\n'):
                                    print(f"   {line}")
                                print(f"   {'─'*76}")
                            
                            # Измеряем память во время инференса (только для локальных моделей)
                            if not is_api_model:
                                memory_sample = get_gpu_memory_usage()
                                memory_samples.append(memory_sample["allocated"])
                            break
                        except KeyboardInterrupt:
                            # Пробрасываем KeyboardInterrupt наверх для обработки в основном цикле
                            raise
                        except Exception as e:
                            error_msg = str(e)
                            # Для API моделей выводим полную ошибку без обрезки (всегда, так как это ошибка)
                            if is_api_model:
                                print(f"  ⚠️ [{i+1}/{len(self.texts)}] Ошибка при генерации (попытка {attempt+1}/{num_retries}):")
                                print(f"     {error_msg}")
                            else:
                                # Для локальных моделей обрезаем при не verbose режиме
                                error_display = error_msg if verbose else error_msg[:100]
                                print(f"  ⚠️ [{i+1}/{len(self.texts)}] Ошибка при генерации (попытка {attempt+1}/{num_retries}): {error_display}")
                            if attempt < num_retries - 1:
                                time.sleep(4 + attempt * 2)
                            else:
                                # Если все попытки исчерпаны, сохраняем детальную информацию об ошибке
                                import traceback
                                traceback_str = traceback.format_exc()
                                # Для API моделей сохраняем полный traceback
                                traceback_display = traceback_str if is_api_model else traceback_str[:200]
                                parsing_errors.append({
                                    "text_index": i,
                                    "text": text,
                                    "error": f"Критическая ошибка генерации после {num_retries} попыток: {error_msg}. Traceback: {traceback_display}",
                                    "response": ""
                                })
                    
                    if not response_text:
                        print(f"  ❌ [{i+1}/{len(self.texts)}] Ответ не получен — пропуск")
                        if error_msg:
                            # Для API моделей выводим полную ошибку без обрезки (всегда, так как это ошибка)
                            if is_api_model:
                                print(f"     Последняя ошибка: {error_msg}")
                            else:
                                # Для локальных моделей обрезаем при не verbose режиме
                                error_display = error_msg if verbose else error_msg[:200]
                                print(f"     Последняя ошибка: {error_display}")
                        parsing_errors.append(f"Текст #{i}: не получен ответ. Ошибка: {error_msg if error_msg else 'Неизвестная ошибка'}")
                        results.append({
                            "text": text,
                            "json": "",
                            "json_parsed": {},
                            "is_valid": False
                        })
                        continue
                    
                    # Извлекаем JSON
                    json_part = extract_json_from_response(response_text)
                    parsed_json = parse_json_safe(json_part)
                    is_valid = is_valid_json(json_part)
                    
                    if not is_valid:
                        # Для API моделей при verbose выводим полный JSON, иначе обрезаем
                        json_display = json_part if (is_api_model and verbose) else (json_part[:200] if len(json_part) > 200 else json_part)
                        parsing_errors.append({
                            "text_index": i,
                            "text": text,
                            "error": f"Невалидный JSON. Ответ: {json_display}",
                            "response": json_part[:500]
                        })
                    
                    results.append({
                        "text": text,
                        "json": json_part,
                        "json_parsed": parsed_json,
                        "is_valid": is_valid
                    })
            
            # Выводим прогресс после каждого запроса
            elapsed_total = time.time() - total_start_time
            avg_time = sum(times) / len(times) if times else 0
            progress_pct = ((i + 1) / len(self.texts)) * 100
            remaining = len(self.texts) - (i + 1)
            eta_seconds = avg_time * remaining if avg_time > 0 else 0
            eta_minutes = eta_seconds / 60
            
            valid_count = sum(1 for r in results if r["is_valid"])
            invalid_count = (i + 1) - valid_count
            
            # Форматируем время
            if eta_minutes < 1:
                eta_str = f"{eta_seconds:.0f} сек"
            else:
                eta_str = f"{eta_minutes:.1f} мин"
            
            # Выводим статус после каждого запроса (зависит от verbose)
            if verbose:
                # Подробный вывод при verbose=True
                status_line = (
                    f"  ✅ [{i + 1}/{len(self.texts)}] ({progress_pct:.1f}%) | "
                    f"Валидных: {valid_count} | Невалидных: {invalid_count} | "
                    f"ETA: {eta_str}"
                )
                print(status_line)
            else:
                # Короткий вывод при verbose=False (только счетчик и основные метрики)
                status_line = (
                    f"  [{i + 1}/{len(self.texts)}] "
                    f"✓: {valid_count} ✗: {invalid_count} | "
                    f"ETA: {eta_str}"
                )
                print(f"\r{status_line}", end="", flush=True)
            
            # Подробный прогресс каждые 10 текстов или в конце (только при verbose)
            if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(self.texts)):
                print()  # Новая строка для подробного вывода
                print(f"     📊 Детальная статистика:")
                print(f"        • Прогресс: {progress_pct:.1f}% ({i + 1}/{len(self.texts)})")
                print(f"        • Валидных JSON: {valid_count} | Невалидных: {invalid_count}")
                print(f"        • Средняя скорость: {avg_time:.3f} сек/ответ")
                print(f"        • Прошло времени: {elapsed_total/60:.1f} мин | Осталось: ~{eta_minutes:.1f} мин")
                print()
                
                last_processed_index = i
            else:
                last_processed_index = i
        
        except KeyboardInterrupt:
            interrupted = True
            last_processed_index = i if 'i' in locals() else -1
            print(f"\n\n{'='*80}")
            print(f"⚠️  ПРЕРЫВАНИЕ ОБРАБОТКИ ПОЛЬЗОВАТЕЛЕМ")
            print(f"{'='*80}")
            print(f"Обработано текстов: {len(results)}/{len(self.texts)}")
            print(f"Последний обработанный индекс: {last_processed_index + 1}")
            print()
            
            while True:
                try:
                    choice = input("Выберите действие:\n  1 - Сохранить промежуточные результаты и завершить\n  2 - Продолжить обработку\n  3 - Завершить без сохранения\nВаш выбор (1/2/3): ").strip()
                    
                    if choice == "1":
                        print("\n💾 Сохранение промежуточных результатов...")
                        # Продолжим выполнение для сохранения результатов
                        break
                    elif choice == "2":
                        print("\n▶️  Продолжаем обработку...\n")
                        # Продолжаем цикл с того места, где остановились
                        try:
                            for i in range(last_processed_index + 1, len(self.texts)):
                                response_text = ""
                                error_msg = None
                                
                                if use_multi_agent:
                                    try:
                                        # Выводим сообщение только при verbose режиме
                                        if verbose:
                                            print(f"   🔄 Мультиагентная обработка текста {i+1}/{len(self.texts)}:")
                                        start_time = time.time()
                                        result = process_with_multi_agent(
                                            text=self.texts[i],
                                            generator=generator,
                                            max_new_tokens=max_new_tokens,
                                            multi_agent_mode=multi_agent_mode
                                        )
                                        elapsed = time.time() - start_time
                                        times.append(elapsed)
                                        
                                        memory_sample = get_gpu_memory_usage()
                                        memory_samples.append(memory_sample["allocated"])
                                        
                                        response_text = result.get("response", "")
                                        json_part = result.get("json", "")
                                        parsed_json = result.get("json_parsed", {})
                                        is_valid = result.get("is_valid", False)
                                        error_msg = result.get("error")
                                        
                                        if error_msg:
                                            parsing_errors.append(f"Текст #{i}: ошибка в мультиагентном подходе. Ошибка: {error_msg}")
                                        
                                        if not is_valid and json_part:
                                            # Для API моделей при verbose выводим полный JSON, иначе обрезаем
                                            json_display = json_part if (is_api_model and verbose) else (json_part[:200] if len(json_part) > 200 else json_part)
                                            parsing_errors.append(f"Текст #{i}: невалидный JSON. Ответ: {json_display}")
                                        
                                        results.append({
                                            "text": self.texts[i],
                                            "json": json_part,
                                            "json_parsed": parsed_json,
                                            "is_valid": is_valid
                                        })
                                    except Exception as e:
                                        error_msg = str(e)
                                        import traceback
                                        parsing_errors.append(f"Текст #{i}: критическая ошибка в мультиагентном подходе. Ошибка: {error_msg}. Traceback: {traceback.format_exc()[:200]}")
                                        results.append({
                                            "text": self.texts[i],
                                            "json": "",
                                            "json_parsed": {},
                                            "is_valid": False
                                        })
                                else:
                                    prompt = prompt_template(self.texts[i])
                                    
                                    for attempt in range(num_retries):
                                        try:
                                            start_time = time.time()
                                            repetition_penalty = hyperparameters.get("repetition_penalty")
                                            # Для API моделей передаем model_name из hyperparameters
                                            if is_api_model and "model_name" in hyperparameters:
                                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens, model_name=hyperparameters["model_name"])
                                            elif repetition_penalty is not None:
                                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens, repetition_penalty=repetition_penalty)
                                            elif "enable_thinking" in hyperparameters:
                                                # Для Qwen3 передаем enable_thinking из hyperparameters
                                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens, enable_thinking=hyperparameters.get("enable_thinking", True))
                                            else:
                                                response_text = generate_func(model, tokenizer, prompt, max_new_tokens)
                                            elapsed = time.time() - start_time
                                            times.append(elapsed)
                                            
                                            # Выводим исходный текст и полный ответ в консоль (только при verbose)
                                            if verbose:
                                                print(f"   📝 Исходный текст для анализа:")
                                                print(f"   {'─'*76}")
                                                for line in self.texts[i].split('\n'):
                                                    print(f"   {line}")
                                                print(f"   {'─'*76}")
                                                model_type_label = "API модели" if is_api_model else "модели"
                                                print(f"   📋 Полный ответ {model_type_label}:")
                                                print(f"   {'─'*76}")
                                                for line in response_text.split('\n'):
                                                    print(f"   {line}")
                                                print(f"   {'─'*76}")
                                            
                                            # Измеряем память во время инференса (только для локальных моделей)
                                            if not is_api_model:
                                                memory_sample = get_gpu_memory_usage()
                                                memory_samples.append(memory_sample["allocated"])
                                            break
                                        except Exception as e:
                                            error_msg = str(e)
                                            # Для API моделей выводим полную ошибку без обрезки (всегда, так как это ошибка)
                                            if is_api_model:
                                                print(f"  ⚠️ [{i+1}/{len(self.texts)}] Ошибка при генерации (попытка {attempt+1}/{num_retries}):")
                                                print(f"     {error_msg}")
                                            else:
                                                # Для локальных моделей обрезаем при не verbose режиме
                                                error_display = error_msg if verbose else error_msg[:100]
                                                print(f"  ⚠️ [{i+1}/{len(self.texts)}] Ошибка при генерации (попытка {attempt+1}/{num_retries}): {error_display}")
                                            if attempt < num_retries - 1:
                                                time.sleep(4 + attempt * 2)
                                            else:
                                                import traceback
                                                traceback_str = traceback.format_exc()
                                                # Для API моделей сохраняем полный traceback
                                                if is_api_model:
                                                    parsing_errors.append(f"Текст #{i}: критическая ошибка генерации после {num_retries} попыток. Ошибка: {error_msg}. Traceback: {traceback_str}")
                                                else:
                                                    parsing_errors.append(f"Текст #{i}: критическая ошибка генерации после {num_retries} попыток. Ошибка: {error_msg}. Traceback: {traceback_str[:200]}")
                                
                                    if not response_text:
                                        print(f"  ❌ [{i+1}/{len(self.texts)}] Ответ не получен — пропуск")
                                        if error_msg:
                                            # Для API моделей выводим полную ошибку без обрезки (всегда, так как это ошибка)
                                            if is_api_model:
                                                print(f"     Последняя ошибка: {error_msg}")
                                            else:
                                                # Для локальных моделей обрезаем при не verbose режиме
                                                error_display = error_msg if verbose else error_msg[:200]
                                                print(f"     Последняя ошибка: {error_display}")
                                        parsing_errors.append({
                                            "text_index": i,
                                            "text": self.texts[i],
                                            "error": f"Не получен ответ. Ошибка: {error_msg if error_msg else 'Неизвестная ошибка'}",
                                            "response": ""
                                        })
                                        results.append({
                                            "text": self.texts[i],
                                            "json": "",
                                            "json_parsed": {},
                                            "is_valid": False
                                        })
                                        continue
                                    
                                    json_part = extract_json_from_response(response_text)
                                    parsed_json = parse_json_safe(json_part)
                                    is_valid = is_valid_json(json_part)
                                    
                                    if not is_valid:
                                        # Для API моделей при verbose выводим полный JSON, иначе обрезаем
                                        json_display = json_part if (is_api_model and verbose) else (json_part[:200] if len(json_part) > 200 else json_part)
                                        parsing_errors.append({
                                            "text_index": i,
                                            "text": self.texts[i],
                                            "error": f"Невалидный JSON. Ответ: {json_display}",
                                            "response": response_text[:500] if response_text else json_part[:500]
                                        })
                                    
                                    results.append({
                                        "text": self.texts[i],
                                        "json": json_part,
                                        "json_parsed": parsed_json,
                                        "is_valid": is_valid
                                    })
                                
                                # Выводим прогресс
                                elapsed_total = time.time() - total_start_time
                                avg_time = sum(times) / len(times) if times else 0
                                progress_pct = ((i + 1) / len(self.texts)) * 100
                                remaining = len(self.texts) - (i + 1)
                                eta_seconds = avg_time * remaining if avg_time > 0 else 0
                                eta_minutes = eta_seconds / 60
                                
                                valid_count = sum(1 for r in results if r["is_valid"])
                                invalid_count = (i + 1) - valid_count
                                
                                if eta_minutes < 1:
                                    eta_str = f"{eta_seconds:.0f} сек"
                                else:
                                    eta_str = f"{eta_minutes:.1f} мин"
                                
                                status_line = (
                                    f"  [{i + 1}/{len(self.texts)}] "
                                    f"✓: {valid_count} ✗: {invalid_count} | "
                                    f"Скорость: {avg_time:.2f}с/ответ | "
                                    f"Осталось: ~{eta_str}"
                                )
                                print(f"\r{status_line}", end="", flush=True)
                                
                                # Детальная статистика только при verbose
                                if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(self.texts)):
                                    print()
                                    print(f"     📊 Детальная статистика:")
                                    print(f"        • Прогресс: {progress_pct:.1f}% ({i + 1}/{len(self.texts)})")
                                    print(f"        • Валидных JSON: {valid_count} | Невалидных: {invalid_count}")
                                    print(f"        • Средняя скорость: {avg_time:.3f} сек/ответ")
                                    print(f"        • Прошло времени: {elapsed_total/60:.1f} мин | Осталось: ~{eta_minutes:.1f} мин")
                                    print()
                        except KeyboardInterrupt:
                            print(f"\n\n⚠️  Повторное прерывание. Сохранение промежуточных результатов...")
                            interrupted = True
                            break
                        break
                    elif choice == "3":
                        print("\n❌ Завершение без сохранения...")
                        return {
                            "status": "interrupted",
                            "message": "Обработка прервана пользователем без сохранения",
                            "processed_count": len(results),
                            "total_count": len(self.texts)
                        }
                    else:
                        print("Пожалуйста, введите 1, 2 или 3")
                except KeyboardInterrupt:
                    print("\n\n⚠️  Повторное прерывание. Сохранение промежуточных результатов...")
                    interrupted = True
                    break
        
        # Вычисляем метрики
        total_time = time.time() - total_start_time
        print(f"\n{'='*80}")
        print(f"📊 ВЫЧИСЛЕНИЕ МЕТРИК")
        print(f"{'='*80}\n")
        
        # Процент невалидных JSON
        invalid_count = sum(1 for r in results if not r["is_valid"])
        valid_count = len(results) - invalid_count
        parsing_error_rate = invalid_count / len(results) if results else 0.0
        
        # Статистика по времени
        avg_speed = sum(times) / len(times) if times else 0.0
        min_time = min(times) if times else 0.0
        max_time = max(times) if times else 0.0
        total_inference_time = sum(times)
        
        # Использование памяти во время инференса
        # Используем среднее значение из всех измерений во время инференса
        if is_api_model:
            # Для API моделей не измеряем память
            memory_during_inference_avg = 0.0
            memory_during_inference_max = 0.0
            memory_during_inference_min = 0.0
        elif memory_samples:
            memory_during_inference_avg = sum(memory_samples) / len(memory_samples)
            memory_during_inference_max = max(memory_samples)
            memory_during_inference_min = min(memory_samples)
        else:
            # Fallback: измеряем сейчас, если не было измерений
            current_memory = get_gpu_memory_usage()
            memory_during_inference_avg = current_memory["allocated"]
            memory_during_inference_max = current_memory["allocated"]
            memory_during_inference_min = current_memory["allocated"]
        
        # Для совместимости сохраняем среднее значение как основное
        memory_during_inference = {"allocated": memory_during_inference_avg}
        
        print(f"⏱️  ВРЕМЯ ВЫПОЛНЕНИЯ:")
        print(f"   • Общее время: {total_time/60:.2f} минут ({total_time:.2f} секунд)")
        print(f"   • Время инференса: {total_inference_time/60:.2f} минут")
        print(f"   • Время загрузки модели: {load_time:.2f} секунд")
        print(f"   • Средняя скорость ответа: {avg_speed:.3f} сек/ответ")
        print(f"   • Минимальное время: {min_time:.3f} сек")
        print(f"   • Максимальное время: {max_time:.3f} сек")
        print()
        
        # Подготавливаем примеры промптов для вывода и сохранения
        example_text = self.texts[0] if self.texts else "Пример текста"
        
        workflow_description = ""  # Инициализируем для использования в выводе
        workflow_prompts = None  # Сохраняем для повторного использования
        if use_multi_agent:
            # Используем систему конфигурации workflow для получения промптов
            from workflow_config import get_workflow_prompts
            workflow_prompts = get_workflow_prompts(multi_agent_mode, example_text)
            full_prompt_example = workflow_prompts["full_prompt_example"]
            workflow_description = workflow_prompts.get("description", "")
        else:
            full_prompt_example = prompt_template(example_text)
        
        # Выводим информацию о промпте и режиме
        print(f"📝 ИСПОЛЬЗОВАННЫЙ ПРОМПТ:")
        if use_multi_agent:
            print(f"   • Режим: Мультиагентный ({multi_agent_mode})")
            print(f"   • Используются специализированные промпты из prompt_config.py")
            print(f"   • Агенты: {workflow_description}")
            print(f"   • Полный текст всех промптов (пример с первым текстом):")
            print(f"{'─'*80}")
            # Выводим промпты с отступами для читаемости
            prompt_lines = full_prompt_example.split('\n')
            for line in prompt_lines[:50]:  # Первые 50 строк, чтобы не перегружать консоль
                print(f"   {line}")
            if len(prompt_lines) > 50:
                print(f"   ... (ещё {len(prompt_lines) - 50} строк, полный текст сохранён в отчёте)")
            print(f"{'─'*80}")
        else:
            print(f"   • Режим: Одноагентный")
            print(f"   • Шаблон: {prompt_template.__name__ if hasattr(prompt_template, '__name__') else str(prompt_template)}")
            print(f"   • Полный текст промпта (пример с первым текстом):")
            print(f"{'─'*80}")
            # Выводим промпт с отступами для читаемости
            prompt_lines = full_prompt_example.split('\n')
            for line in prompt_lines[:30]:  # Первые 30 строк, чтобы не перегружать консоль
                print(f"   {line}")
            if len(prompt_lines) > 30:
                print(f"   ... (ещё {len(prompt_lines) - 30} строк, полный текст сохранён в отчёте)")
            print(f"{'─'*80}")
        print()
        
        if is_api_model:
            print(f"💾 ИНФОРМАЦИЯ О РЕСУРСАХ:")
            print(f"   • Тип: API (Google Generative AI)")
            print(f"   • Модель доступна через API")
            print()
        else:
            print(f"💾 ИСПОЛЬЗОВАНИЕ ПАМЯТИ:")
            print(f"   • После загрузки модели: {memory_after_load['allocated']:.2f} GB")
            print(f"   • Во время инференса (среднее): {memory_during_inference_avg:.2f} GB")
            print(f"   • Во время инференса (максимум): {memory_during_inference_max:.2f} GB")
            print(f"   • Во время инференса (минимум): {memory_during_inference_min:.2f} GB")
            print(f"   • Изменение от загрузки: {memory_during_inference_avg - memory_after_load['allocated']:+.2f} GB")
            print()
        
        print(f"📝 СТАТИСТИКА ПАРСИНГА JSON:")
        print(f"   • Всего обработано: {len(results)}")
        print(f"   • Валидных JSON: {valid_count} ({100-parsing_error_rate*100:.1f}%)")
        print(f"   • Невалидных JSON: {invalid_count} ({parsing_error_rate*100:.1f}%)")
        print(f"   • Ошибок парсинга: {len(parsing_errors)}")
        if parsing_errors:
            print(f"\n   📋 Полный список ошибок парсинга ({len(parsing_errors)} ошибок):")
            print(f"   {'─'*76}")
            for i, error in enumerate(parsing_errors, 1):
                # Обрезаем длинные ошибки для консоли
                error_display = error[:200] + "..." if len(error) > 200 else error
                print(f"   {i}. {error_display}")
            print(f"   {'─'*76}")
        print()
        
        # Качество ответов (если есть ground truth)
        quality_metrics = None
        if self.ground_truths and len(self.ground_truths) == len(results):
            try:
                print(f"🎯 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА...")
                # Фильтруем и нормализуем predictions: должны быть словарями
                predictions = []
                for r in results:
                    json_parsed = r.get("json_parsed", {})
                    # Если это список, пропускаем или преобразуем в словарь
                    if isinstance(json_parsed, list):
                        # Если список пустой или содержит не словари, используем пустой словарь
                        predictions.append({})
                    elif isinstance(json_parsed, dict):
                        predictions.append(json_parsed)
                    else:
                        predictions.append({})
                
                # Также проверяем ground_truths
                ground_truths_normalized = []
                for gt in self.ground_truths:
                    if isinstance(gt, list):
                        ground_truths_normalized.append({})
                    elif isinstance(gt, dict):
                        ground_truths_normalized.append(gt)
                    else:
                        ground_truths_normalized.append({})
                
                # Извлекаем тексты и ответы из results
                texts_for_metrics = []
                responses_for_metrics = []
                for r in results:
                    texts_for_metrics.append(r.get("text", ""))
                    responses_for_metrics.append(r.get("json", ""))  # json содержит ответ модели
                
                quality_metrics = calculate_quality_metrics(
                    predictions, ground_truths_normalized,
                    texts=texts_for_metrics,
                    responses=responses_for_metrics
                )
                
                # Проверяем, что quality_metrics - это словарь
                if not isinstance(quality_metrics, dict):
                    print(f"   ⚠️ Ошибка: calculate_quality_metrics вернула не словарь, а {type(quality_metrics)}")
                    quality_metrics = None
                else:
                    mass_dolya = quality_metrics.get('массовая доля', {})
                    prochee = quality_metrics.get('прочее', {})
                    
                    print(f"   ✅ Метрики качества вычислены:")
                    print(f"   📊 Группа 'массовая доля':")
                    print(f"      • Accuracy: {mass_dolya.get('accuracy', 0):.2%}")
                    print(f"      • Precision: {mass_dolya.get('precision', 0):.2%}")
                    print(f"      • Recall: {mass_dolya.get('recall', 0):.2%}")
                    print(f"      • F1-score: {mass_dolya.get('f1', 0):.2%}")
                    print(f"      • TP: {mass_dolya.get('tp', 0)}, FP: {mass_dolya.get('fp', 0)}, FN: {mass_dolya.get('fn', 0)}")
                    print(f"      • Количество сравнений: {mass_dolya.get('количество_сравнений', 0)}")
                    print(f"      • Примеры ошибок: {len(mass_dolya.get('ошибки', []))}")
                    print(f"   📊 Группа 'прочее':")
                    print(f"      • Accuracy: {prochee.get('accuracy', 0):.2%}")
                    print(f"      • Precision: {prochee.get('precision', 0):.2%}")
                    print(f"      • Recall: {prochee.get('recall', 0):.2%}")
                    print(f"      • F1-score: {prochee.get('f1', 0):.2%}")
                    print(f"      • TP: {prochee.get('tp', 0)}, FP: {prochee.get('fp', 0)}, FN: {prochee.get('fn', 0)}")
                    print(f"      • Количество сравнений: {prochee.get('количество_сравнений', 0)}")
                    print(f"      • Примеры ошибок: {len(prochee.get('ошибки', []))}")
            except Exception as e:
                print(f"   ⚠️ Ошибка при вычислении метрик качества: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()
        else:
            print(f"   ⚠️ Ground truth не загружен или не совпадает по размеру с результатами")
            if not self.ground_truths:
                print(f"      (Ground truth не найден в датасете)")
            elif len(self.ground_truths) != len(results):
                print(f"      (Размеры не совпадают: GT={len(self.ground_truths)}, Results={len(results)})")
        print()
        
        # Анализ через Gemini API (если включен)
        gemini_analysis = None
        if use_gemini_analysis and analyze_errors_with_gemini is not None:
            if gemini_api_key is None:
                gemini_api_key = os.environ.get("GEMINI_API_KEY")
            
            if gemini_api_key:
                print(f"🤖 ЗАПУСК АНАЛИЗА ЧЕРЕЗ GEMINI API...")
                try:
                    gemini_analysis = analyze_errors_with_gemini(
                        model_name=model_name,
                        parsing_errors=parsing_errors,
                        quality_metrics=quality_metrics or {},
                        hyperparameters=hyperparameters,
                        prompt_full_text=full_prompt_example,
                        gemini_api_key=gemini_api_key
                    )
                    
                    if gemini_analysis.get("status") == "success":
                        print(f"   ✅ Анализ от Gemini получен успешно!")
                        analysis_text = gemini_analysis.get("analysis", "")
                        if analysis_text:
                            print(f"\n   {'─'*76}")
                            print(f"   📝 АНАЛИЗ И РЕКОМЕНДАЦИИ ОТ GEMINI:")
                            print(f"   {'─'*76}")
                            # Выводим анализ с отступами для читаемости
                            analysis_lines = analysis_text.split('\n')
                            for line in analysis_lines[:50]:  # Первые 50 строк
                                print(f"   {line}")
                            if len(analysis_lines) > 50:
                                print(f"   ... (ещё {len(analysis_lines) - 50} строк, полный текст сохранён в отчёте)")
                            print(f"   {'─'*76}")
                    else:
                        print(f"   ⚠️ Анализ через Gemini не удался: {gemini_analysis.get('message', 'Неизвестная ошибка')}")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при анализе через Gemini: {e}")
                    gemini_analysis = {
                        "status": "error",
                        "message": str(e)
                    }
            else:
                print(f"   ⚠️ GEMINI_API_KEY не установлен, пропускаем анализ через Gemini")
        elif use_gemini_analysis and analyze_errors_with_gemini is None:
            print(f"   ⚠️ Модуль gemini_analyzer не доступен, пропускаем анализ через Gemini")
        print()
        
        # Формируем дополнительную информацию о промптах для сохранения в отчёт
        if use_multi_agent:
            # Используем уже полученные workflow_prompts (избегаем дублирования вызова)
            prompt_info = workflow_prompts["prompt_info"]
        else:
            # Для одноагентного режима full_prompt_example уже создан выше
            prompt_info = None
        
        # Формируем итоговый результат
        # Создаем копию гиперпараметров для сохранения (чтобы гарантировать сохранение всех значений)
        hyperparameters_to_save = copy.deepcopy(hyperparameters)
        
        evaluation_result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_name": model_name,
            "interrupted": interrupted,
            "processed_count": len(results),
            "total_count": len(self.texts),
            "multi_agent_mode": multi_agent_mode if use_multi_agent else None,
            "gpu_info": gpu_info_before if not is_api_model else {"api": True},
            "gpu_memory_after_load_gb": memory_after_load["allocated"] if not is_api_model else 0.0,
            "gpu_memory_during_inference_gb": memory_during_inference_avg if not is_api_model else 0.0,
            "gpu_memory_during_inference_max_gb": memory_during_inference_max if not is_api_model else 0.0,
            "gpu_memory_during_inference_min_gb": memory_during_inference_min if not is_api_model else 0.0,
            "api_model": is_api_model,
            "average_response_time_seconds": avg_speed,
            "parsing_error_rate": parsing_error_rate,
            "parsing_errors_count": len(parsing_errors),
            "quality_metrics": quality_metrics,
            "hyperparameters": hyperparameters_to_save,
            "prompt_template": prompt_template.__name__ if hasattr(prompt_template, '__name__') else str(prompt_template) if not use_multi_agent else f"multi_agent_{multi_agent_mode}",
            "prompt_full_text": full_prompt_example,
            "prompt_info": prompt_info,
            "parsing_errors": parsing_errors,
            "total_samples": len(results),
            "valid_json_count": len(results) - invalid_count,
            "invalid_json_count": invalid_count,
            "gemini_analysis": gemini_analysis
        }
        
        # Сохраняем результаты
        print(f"💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
        self._save_results(evaluation_result, results)
        
        print(f"\n{'='*80}")
        if interrupted:
            print(f"⚠️  ОЦЕНКА ЗАВЕРШЕНА С ПРЕРЫВАНИЕМ")
        else:
            print(f"✅ ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"{'='*80}\n")
        
        return evaluation_result
    
    def _save_results(self, evaluation_result: Dict[str, Any], results: List[Dict[str, Any]]):
        """Сохраняет результаты в файлы"""
        timestamp = evaluation_result["timestamp"]
        model_name_safe = sanitize_filename(evaluation_result["model_name"])
        
        # Добавляем информацию о мультиагентном режиме в имя файла, если он используется
        multi_agent_mode = evaluation_result.get("multi_agent_mode")
        multi_agent_suffix = f"_{multi_agent_mode}" if multi_agent_mode else ""
        
        # Сохраняем детальные результаты
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, f"results_{model_name_safe}{multi_agent_suffix}_{timestamp}.csv")
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 Детальные результаты сохранены: {csv_path}")
        
        # Сохраняем метрики
        # Создаем копию для сохранения в JSON
        evaluation_result_for_json = copy.deepcopy(evaluation_result)
        quality_metrics_for_json = evaluation_result_for_json.get("quality_metrics")
        
        # Собираем все ошибки из quality_metrics (они уже в формате словарей)
        all_quality_errors = []
        if quality_metrics_for_json:
            for group in ["массовая доля", "прочее"]:
                if group in quality_metrics_for_json:
                    # Берем все ошибки (не только первые 10)
                    group_errors = quality_metrics_for_json[group].get("все_ошибки", [])
                    # Проверяем, что ошибки уже в формате словарей
                    for error in group_errors:
                        if isinstance(error, dict):
                            all_quality_errors.append(error)
                        else:
                            # Для обратной совместимости: преобразуем строку в словарь
                            all_quality_errors.append({"error": str(error)})
                    # Удаляем поле "все_ошибки" и "ошибки" перед сохранением в JSON (чтобы не дублировать)
                    quality_metrics_for_json[group].pop("все_ошибки", None)
                    quality_metrics_for_json[group].pop("ошибки", None)
        
        # Подготавливаем ошибки для сохранения
        parsing_errors_list = evaluation_result_for_json.get("parsing_errors", [])
        
        # Объединяем parsing_errors и quality_errors
        all_errors = parsing_errors_list + all_quality_errors
        
        # Группируем ошибки по текстам
        errors_by_text = {}  # {text_index: {"text": str, "response": str, "errors": [str]}}
        
        for error in all_errors:
            if isinstance(error, dict):
                text_idx = error.get("text_index", 0)
                text = error.get("text", "")
                response = error.get("response", "")
                error_msg = error.get("error", "")
                
                if text_idx not in errors_by_text:
                    errors_by_text[text_idx] = {
                        "text_index": text_idx,
                        "text": text,
                        "response": response,
                        "errors": []
                    }
                
                # Добавляем ошибку в список ошибок для этого текста
                if error_msg:
                    errors_by_text[text_idx]["errors"].append(error_msg)
                
                # Обновляем text и response, если они есть (могут быть разными для разных ошибок одного текста)
                if text and not errors_by_text[text_idx]["text"]:
                    errors_by_text[text_idx]["text"] = text
                if response and not errors_by_text[text_idx]["response"]:
                    errors_by_text[text_idx]["response"] = response
        
        # Преобразуем в список записей (каждая запись - текст с его ошибками)
        errors_for_save = list(errors_by_text.values())
        
        # Добавляем ошибки в результат для сохранения
        # Все ошибки сохраняются в структурированном виде: список записей {text_index, text, response, errors}
        evaluation_result_for_json["ошибки"] = errors_for_save
        
        metrics_path = os.path.join(self.output_dir, f"metrics_{model_name_safe}{multi_agent_suffix}_{timestamp}.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result_for_json, f, ensure_ascii=False, indent=2)
        print(f"💾 Метрики сохранены: {metrics_path}")
        print(f"   📋 Сохраненные гиперпараметры: {list(evaluation_result.get('hyperparameters', {}).keys())}")
        
        # Обновляем общий файл со всеми прогонами
        summary_path = os.path.join(self.output_dir, "evaluation_summary.jsonl")
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation_result, ensure_ascii=False) + '\n')
        print(f"💾 Результат добавлен в общий файл: {summary_path}")
        
        # Сохраняем ошибки качества в отдельный файл
        quality_metrics = evaluation_result.get("quality_metrics")
        if quality_metrics:
            errors_path = os.path.join(self.output_dir, f"quality_errors_{model_name_safe}{multi_agent_suffix}_{timestamp}.txt")
            with open(errors_path, 'w', encoding='utf-8') as f:
                f.write(f"Ошибки качества для модели: {evaluation_result['model_name']}\n")
                f.write(f"Дата: {timestamp}\n")
                f.write(f"{'='*80}\n\n")
                
                # Ошибки для группы "массовая доля" (используем все_ошибки, если есть, иначе ошибки)
                mass_dolya = quality_metrics.get('массовая доля', {})
                mass_errors = mass_dolya.get('все_ошибки', mass_dolya.get('ошибки', []))
                if mass_errors:
                    f.write(f"ОШИБКИ КАЧЕСТВА: МАССОВАЯ ДОЛЯ\n")
                    f.write(f"Всего ошибок: {len(mass_errors)}\n")
                    f.write(f"{'─'*80}\n")
                    for i, error in enumerate(mass_errors, 1):
                        f.write(f"{i}. {error}\n")
                    f.write(f"\n")
                else:
                    f.write(f"ОШИБКИ КАЧЕСТВА: МАССОВАЯ ДОЛЯ\n")
                    f.write(f"Ошибок не обнаружено.\n\n")
                
                # Ошибки для группы "прочее" (используем все_ошибки, если есть, иначе ошибки)
                prochee = quality_metrics.get('прочее', {})
                prochee_errors = prochee.get('все_ошибки', prochee.get('ошибки', []))
                if prochee_errors:
                    f.write(f"ОШИБКИ КАЧЕСТВА: ПРОЧЕЕ\n")
                    f.write(f"Всего ошибок: {len(prochee_errors)}\n")
                    f.write(f"{'─'*80}\n")
                    for i, error in enumerate(prochee_errors, 1):
                        f.write(f"{i}. {error}\n")
                    f.write(f"\n")
                else:
                    f.write(f"ОШИБКИ КАЧЕСТВА: ПРОЧЕЕ\n")
                    f.write(f"Ошибок не обнаружено.\n\n")
            
            print(f"💾 Ошибки качества сохранены: {errors_path}")
    
    @staticmethod
    def reevaluate_from_file(
        results_csv_path: str,
        dataset_path: str,
        output_dir: str = "results",
        model_name: str = None,
        use_gemini_analysis: bool = False,
        gemini_api_key: str = None
    ) -> Dict[str, Any]:
        """
        Переоценивает результаты из сохраненного CSV файла без повторного запуска модели.
        
        Args:
            results_csv_path: путь к CSV файлу с результатами (например, results_model_name_timestamp.csv)
            dataset_path: путь к исходному датасету для получения ground truth
            output_dir: директория для сохранения обновленных результатов
            model_name: имя модели (если None, извлекается из имени файла)
        
        Returns:
            словарь с обновленными метриками
        """
        print(f"\n{'='*80}")
        print(f"🔄 ПЕРЕОЦЕНКА РЕЗУЛЬТАТОВ ИЗ ФАЙЛА")
        print(f"{'='*80}\n")
        
        # Загружаем результаты из CSV
        print(f"📂 Загрузка результатов из: {results_csv_path}")
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"Файл не найден: {results_csv_path}")
        
        df_results = pd.read_csv(results_csv_path)
        print(f"   • Загружено записей: {len(df_results)}")
        
        # Проверяем наличие необходимых колонок
        required_columns = ['text', 'json_parsed']
        missing_columns = [col for col in required_columns if col not in df_results.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
        
        # Загружаем ground truth из датасета
        print(f"📂 Загрузка ground truth из: {dataset_path}")
        df_full = pd.read_excel(dataset_path)
        
        if "json_parsed" not in df_full.columns:
            raise ValueError("В датасете отсутствует колонка 'json_parsed' с ground truth")
        
        # Извлекаем ground truth
        ground_truths = []
        for idx, row in df_full.iterrows():
            gt = row.get("json_parsed", {})
            if isinstance(gt, str):
                try:
                    gt = json.loads(gt)
                except:
                    gt = parse_json_safe(gt)
            ground_truths.append(gt if isinstance(gt, dict) else {})
        
        print(f"   • Загружено ground truth записей: {len(ground_truths)}")
        
        # Извлекаем предсказания из результатов
        # Если json_parsed пустой или невалидный, пытаемся перепарсить из колонки json
        predictions = []
        reparse_count = 0
        for idx, row in df_results.iterrows():
            pred = row.get("json_parsed", {})
            is_valid = row.get("is_valid", False)
            
            # Если json_parsed пустой или невалидный, пытаемся перепарсить из колонки json
            pred_is_valid = pred and isinstance(pred, dict) and len(pred) > 0
            if not pred_is_valid or not is_valid:
                json_str = row.get("json", "")
                if json_str:
                    # Используем улучшенные функции парсинга
                    extracted_json = extract_json_from_response(str(json_str))
                    new_pred = parse_json_safe(extracted_json)
                    if new_pred and isinstance(new_pred, dict) and len(new_pred) > 0:
                        pred = new_pred
                        reparse_count += 1
                    elif not new_pred:
                        # Если не удалось распарсить через extract_json_from_response, пробуем parse_json_safe напрямую
                        new_pred = parse_json_safe(str(json_str))
                        if new_pred and isinstance(new_pred, dict) and len(new_pred) > 0:
                            pred = new_pred
                            reparse_count += 1
            
            # Если pred все еще строка, пытаемся её распарсить
            if isinstance(pred, str):
                try:
                    pred = json.loads(pred)
                except:
                    pred = parse_json_safe(pred)
            
            predictions.append(pred if isinstance(pred, dict) else {})
        
        if reparse_count > 0:
            print(f"   • Перепарсено {reparse_count} JSON из колонки json с использованием улучшенных функций")
        
        # Проверяем соответствие количества
        if len(predictions) != len(ground_truths):
            print(f"⚠️  Предупреждение: количество предсказаний ({len(predictions)}) не совпадает с количеством ground truth ({len(ground_truths)})")
            min_len = min(len(predictions), len(ground_truths))
            predictions = predictions[:min_len]
            ground_truths = ground_truths[:min_len]
            print(f"   • Используется {min_len} записей")
        
        # Пересчитываем метрики качества
        print(f"\n📊 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА...")
        try:
            # В reevaluate нет доступа к текстам и ответам, передаем None
            quality_metrics = calculate_quality_metrics(predictions, ground_truths, texts=None, responses=None)
            print(f"✅ Метрики успешно вычислены")
        except Exception as e:
            print(f"⚠️  Ошибка при вычислении метрик качества: {e}")
            import traceback
            traceback.print_exc()
            quality_metrics = None
        
        # Подсчитываем статистику по парсингу
        valid_count = sum(1 for p in predictions if p and isinstance(p, dict))
        invalid_count = len(predictions) - valid_count
        parsing_error_rate = invalid_count / len(predictions) if predictions else 0.0
        
        # Извлекаем parsing errors из CSV, если есть (только для тех, что все еще невалидны после перепарсинга)
        parsing_errors = []
        if "json" in df_results.columns:
            for idx, (_, row) in enumerate(df_results.iterrows()):
                pred = predictions[idx] if idx < len(predictions) else {}
                # Проверяем, является ли предсказание валидным после перепарсинга
                if not pred or not isinstance(pred, dict) or len(pred) == 0:
                    json_str = row.get("json", "")
                    if json_str:
                        # Обрезаем длинные сообщения для компактности
                        json_display = str(json_str)[:500] if len(str(json_str)) > 500 else str(json_str)
                        parsing_errors.append(f"Текст #{idx}: невалидный JSON. Ответ: {json_display}")
        
        # Извлекаем имя модели из имени файла, если не указано
        if model_name is None:
            filename = os.path.basename(results_csv_path)
            # Формат: results_model_name_timestamp.csv
            parts = filename.replace("results_", "").replace(".csv", "").split("_")
            if len(parts) >= 2:
                # Берем все части кроме последней (timestamp)
                model_name = "_".join(parts[:-1])
            else:
                model_name = "unknown"
        
        # Анализ через Gemini API (если включен)
        gemini_analysis = None
        if use_gemini_analysis and analyze_errors_with_gemini is not None:
            if gemini_api_key is None:
                gemini_api_key = os.environ.get("GEMINI_API_KEY")
            
            if gemini_api_key:
                print(f"\n🤖 ЗАПУСК АНАЛИЗА ЧЕРЕЗ GEMINI API...")
                try:
                    # Для reevaluate нам нужны гиперпараметры - пытаемся загрузить из метрик, если есть
                    # Или создаем минимальный набор
                    hyperparameters = {"reevaluated": True}
                    
                    # Пытаемся загрузить гиперпараметры из исходного файла метрик, если он существует
                    metrics_file_pattern = f"metrics_{sanitize_filename(model_name)}_*.json"
                    metrics_files = glob.glob(os.path.join(os.path.dirname(results_csv_path), metrics_file_pattern))
                    if metrics_files:
                        # Берем последний файл метрик
                        try:
                            with open(metrics_files[-1], 'r', encoding='utf-8') as f:
                                existing_metrics = json.load(f)
                                hyperparameters = existing_metrics.get("hyperparameters", hyperparameters)
                        except:
                            pass
                    
                    gemini_analysis = analyze_errors_with_gemini(
                        model_name=model_name,
                        parsing_errors=parsing_errors,
                        quality_metrics=quality_metrics or {},
                        hyperparameters=hyperparameters,
                        prompt_full_text=None,  # Для reevaluate промпт недоступен
                        gemini_api_key=gemini_api_key
                    )
                    
                    if gemini_analysis.get("status") == "success":
                        print(f"   ✅ Анализ от Gemini получен успешно!")
                        analysis_text = gemini_analysis.get("analysis", "")
                        if analysis_text:
                            print(f"\n   {'─'*76}")
                            print(f"   📝 АНАЛИЗ И РЕКОМЕНДАЦИИ ОТ GEMINI:")
                            print(f"   {'─'*76}")
                            # Выводим анализ с отступами для читаемости
                            analysis_lines = analysis_text.split('\n')
                            for line in analysis_lines[:50]:  # Первые 50 строк
                                print(f"   {line}")
                            if len(analysis_lines) > 50:
                                print(f"   ... (ещё {len(analysis_lines) - 50} строк, полный текст сохранён в отчёте)")
                            print(f"   {'─'*76}")
                    else:
                        print(f"   ⚠️ Анализ через Gemini не удался: {gemini_analysis.get('message', 'Неизвестная ошибка')}")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при анализе через Gemini: {e}")
                    gemini_analysis = {
                        "status": "error",
                        "message": str(e)
                    }
            else:
                print(f"   ⚠️ GEMINI_API_KEY не установлен, пропускаем анализ через Gemini")
        elif use_gemini_analysis and analyze_errors_with_gemini is None:
            print(f"   ⚠️ Модуль gemini_analyzer не доступен, пропускаем анализ через Gemini")
        print()
        
        # Формируем обновленный результат
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_result = {
            "timestamp": timestamp,
            "model_name": model_name,
            "reevaluated_from": results_csv_path,
            "parsing_error_rate": parsing_error_rate,
            "parsing_errors_count": len(parsing_errors),
            "quality_metrics": quality_metrics,
            "parsing_errors": parsing_errors,
            "total_samples": len(predictions),
            "valid_json_count": valid_count,
            "invalid_json_count": invalid_count,
            "gemini_analysis": gemini_analysis
        }
        
        # Сохраняем обновленные метрики
        print(f"\n💾 СОХРАНЕНИЕ ОБНОВЛЕННЫХ РЕЗУЛЬТАТОВ...")
        os.makedirs(output_dir, exist_ok=True)
        
        model_name_safe = sanitize_filename(model_name)
        
        # Пытаемся найти исходный файл метрик для извлечения multi_agent_mode
        multi_agent_mode = None
        metrics_file_pattern = os.path.join(output_dir, f"metrics_{model_name_safe}_*.json")
        metrics_files = glob.glob(metrics_file_pattern)
        original_metrics_files = [f for f in metrics_files if "_reevaluated" not in f]
        if original_metrics_files:
            try:
                with open(original_metrics_files[-1], 'r', encoding='utf-8') as f:
                    original_metrics = json.load(f)
                multi_agent_mode = original_metrics.get("multi_agent_mode")
            except Exception:
                pass  # Если не удалось загрузить, просто пропускаем
        
        # Добавляем информацию о мультиагентном режиме в имя файла, если он используется
        multi_agent_suffix = f"_{multi_agent_mode}" if multi_agent_mode else ""
        metrics_path = os.path.join(output_dir, f"metrics_{model_name_safe}{multi_agent_suffix}_{timestamp}_reevaluated.json")
        
        # Создаем копию для сохранения в JSON без поля "все_ошибки" (чтобы не перегружать файл)
        evaluation_result_for_json = copy.deepcopy(evaluation_result)
        quality_metrics_for_json = evaluation_result_for_json.get("quality_metrics")
        if quality_metrics_for_json:
            for group in ["массовая доля", "прочее"]:
                if group in quality_metrics_for_json:
                    # Удаляем поле "все_ошибки" перед сохранением в JSON
                    quality_metrics_for_json[group].pop("все_ошибки", None)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result_for_json, f, ensure_ascii=False, indent=2)
        print(f"💾 Обновленные метрики сохранены: {metrics_path}")
        
        # Сохраняем ошибки качества в отдельный файл
        if quality_metrics:
            errors_path = os.path.join(output_dir, f"quality_errors_{model_name_safe}{multi_agent_suffix}_{timestamp}_reevaluated.txt")
            with open(errors_path, 'w', encoding='utf-8') as f:
                f.write(f"Ошибки качества для модели: {model_name}\n")
                f.write(f"Дата: {timestamp}\n")
                f.write(f"Переоценено из: {results_csv_path}\n")
                f.write(f"{'='*80}\n\n")
                
                # Ошибки для группы "массовая доля" (используем все_ошибки, если есть, иначе ошибки)
                mass_dolya = quality_metrics.get('массовая доля', {})
                mass_errors = mass_dolya.get('все_ошибки', mass_dolya.get('ошибки', []))
                if mass_errors:
                    f.write(f"ОШИБКИ КАЧЕСТВА: МАССОВАЯ ДОЛЯ\n")
                    f.write(f"Всего ошибок: {len(mass_errors)}\n")
                    f.write(f"{'─'*80}\n")
                    for i, error in enumerate(mass_errors, 1):
                        f.write(f"{i}. {error}\n")
                    f.write(f"\n")
                else:
                    f.write(f"ОШИБКИ КАЧЕСТВА: МАССОВАЯ ДОЛЯ\n")
                    f.write(f"Ошибок не обнаружено.\n\n")
                
                # Ошибки для группы "прочее" (используем все_ошибки, если есть, иначе ошибки)
                prochee = quality_metrics.get('прочее', {})
                prochee_errors = prochee.get('все_ошибки', prochee.get('ошибки', []))
                if prochee_errors:
                    f.write(f"ОШИБКИ КАЧЕСТВА: ПРОЧЕЕ\n")
                    f.write(f"Всего ошибок: {len(prochee_errors)}\n")
                    f.write(f"{'─'*80}\n")
                    for i, error in enumerate(prochee_errors, 1):
                        f.write(f"{i}. {error}\n")
                    f.write(f"\n")
                else:
                    f.write(f"ОШИБКИ КАЧЕСТВА: ПРОЧЕЕ\n")
                    f.write(f"Ошибок не обнаружено.\n\n")
            
            print(f"💾 Ошибки качества сохранены: {errors_path}")
        
        # Выводим сводку
        print(f"\n{'='*80}")
        print(f"✅ ПЕРЕОЦЕНКА ЗАВЕРШЕНА!")
        print(f"{'='*80}")
        print(f"📌 Итоговая сводка:")
        print(f"   • Модель: {model_name}")
        print(f"   • Обработано текстов: {len(predictions)}")
        print(f"   • Ошибки парсинга: {parsing_error_rate:.2%} ({invalid_count}/{len(predictions)})")
        if quality_metrics:
            mass_acc = quality_metrics.get('массовая доля', {}).get('accuracy', 0)
            prochee_acc = quality_metrics.get('прочее', {}).get('accuracy', 0)
            mass_f1 = quality_metrics.get('массовая доля', {}).get('f1', 0)
            prochee_f1 = quality_metrics.get('прочее', {}).get('f1', 0)
            print(f"   • Качество 'массовая доля': Accuracy={mass_acc:.2%}, F1={mass_f1:.2%}")
            print(f"   • Качество 'прочее': Accuracy={prochee_acc:.2%}, F1={prochee_f1:.2%}")
        print(f"{'='*80}\n")
        
        return evaluation_result

