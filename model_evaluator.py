"""
Основной класс для оценки моделей LLM
"""
import torch
import gc
import time
import logging
import pandas as pd
import json
import copy
import glob
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import os

import inspect
from utils import build_prompt3, parse_json_safe, is_valid_json, extract_json_from_response, get_few_shot_examples_path
from structured_schemas import latin_to_cyrillic_output, latin_keys_to_cyrillic_in_json_str, LATIN_TO_CYRILLIC_KEYS
from metrics import calculate_quality_metrics, validate_with_pydantic, calculate_raw_output_metrics
from gpu_info import get_gpu_info, get_gpu_memory_usage
from multi_agent_graph import process_with_multi_agent
from config import PROMPT_TEMPLATE_NAME, MAX_INFERENCE_TIME_MINUTES, OUTPUT_DIR
import prompt_config
from metrics_printer import MetricsPrinter
from file_manager import FileManager
import re


class _MultiAgentGenerator:
    """
    Обёртка над generate_func с теми же гиперпараметрами, что и в стандартном режиме.
    Для multi_agent по умолчанию отключает enable_thinking, чтобы избежать обрывков/режима thinking.
    """
    def __init__(self, model, tokenizer, generate_func: Callable, hyperparameters: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.generate_func = generate_func
        self.hp = hyperparameters or {}
        self._sig = inspect.signature(generate_func).parameters

    def generate(self, prompt: str, max_new_tokens: int = 1792, **kwargs) -> str:
        optional = {}
        if "repetition_penalty" in self._sig:
            optional["repetition_penalty"] = self.hp.get("repetition_penalty") or kwargs.get("repetition_penalty")
        if "max_length" in self._sig:
            optional["max_length"] = self.hp.get("max_length") or kwargs.get("max_length")
        if "enable_thinking" in self._sig:
            optional["enable_thinking"] = self.hp.get("enable_thinking", False)
        optional = {k: v for k, v in optional.items() if v is not None or k == "enable_thinking"}
        return self.generate_func(
            self.model, self.tokenizer, prompt, max_new_tokens, **optional
        )


try:
    from gemini_analyzer import analyze_errors_with_gemini
except ImportError:
    analyze_errors_with_gemini = None


def _is_outlines_vocabulary_error(exc: BaseException) -> bool:
    """Проверяет, является ли ошибка outlines: vocabulary/encoding (пробелы, токенизатор)."""
    msg = str(exc).lower()
    if "vocabulary" in msg and ("incompatible" in msg or "incompat" in msg):
        return True
    if "encoding issue" in msg and "vocabulary" in msg:
        return True
    if "found no transitions" in msg and "missing tokens" in msg:
        return True
    return False


# Контекст текущей оценки: предупреждения во время evaluate_model пишутся в model_errors.log с указанием модели
_eval_warning_context: Dict[str, Optional[str]] = {"model_name": None, "output_dir": None}
_prev_showwarning = None


def _eval_showwarning(message, category, filename, lineno, file=None, line=None):
    """Перехват предупреждений во время оценки: запись в model_errors.log с именем модели."""
    ctx = _eval_warning_context
    if ctx.get("output_dir") and ctx.get("model_name"):
        _append_to_model_errors_log(
            ctx["output_dir"],
            "WARNING (во время оценки)",
            ctx["model_name"],
            f"{category.__name__ if category else 'Warning'}: {message}",
        )
    if _prev_showwarning is not None:
        _prev_showwarning(message, category, filename, lineno, file, line)


def _append_to_model_errors_log(output_dir: str, title: str, model_name: str, message: str) -> None:
    """Дописывает ошибку или предупреждение в model_errors.log."""
    log_path = os.path.join(output_dir, "model_errors.log")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'='*80}\n"
                f"{title}\n"
                f"{'='*80}\n"
                f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Модель: {model_name}\n"
                f"Сообщение: {message}\n"
                f"{'='*80}\n"
            )
    except Exception:
        pass


class InferenceCriticalFailure(Exception):
    """Выбрасывается при исчерпании всех попыток генерации на одном примере; оценку модели завершаем досрочно."""
    def __init__(self, message: str, text_index: int, num_retries: int):
        self.message = message
        self.text_index = text_index
        self.num_retries = num_retries
        super().__init__(message)


class StopAllModelsInterrupt(Exception):
    """Выбрасывается при выборе пункта 4 (прервать оценку всех моделей); должна поймать run_all_models и выйти из цикла по моделям."""
    pass


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
        # Создаем FileManager для работы с файлами
        self.file_manager = FileManager()
        self.file_manager.ensure_directory(output_dir)
        
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
    
    def _generate_response_with_retries(self, 
                                       model, tokenizer, prompt, generate_func,
                                       hyperparameters, max_new_tokens, num_retries,
                                       is_api_model, verbose, text_index, total_texts, text,
                                       times, memory_samples, parsing_errors, model_name: str = "unknown"):
        """
        Генерирует ответ модели с повторными попытками при ошибках.
        
        Args:
            text: исходный текст (для сохранения в ошибках)
            model_name: имя модели для логирования
        
        Returns:
            tuple: (response_text, elapsed_time, error_msg, outlines_skip)
            outlines_skip=True — outlines упал (vocabulary/encoding), вернули пустой ответ, лог записан
        """
        response_text = ""
        error_msg = None
        is_ollama = bool(hyperparameters.get("ollama", False))
        is_vllm = bool(hyperparameters.get("vllm", False))
        
        for attempt in range(num_retries):
            try:
                start_time = time.time()
                # Извлекаем параметры для structured output и outlines
                structured_output = hyperparameters.get("structured_output", False)
                use_outlines = hyperparameters.get("use_outlines", False)
                use_guidance = hyperparameters.get("use_guidance", False)
                response_schema = None
                # response_schema: общая схема с кириллическими alias для всех режимов
                if use_guidance:
                    from structured_schemas import FertilizerExtractionOutput
                    response_schema = FertilizerExtractionOutput
                elif use_outlines or structured_output:
                    from structured_schemas import FertilizerExtractionOutput
                    response_schema = FertilizerExtractionOutput
                
                # Передаем repetition_penalty и max_length из гиперпараметров, если есть
                repetition_penalty = hyperparameters.get("repetition_penalty")
                max_length = hyperparameters.get("max_length")
                pt_name = hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME

                # vLLM: OpenAI-совместимый API (сервер запускается отдельно)
                if is_vllm:
                    response_text = generate_func(
                        model, tokenizer, prompt, max_new_tokens,
                        hyperparameters=hyperparameters,
                        repetition_penalty=repetition_penalty,
                        max_length=max_length,
                    )
                # Ollama: те же гиперпараметры, что и у локальной генерации (temperature, repeat_penalty, num_ctx)
                elif is_ollama:
                    response_text = generate_func(
                        model, tokenizer, prompt, max_new_tokens,
                        hyperparameters=hyperparameters,
                        repetition_penalty=repetition_penalty,
                        max_length=max_length,
                    )
                # Для API моделей передаем model_name и structured_output из hyperparameters
                elif is_api_model and "model_name" in hyperparameters:
                    response_text = generate_func(
                        model, tokenizer, prompt, max_new_tokens,
                        model_name=hyperparameters["model_name"],
                        structured_output=structured_output,
                        response_schema=response_schema,
                        max_length=max_length
                    )
                # Для локальных моделей с guidance (llguidance, по умолчанию схема RUS)
                elif use_guidance and not is_api_model and response_schema is not None:
                    response_text = generate_func(
                        model, tokenizer, prompt, max_new_tokens,
                        structured_output=True,
                        response_schema=response_schema,
                        use_guidance=True,
                        prompt_template_name=pt_name,
                        max_length=max_length
                    )
                # Для локальных моделей с outlines (response_schema используется outlines; structured_output добавляет схему в промпт)
                elif use_outlines and not is_api_model and response_schema is not None:
                    response_text = generate_func(
                        model, tokenizer, prompt, max_new_tokens,
                        structured_output=structured_output,
                        response_schema=response_schema,
                        use_outlines=True,
                        prompt_template_name=pt_name,
                        pydantic_outlines=hyperparameters.get("pydantic_outlines", False),
                        max_length=max_length
                    )
                # Для локальных моделей с structured_output (без outlines)
                elif structured_output and not is_api_model and response_schema is not None:
                    response_text = generate_func(
                        model, tokenizer, prompt, max_new_tokens,
                        structured_output=structured_output,
                        response_schema=response_schema,
                        use_outlines=False,
                        max_length=max_length
                    )
                elif repetition_penalty is not None:
                    response_text = generate_func(model, tokenizer, prompt, max_new_tokens, repetition_penalty=repetition_penalty, max_length=max_length)
                elif "enable_thinking" in hyperparameters:
                    # Для Qwen3 передаем enable_thinking из hyperparameters (по умолчанию False)
                    enable_thinking_value = hyperparameters.get("enable_thinking", False)
                    response_text = generate_func(model, tokenizer, prompt, max_new_tokens, enable_thinking=enable_thinking_value, max_length=max_length)
                else:
                    response_text = generate_func(model, tokenizer, prompt, max_new_tokens, max_length=max_length)
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Только локальный PyTorch: для API / Ollama / vLLM память берётся отдельно (ниже по коду)
                if not is_api_model and not is_ollama and not is_vllm:
                    memory_sample = get_gpu_memory_usage()
                    memory_samples.append(memory_sample["allocated"])
                
                return response_text, elapsed, None, False

            except KeyboardInterrupt:
                raise
            except Exception as e:
                error_msg = str(e)
                use_outlines = hyperparameters.get("use_outlines", False)
                use_guidance = hyperparameters.get("use_guidance", False)
                vocab_err = _is_outlines_vocabulary_error(e)
                if use_outlines and not is_api_model and vocab_err:
                    _append_to_model_errors_log(
                        self.output_dir,
                        title="ОШИБКА OUTLINES (VOCABULARY/ENCODING)",
                        model_name=model_name,
                        message=f"Ответ #{text_index+1}/{total_texts}: {error_msg}",
                    )
                    print(f"  ⚠️ Ответ #{text_index+1}/{total_texts} - Outlines vocabulary/encoding, пустой ответ, лог записан")
                    return "", 0, error_msg, True
                if use_guidance and not is_api_model and vocab_err:
                    _append_to_model_errors_log(
                        self.output_dir,
                        title="ОШИБКА GUIDANCE (VOCABULARY/ENCODING)",
                        model_name=model_name,
                        message=f"Ответ #{text_index+1}/{total_texts}: {error_msg}",
                    )
                    print(f"  ⚠️ Ответ #{text_index+1}/{total_texts} - Guidance vocabulary/encoding, пустой ответ, лог записан")
                    return "", 0, error_msg, True

                # 400 (bad request) у API — не повторяем
                if is_api_model and ("400" in error_msg or "Ошибка запроса (400)" in error_msg):
                    print(f"  ❌ Ответ #{text_index+1}/{total_texts} - Ошибка запроса (400), повторные попытки не выполняются")
                    parsing_errors.append({
                        "text_index": text_index,
                        "text": text,
                        "error": f"Ошибка запроса API (400): {error_msg}",
                        "response": ""
                    })
                    raise InferenceCriticalFailure(error_msg, text_index, 1)

                # Ошибка "модель не найдена" у Ollama — не повторяем (fail-fast),
                # иначе будет много бесполезных попыток для каждого текста.
                ollama_not_found = (
                    is_ollama
                    and (
                        ("ollama api error" in error_msg.lower() and "not found" in error_msg.lower())
                        or ("model" in error_msg.lower() and "not found" in error_msg.lower())
                        or ("model '" in error_msg.lower() and "' not found" in error_msg.lower())
                    )
                )
                if ollama_not_found:
                    print(
                        f"  ❌ Ответ #{text_index+1}/{total_texts} - Ollama модель не найдена, "
                        f"повторные попытки не выполняются"
                    )
                    parsing_errors.append({
                        "text_index": text_index,
                        "text": text,
                        "error": f"Ollama модель не найдена: {error_msg}",
                        "response": ""
                    })
                    raise InferenceCriticalFailure(error_msg, text_index, 1)

                vllm_not_found = (
                    is_vllm
                    and (
                        "404" in error_msg
                        or "model_not_found" in error_msg.lower()
                        or ("model" in error_msg.lower() and "not found" in error_msg.lower())
                    )
                )
                if vllm_not_found:
                    print(
                        f"  ❌ Ответ #{text_index+1}/{total_texts} - vLLM: модель не найдена на сервере, "
                        f"повторные попытки не выполняются"
                    )
                    parsing_errors.append({
                        "text_index": text_index,
                        "text": text,
                        "error": f"vLLM модель не найдена: {error_msg}",
                        "response": ""
                    })
                    raise InferenceCriticalFailure(error_msg, text_index, 1)

                # Сервер vLLM не слушает порт / сеть — повторы не помогут.
                vllm_unreachable = is_vllm and (
                    "Connection refused" in error_msg
                    or "[Errno 111]" in error_msg
                    or "Errno 10061" in error_msg
                    or "[WinError 10061]" in error_msg
                )
                if vllm_unreachable:
                    print(
                        f"  ❌ Ответ #{text_index+1}/{total_texts} - vLLM: нет соединения с сервером, "
                        f"повторные попытки не выполняются"
                    )
                    for line in (error_msg.splitlines() or [error_msg]):
                        print(f"     {line}")
                    parsing_errors.append({
                        "text_index": text_index,
                        "text": text,
                        "error": error_msg,
                        "response": "",
                    })
                    raise InferenceCriticalFailure(error_msg, text_index, 1)

                if is_api_model or is_vllm:
                    print(f"  ⚠️ Ответ #{text_index+1}/{total_texts} - Ошибка при генерации (попытка {attempt+1}/{num_retries}):")
                    for line in (error_msg.splitlines() or [error_msg]):
                        print(f"     {line}")
                else:
                    error_display = error_msg if verbose else error_msg[:100]
                    print(f"  ⚠️ Ответ #{text_index+1}/{total_texts} - Ошибка при генерации (попытка {attempt+1}/{num_retries}): {error_display}")
                if attempt < num_retries - 1:
                    time.sleep(4 + attempt * 2)
                else:
                    import traceback
                    traceback_str = traceback.format_exc()
                    traceback_display = traceback_str if (is_api_model or is_vllm) else traceback_str[:200]
                    parsing_errors.append({
                        "text_index": text_index,
                        "text": text,
                        "error": f"Критическая ошибка генерации после {num_retries} попыток: {error_msg}. Traceback: {traceback_display}",
                        "response": ""
                    })
                    raise InferenceCriticalFailure(error_msg, text_index, num_retries)

        return None, 0, error_msg, False
    
    def _print_verbose_output(self, text, response_text, is_api_model, text_index, total_texts):
        """Выводит исходный текст и полный ответ в консоль (только при verbose)"""
        print(f"\n   Ответ #{text_index + 1}/{total_texts} - Исходный текст для анализа:")
        print(f"   {'-'*76}")
        for line in text.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*76}")
        model_type_label = "API модели" if is_api_model else "модели"
        print(f"   Ответ #{text_index + 1}/{total_texts} - Полный ответ {model_type_label}:")
        print(f"   {'-'*76}")
        for line in response_text.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*76}")
        # Если ответ содержит латинские ключи (mass_fractions, other_params), выводим версию на кириллице
        try:
            json_part = extract_json_from_response(response_text)
            parsed = parse_json_safe(json_part)
            if parsed and isinstance(parsed, dict) and any(k in LATIN_TO_CYRILLIC_KEYS for k in parsed):
                converted = latin_to_cyrillic_output(parsed)
                cyrillic_str = json.dumps(converted, ensure_ascii=False, indent=2)
                print(f"   Версия на кириллице:")
                print(f"   {'-'*76}")
                for line in cyrillic_str.split('\n'):
                    print(f"   {line}")
                print(f"   {'-'*76}")
        except Exception:
            pass
    
    @staticmethod
    def _drop_empty_and_zero_values(obj):
        """
        Рекурсивно удаляет из словарей ключи со значениями [], '' и None.
        После такой очистки Pydantic не падает на полях типа \"значение\": [].
        """
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if v == [] or v == "" or v is None:
                    continue
                out[k] = ModelEvaluator._drop_empty_and_zero_values(v)
            return out
        if isinstance(obj, list):
            return [ModelEvaluator._drop_empty_and_zero_values(item) for item in obj]
        return obj

    def _clean_parsed_json(self, parsed_json):
        """
        Удаляет из parsed_json записи с None или [None, None] значениями,
        а также рекурсивно удаляет атрибуты с пустыми/нулевыми значениями ([], 0, '').
        Такие значения заставляют Pydantic падать (ожидается str/float, а не list).
        
        Args:
            parsed_json: распарсенный JSON словарь
            
        Returns:
            dict: очищенный JSON словарь
        """
        if not isinstance(parsed_json, dict):
            return parsed_json
        
        cleaned = {}
        
        # Обрабатываем "массовая доля"
        if "массовая доля" in parsed_json:
            mass_fractions = parsed_json["массовая доля"]
            if isinstance(mass_fractions, list):
                cleaned_mass = []
                for item in mass_fractions:
                    if isinstance(item, dict):
                        # Проверяем значение "массовая доля"
                        mass_value = item.get("массовая доля")
                        # Пропускаем записи с None
                        if mass_value is None:
                            continue
                        # Пропускаем [None, None]
                        if isinstance(mass_value, list) and len(mass_value) == 2:
                            if mass_value[0] is None and mass_value[1] is None:
                                continue
                        # Если все значения None, пропускаем
                        if isinstance(mass_value, list) and all(v is None for v in mass_value):
                            continue
                        cleaned_mass.append(item)
                    else:
                        # Если это не словарь, оставляем как есть
                        cleaned_mass.append(item)
                cleaned["массовая доля"] = cleaned_mass
            else:
                cleaned["массовая доля"] = mass_fractions
        else:
            # Если ключа нет, не добавляем его
            pass
        
        # Обрабатываем "прочее"
        if "прочее" in parsed_json:
            other_params = parsed_json["прочее"]
            if isinstance(other_params, list):
                cleaned_other = []
                for item in other_params:
                    if isinstance(item, dict):
                        # Проверяем все значения в словаре
                        has_valid_value = False
                        for key, value in item.items():
                            if value is None:
                                continue
                            if isinstance(value, list):
                                # Пропускаем списки из None
                                if all(v is None for v in value):
                                    continue
                                # Пропускаем [None, None]
                                if len(value) == 2 and value[0] is None and value[1] is None:
                                    continue
                            # Если есть хотя бы одно непустое значение, оставляем запись
                            if value is not None and value != "":
                                has_valid_value = True
                                break
                        if has_valid_value:
                            cleaned_other.append(item)
                    else:
                        # Если это не словарь, оставляем как есть
                        cleaned_other.append(item)
                cleaned["прочее"] = cleaned_other
            else:
                cleaned["прочее"] = other_params
        else:
            # Если ключа нет, не добавляем его
            pass
        
        # Копируем остальные ключи, если есть
        for key in parsed_json:
            if key not in ["массовая доля", "прочее"]:
                cleaned[key] = parsed_json[key]
        
        # Рекурсивно убираем атрибуты с [], 0, 0.0, '' — иначе Pydantic выдаёт ошибки типа float_type/string_type
        return self._drop_empty_and_zero_values(cleaned)
    
    def _process_response(self, response_text, text, text_index, is_api_model, verbose, parsing_errors):
        """
        Обрабатывает ответ модели: валидация, парсинг JSON, извлечение данных.
        
        Returns:
            dict: словарь с результатами обработки
        """
        # Валидация raw output через Pydantic (этап 1)
        raw_validation = validate_with_pydantic(response_text, stage="raw")
        
        # Извлекаем JSON
        json_part = extract_json_from_response(response_text)
        parsed_json = parse_json_safe(json_part)
        is_valid = is_valid_json(json_part)
        # Если распарсенный JSON содержит латинские ключи (mass_fractions, other_params), приводим к кириллице
        if parsed_json and isinstance(parsed_json, dict) and any(k in LATIN_TO_CYRILLIC_KEYS for k in parsed_json):
            parsed_json = latin_to_cyrillic_output(parsed_json)
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass
        elif not parsed_json and any(f'"{k}"' in json_part for k in LATIN_TO_CYRILLIC_KEYS):
            # Невалидный/обрезанный JSON: парсинг не удался, но в строке есть латинские ключи — заменяем на кириллицу
            json_part = latin_keys_to_cyrillic_in_json_str(json_part)
        # Очищаем parsed_json от записей с None или [None, None]
        if parsed_json and isinstance(parsed_json, dict):
            parsed_json = self._clean_parsed_json(parsed_json)
            # Обновляем json_part после очистки
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass  # Если не удалось сериализовать, оставляем исходный json_part
        
        # Валидация после парсинга через Pydantic (этап 2)
        parsed_validation = validate_with_pydantic(parsed_json, stage="parsed")
        
        if not is_valid:
            # Для API моделей при verbose выводим полный JSON, иначе обрезаем
            json_display = json_part if (is_api_model and verbose) else (json_part[:200] if len(json_part) > 200 else json_part)
            parsing_errors.append({
                "text_index": text_index,
                "text": text,
                "error": f"Невалидный JSON. Ответ: {json_display}",
                "response": json_part[:500]
            })
        
        return {
            "text": text,
            "json": json_part,
            "json_parsed": parsed_json,
            "is_valid": is_valid,
            "raw_output": response_text,  # Сохраняем raw output для метрик
            "raw_validation": raw_validation,  # Результат валидации raw output
            "parsed_validation": parsed_validation  # Результат валидации после парсинга
        }
    
    def _handle_no_response(self, text, text_index, total_texts, error_msg, is_api_model, verbose, parsing_errors):
        """
        Обрабатывает случай, когда ответ не был получен.
        
        Returns:
            dict: словарь с пустым результатом
        """
        print(f"  ❌ Ответ #{text_index+1}/{total_texts} - Ответ не получен — пропуск")
        if error_msg:
            # Для API моделей выводим полную ошибку без обрезки (всегда, так как это ошибка)
            if is_api_model:
                print(f"     Последняя ошибка: {error_msg}")
            else:
                # Для локальных моделей обрезаем при не verbose режиме
                error_display = error_msg if verbose else error_msg[:200]
                print(f"     Последняя ошибка: {error_display}")
        parsing_errors.append(f"Текст #{text_index}: не получен ответ. Ошибка: {error_msg if error_msg else 'Неизвестная ошибка'}")
        empty_validation = {"is_valid": False, "errors": [], "validated_data": None}
        return {
            "text": text,
            "json": "",
            "json_parsed": {},
            "is_valid": False,
            "raw_output": "",
            "raw_validation": empty_validation,
            "parsed_validation": empty_validation,
        }
    
    def _print_progress(self, i, total_texts, results, times, total_start_time, verbose):
        """Выводит прогресс обработки"""
        elapsed_total = time.time() - total_start_time
        avg_time = sum(times) / len(times) if times else 0
        progress_pct = ((i + 1) / total_texts) * 100
        remaining = total_texts - (i + 1)
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
                f"  ✅ Ответ #{i + 1}/{total_texts} обработан ({progress_pct:.1f}%) | "
                f"Валидных: {valid_count} | Невалидных: {invalid_count} | "
                f"ETA: {eta_str}"
            )
            print(status_line)
        else:
            # Короткий вывод при verbose=False (только счетчик и основные метрики)
            status_line = (
                f"  Ответ #{i + 1}/{total_texts} | "
                f"✓: {valid_count} ✗: {invalid_count} | "
                f"ETA: {eta_str}"
            )
            print(f"\r{status_line}", end="", flush=True)
        
        # Подробный прогресс каждые 10 текстов или в конце (только при verbose)
        if verbose and ((i + 1) % 10 == 0 or (i + 1) == total_texts):
            print()  # Новая строка для подробного вывода
            print(f"     📊 Детальная статистика:")
            print(f"        • Прогресс: {progress_pct:.1f}% ({i + 1}/{total_texts})")
            print(f"        • Валидных JSON: {valid_count} | Невалидных: {invalid_count}")
            print(f"        • Средняя скорость: {avg_time:.3f} сек/ответ")
            print(f"        • Прошло времени: {elapsed_total/60:.1f} мин | Осталось: ~{eta_minutes:.1f} мин")
            print()
    
    def evaluate_model(self,
                      model_name: str,
                      load_model_func: Callable,
                      generate_func: Callable,
                      hyperparameters: Dict[str, Any],
                      prompt_template: str = None,
                      max_new_tokens: int = 1792,
                      num_retries: int = 2,
                      verbose: bool = False,
                      use_gemini_analysis: bool = False,
                      gemini_api_key: str = None,
                      model_key: str = None,
                      stop_all_on_interrupt: bool = False) -> Dict[str, Any]:
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
        # max_new_tokens и max_length только из hyperparameters (models.yaml), без подстановок
        max_new_tokens = hyperparameters.get("max_new_tokens")
        max_length_hp = hyperparameters.get("max_length")
        if max_new_tokens is not None:
            print(f"   max_new_tokens для генерации: {max_new_tokens}")
        if max_length_hp is not None:
            print(f"   max_length для генерации: {max_length_hp}")

        # Определяем режим работы из гиперпараметров
        multi_agent_mode = hyperparameters.get("multi_agent_mode", None)
        use_multi_agent = multi_agent_mode is not None and multi_agent_mode != ""
        
        # Определяем, является ли модель API-моделью, Ollama или vLLM (удалённый инференс)
        is_api_model = hyperparameters.get("api_model", False)
        is_ollama = hyperparameters.get("ollama", False)
        is_vllm = hyperparameters.get("vllm", False)
        if not is_api_model and not is_ollama and not is_vllm:
            pass  # Локальная модель с весами
        
        # Устанавливаем num_retries для API, Ollama и vLLM (10 попыток)
        if is_api_model or is_ollama or is_vllm:
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
        
        # Проверка существования промпта (до загрузки модели)
        effective_prompt_name = hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME
        if not hasattr(prompt_config, effective_prompt_name):
            err = f"Промпт '{effective_prompt_name}' не найден в prompt_config.py. Оценка не запущена."
            print(f"\n{'='*80}")
            print("ОШИБКА КОНФИГУРАЦИИ")
            print(f"{'='*80}")
            print(f"   {err}")
            print(f"{'='*80}\n")
            return {
                "status": "error",
                "error": err,
            }

        # Режим MINIMAL_INSTR_FIVESHOT_APIE (APIE) требует model_key и файл с примерами (XLSX или CSV) для данной модели
        if effective_prompt_name == "MINIMAL_INSTR_FIVESHOT_APIE":
            if not model_key or not model_key.strip():
                err = (
                    "Режим MINIMAL_INSTR_FIVESHOT_APIE (APIE) требует указания ключа модели (model_key). "
                    "Запуск оценки невозможен без передачи model_key в run_evaluation."
                )
            else:
                few_shot_path = get_few_shot_examples_path(model_key, OUTPUT_DIR)
                if not few_shot_path:
                    err = (
                        f"Режим MINIMAL_INSTR_FIVESHOT_APIE (APIE) требует файл с few-shot примерами для модели '{model_key}', "
                        f"но в директории '{OUTPUT_DIR}' не найден файл few_shot_examples_{model_key}_*.xlsx или few_shot_examples_{model_key}_*.csv. "
                        "Сгенерируйте примеры: python few_shot_extractor.py {model_key} [--n-examples 10]. "
                        "Для разметки через Gemini сохраните результат в XLSX: python label_few_shot_with_gemini.py <csv> --model-key {model_key}."
                    ).format(model_key=model_key)
                else:
                    err = None
            if err:
                print(f"\n{'='*80}")
                print("ОШИБКА: НЕВОЗМОЖЕН ЗАПУСК РЕЖИМА APIE")
                print(f"{'='*80}")
                print(f"   {err}")
                print(f"{'='*80}\n")
                return {
                    "status": "error",
                    "error": err,
                }

        # Проверяем, является ли это API или Ollama (до загрузки)
        is_api_model = hyperparameters.get("api_model", False)
        is_ollama = hyperparameters.get("ollama", False)
        is_vllm = hyperparameters.get("vllm", False)

        # Мультиагентный режим не поддерживается для Ollama и vLLM
        if (is_ollama or is_vllm) and use_multi_agent:
            use_multi_agent = False
            multi_agent_mode = None
            print("   Ollama/vLLM: мультиагентный режим не поддерживается, используется одноагентный.\n")

        # Информация о GPU/API до загрузки модели
        if is_api_model:
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ:")
            print(f"   • Тип: API (Google Generative AI)")
            print(f"   • Модель будет использоваться через API")
            print()
            gpu_info_before = {"api": True}
        elif is_ollama:
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ:")
            print(f"   • Тип: Ollama (локальный API)")
            print()
            gpu_info_before = {"ollama": True}
        elif is_vllm:
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ:")
            print(f"   • Тип: vLLM (HTTP API, см. VLLM_BASE_URL)")
            print()
            gpu_info_before = {"vllm": True}
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
        elif is_ollama:
            from gpu_info import get_gpu_memory_usage_nvidia_smi, get_ollama_inference_vram_gb
            _mem = get_gpu_memory_usage_nvidia_smi()
            _ollama_gb = get_ollama_inference_vram_gb()
            memory_after_load = {
                "allocated": _ollama_gb,
                "reserved": 0.0,
                "total": _mem["total_gb"],
            }
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ (Ollama):")
            print(f"   • Тип: Ollama (локальный API)")
            print(
                f"   • GPU: процессы Ollama (compute-apps) {_ollama_gb:.2f} GB "
                f"| карта всего занято {_mem['used_gb']:.2f} / {_mem['total_gb']:.2f} GB"
            )
            print()
            gpu_info_after = {"ollama": True}
        elif is_vllm:
            from gpu_info import get_gpu_memory_usage_nvidia_smi
            _mem = get_gpu_memory_usage_nvidia_smi()
            memory_after_load = {
                "allocated": _mem["used_gb"],
                "reserved": 0.0,
                "total": _mem["total_gb"],
            }
            print(f"📊 ИНФОРМАЦИЯ О РЕСУРСАХ (vLLM):")
            print(f"   • Тип: vLLM (сервер должен держать модель на GPU)")
            print(
                f"   • GPU (nvidia-smi): занято {_mem['used_gb']:.2f} / {_mem['total_gb']:.2f} GB"
            )
            print()
            gpu_info_after = {"vllm": True}
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
        memory_samples = []  # Локально — torch.cuda.memory_allocated; Ollama — VRAM только процессов Ollama (compute-apps)
        total_start_time = time.time()
        
        # Переводим в eval режим только локальные модели (не API, не Ollama, не vLLM)
        if not is_api_model and not is_ollama and not is_vllm and hasattr(model, 'eval'):
            model.eval()
        
        print(f"🔄 ОБРАБОТКА ДАТАСЕТА")
        print(f"{'='*80}")
        print(f"Всего текстов: {len(self.texts)}")
        print(f"{'='*80}\n")
        
        # Создаем обертку для генератора для мультиагентного подхода (те же гиперпараметры, что в стандартном режиме)
        if use_multi_agent:
            if is_api_model:
                from core.generators import APIGenerator
                model_name = hyperparameters.get("model_name", "gemma-3-12b-it")
                generator = APIGenerator(model, tokenizer, model_name=model_name)
            else:
                generator = _MultiAgentGenerator(model, tokenizer, generate_func, hyperparameters)
        
        interrupted = False
        last_processed_index = -1
        timeout_reason = None
        max_inference_time_seconds = MAX_INFERENCE_TIME_MINUTES * 60
        last_outlines_skip = False

        # Чтобы предупреждения (в т.ч. от mistral-common про tokenize=False) писались в model_errors.log с именем модели
        global _prev_showwarning, _eval_warning_context
        _eval_warning_context["model_name"] = model_name
        _eval_warning_context["output_dir"] = self.output_dir
        _prev_showwarning = warnings.showwarning
        warnings.showwarning = _eval_showwarning

        try:
            for i, text in enumerate(self.texts):
                # Проверяем среднее время инференса (сумма/количество) перед обработкой нового текста
                avg_inference_time = sum(times) / len(times) if times else 0
                if avg_inference_time > max_inference_time_seconds:
                    interrupted = True
                    last_processed_index = i - 1
                    timeout_reason = f"Превышен лимит времени ({MAX_INFERENCE_TIME_MINUTES} минут)"
                    avg_minutes = avg_inference_time / 60
                    print(f"\n   ⚠️ Прерывание инференса: превышен лимит времени ({MAX_INFERENCE_TIME_MINUTES} минут)")
                    print(f"   ⏱️ Среднее время инференса: {avg_minutes:.1f} мин/ответ")
                    print(f"   📊 Обработано текстов: {i}/{len(self.texts)}")
                    break
                
                # Выводим номер обрабатываемого ответа
                if not verbose:
                    print(f"\r  🔄 Обработка ответа #{i+1}/{len(self.texts)}...", end="", flush=True)
                
                response_text = ""
                error_msg = None
                
                if use_multi_agent:
                    # Мультиагентный подход
                    try:
                        start_time = time.time()
                        hp = {**hyperparameters, "model_key": model_key or "", "prompt_template_name": hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME}
                        result = process_with_multi_agent(
                            text=text,
                            generator=generator,
                            max_new_tokens=max_new_tokens,
                            multi_agent_mode=multi_agent_mode,
                            hyperparameters=hp,
                        )
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        if verbose:
                            only_agent_1 = (multi_agent_mode == "validation_fix_2agents" and result.get("is_valid") and not result.get("validation_errors"))
                            msg = "Обработка текста" if only_agent_1 else "Мультиагентная обработка текста"
                            print(f"   🔄 Ответ #{i+1}/{len(self.texts)} - {msg}:")
                        # Измеряем память во время инференса (только для локальных моделей с весами)
                        if not is_api_model and not is_ollama and not is_vllm:
                            memory_sample = get_gpu_memory_usage()
                            memory_samples.append(memory_sample["allocated"])
                        
                        response_text = result.get("response", "")
                        if is_ollama and response_text:
                            from gpu_info import get_ollama_inference_vram_gb
                            memory_samples.append(get_ollama_inference_vram_gb())
                        elif is_vllm and response_text:
                            from gpu_info import get_gpu_memory_usage_nvidia_smi
                            memory_samples.append(get_gpu_memory_usage_nvidia_smi()["used_gb"])
                        initial_response_text = result.get("initial_response", "")
                        json_part = result.get("json", "")
                        parsed_json = result.get("json_parsed", {})
                        # Та же постобработка, что и для одноагентного режима: латиница -> кириллица, очистка
                        if parsed_json and isinstance(parsed_json, dict) and any(k in LATIN_TO_CYRILLIC_KEYS for k in parsed_json):
                            parsed_json = latin_to_cyrillic_output(parsed_json)
                            try:
                                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                        elif not parsed_json and json_part and any(f'"{k}"' in json_part for k in LATIN_TO_CYRILLIC_KEYS):
                            json_part = latin_keys_to_cyrillic_in_json_str(json_part)
                            parsed_json = parse_json_safe(extract_json_from_response(json_part))
                        if parsed_json and isinstance(parsed_json, dict):
                            parsed_json = self._clean_parsed_json(parsed_json)
                            try:
                                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
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
                        
                        # Для мультиагентного режима нужно добавить raw_output, raw_validation и parsed_validation
                        # response_text содержит сырой ответ модели
                        raw_output_for_result = response_text
                        # Валидация raw output через Pydantic
                        raw_validation_for_result = validate_with_pydantic(raw_output_for_result, stage="raw")
                        # Валидация после парсинга через Pydantic
                        parsed_validation_for_result = validate_with_pydantic(parsed_json, stage="parsed")
                        
                        results.append({
                            "text": text,
                            "json": json_part,
                            "json_parsed": parsed_json,
                            "is_valid": is_valid,
                            "raw_output": raw_output_for_result,
                            "raw_validation": raw_validation_for_result,
                            "parsed_validation": parsed_validation_for_result,
                            "prompt": result.get("prompt", ""),
                            "fix_prompt": result.get("fix_prompt", ""),
                            "fix_called": result.get("fix_called", False),
                            "initial_response": result.get("initial_response", ""),
                            "validation_errors": result.get("validation_errors", ""),
                        })
                    except Exception as e:
                        error_msg = str(e)
                        import traceback
                        traceback_str = traceback.format_exc()
                        # Для API моделей сохраняем полный traceback
                        traceback_display = traceback_str if (is_api_model or is_ollama or is_vllm) else traceback_str[:200]
                        parsing_errors.append({
                            "text_index": i,
                            "text": text,
                            "error": f"Критическая ошибка в мультиагентном подходе: {error_msg}. Traceback: {traceback_display}",
                            "response": ""
                        })
                        empty_validation = {"is_valid": False, "errors": [], "validated_data": None}
                        results.append({
                            "text": text,
                            "json": "",
                            "json_parsed": {},
                            "is_valid": False,
                            "raw_output": "",
                            "raw_validation": empty_validation,
                            "parsed_validation": empty_validation,
                            "prompt": "",
                            "initial_response": "",
                            "validation_errors": "",
                        })
                else:
                    # Одноагентный подход (оригинальный)
                    so = hyperparameters.get("structured_output", False)
                    uo = hyperparameters.get("use_outlines", False)
                    use_guidance = hyperparameters.get("use_guidance", False)
                    pt_name = hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME
                    rs = None
                    if use_guidance:
                        from structured_schemas import FertilizerExtractionOutput
                        rs = FertilizerExtractionOutput
                    elif uo or so:
                        from structured_schemas import FertilizerExtractionOutput
                        rs = FertilizerExtractionOutput
                    # При --structured-output / --guidance передаём Pydantic-схему в промпт
                    prompt = prompt_template(text, structured_output=so or use_guidance, response_schema=rs, prompt_template_name=pt_name, model_key=model_key)
                    
                    # Генерируем ответ с повторными попытками
                    response_text, elapsed, error_msg, outlines_skip = self._generate_response_with_retries(
                        model, tokenizer, prompt, generate_func,
                        hyperparameters, max_new_tokens, num_retries,
                        is_api_model or is_ollama or is_vllm, verbose, i, len(self.texts), text,
                        times, memory_samples, parsing_errors, model_name=model_name
                    )
                    if outlines_skip:
                        if last_outlines_skip:
                            raise InferenceCriticalFailure(
                                f"Outlines vocabulary/encoding: второй подряд пример #{i+1} упал. {error_msg}",
                                i, num_retries
                            )
                        last_outlines_skip = True
                        result = self._handle_no_response(
                            text, i, len(self.texts), error_msg,
                            is_api_model or is_ollama or is_vllm, verbose, parsing_errors
                        )
                        result["prompt"] = prompt
                        results.append(result)
                        self._print_progress(i, len(self.texts), results, times, total_start_time, verbose)
                        last_processed_index = i
                        continue
                    last_outlines_skip = False
                    # Ollama: замер GPU через nvidia-smi и сбор метрик из ответа API
                    if is_ollama and response_text:
                        from gpu_info import get_ollama_inference_vram_gb
                        memory_samples.append(get_ollama_inference_vram_gb())
                    elif is_vllm and response_text:
                        from gpu_info import get_gpu_memory_usage_nvidia_smi
                        memory_samples.append(get_gpu_memory_usage_nvidia_smi()["used_gb"])
                    # Выводим исходный текст и полный ответ в консоль (только при verbose)
                    # Для validation_fix_2agents иногда финальный response может быть пустым,
                    # но initial_response будет заполнен.
                    if verbose and (response_text or initial_response_text):
                        response_for_print = response_text or initial_response_text
                        self._print_verbose_output(text, response_for_print, is_api_model or is_ollama or is_vllm, i, len(self.texts))
                    
                    if not response_text:
                        result = self._handle_no_response(
                            text, i, len(self.texts), error_msg,
                            is_api_model or is_ollama or is_vllm, verbose, parsing_errors
                        )
                        result["prompt"] = prompt
                        results.append(result)
                        continue
                    
                    # Обрабатываем ответ: валидация, парсинг JSON
                    result = self._process_response(
                        response_text, text, i, is_api_model or is_ollama or is_vllm, verbose, parsing_errors
                    )
                    result["prompt"] = prompt
                    results.append(result)
            
            # Выводим прогресс после каждого запроса
            self._print_progress(i, len(self.texts), results, times, total_start_time, verbose)
            last_processed_index = i
        
        except InferenceCriticalFailure as e:
            print(f"\n\n{'='*80}")
            print(f"ДОСРОЧНОЕ ЗАВЕРШЕНИЕ ОЦЕНКИ МОДЕЛИ")
            print(f"{'='*80}")
            print(f"Не удалось получить ответ после {e.num_retries} попыток на примере #{e.text_index + 1}.")
            print(f"Ошибка: {e.message}")
            print(f"{'='*80}\n")
            self.clear_memory()
            error_text = f"Досрочное завершение: не удалось получить ответ после {e.num_retries} попыток на примере #{e.text_index + 1}. {e.message}"
            _append_to_model_errors_log(
                self.output_dir,
                title="КРИТИЧЕСКАЯ ОШИБКА ГЕНЕРАЦИИ",
                model_name=model_name,
                message=error_text,
            )
            parsing_errors.sort(key=lambda e: e.get("text_index", 999999) if isinstance(e, dict) else 999999)
            return {
                "status": "error",
                "error": error_text,
                "parsing_errors": parsing_errors,
            }
        except KeyboardInterrupt:
            interrupted = True
            last_processed_index = i if 'i' in locals() else -1
            print(f"\n\n{'='*80}")
            print(f"⚠️  ПРЕРЫВАНИЕ ОБРАБОТКИ ПОЛЬЗОВАТЕЛЕМ")
            print(f"{'='*80}")
            print(f"Обработано текстов: {len(results)}/{len(self.texts)}")
            print(f"Последний обработанный индекс: {last_processed_index + 1}")
            print()
            
            if stop_all_on_interrupt:
                menu_lines = [
                    "Выберите действие:",
                    "  1 - Сохранить текущую модель и завершить её (к следующей модели)",
                    "  2 - Продолжить обработку текущей модели",
                    "  3 - Не сохранять текущую модель (предыдущие уже сохранены), к следующей модели",
                    "  4 - Прервать оценку всех моделей (выйти из run_all_models)",
                ]
            else:
                menu_lines = [
                    "Выберите действие:",
                    "  1 - Сохранить текущий прогресс и завершить",
                    "  2 - Продолжить обработку",
                    "  3 - Завершить без сохранения (метрики в консоль)",
                ]
            menu_prompt = "\n".join(menu_lines) + "\nВаш выбор (1/2/3" + ("/4" if stop_all_on_interrupt else "") + "): "
            while True:
                try:
                    choice = input(menu_prompt).strip()
                    
                    if choice == "1":
                        print("\n💾 Сохранение текущего прогресса и завершение...")
                        interrupted = False  # чтобы ниже выполнилось _save_results
                        break
                    elif choice == "2":
                        print("\n▶️  Продолжаем обработку...\n")
                        interrupted = False
                        try:
                            for i in range(last_processed_index + 1, len(self.texts)):
                                # Выводим номер обрабатываемого ответа
                                if not verbose:
                                    print(f"\r  🔄 Обработка ответа #{i+1}/{len(self.texts)}...", end="", flush=True)
                                
                                response_text = ""
                                error_msg = None
                                
                                if use_multi_agent:
                                    try:
                                        start_time = time.time()
                                        hp = {**hyperparameters, "model_key": model_key or "", "prompt_template_name": hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME}
                                        result = process_with_multi_agent(
                                            text=self.texts[i],
                                            generator=generator,
                                            max_new_tokens=max_new_tokens,
                                            multi_agent_mode=multi_agent_mode,
                                            hyperparameters=hp,
                                        )
                                        elapsed = time.time() - start_time
                                        times.append(elapsed)
                                        if verbose:
                                            only_agent_1 = (multi_agent_mode == "validation_fix_2agents" and result.get("is_valid") and not result.get("validation_errors"))
                                            msg = "Обработка текста" if only_agent_1 else "Мультиагентная обработка текста"
                                            print(f"   🔄 Ответ #{i+1}/{len(self.texts)} - {msg}:")
                                        if not is_api_model and not is_ollama and not is_vllm:
                                            memory_sample = get_gpu_memory_usage()
                                            memory_samples.append(memory_sample["allocated"])
                                        
                                        response_text = result.get("response", "")
                                        if is_ollama and response_text:
                                            from gpu_info import get_ollama_inference_vram_gb
                                            memory_samples.append(get_ollama_inference_vram_gb())
                                        elif is_vllm and response_text:
                                            from gpu_info import get_gpu_memory_usage_nvidia_smi
                                            memory_samples.append(get_gpu_memory_usage_nvidia_smi()["used_gb"])
                                        initial_response_text = result.get("initial_response", "")
                                        json_part = result.get("json", "")
                                        parsed_json = result.get("json_parsed", {})
                                        # Та же постобработка, что и для одноагентного режима: латиница -> кириллица, очистка
                                        if parsed_json and isinstance(parsed_json, dict) and any(k in LATIN_TO_CYRILLIC_KEYS for k in parsed_json):
                                            parsed_json = latin_to_cyrillic_output(parsed_json)
                                            try:
                                                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                                            except Exception:
                                                pass
                                        elif not parsed_json and json_part and any(f'"{k}"' in json_part for k in LATIN_TO_CYRILLIC_KEYS):
                                            json_part = latin_keys_to_cyrillic_in_json_str(json_part)
                                            parsed_json = parse_json_safe(extract_json_from_response(json_part))
                                        if parsed_json and isinstance(parsed_json, dict):
                                            parsed_json = self._clean_parsed_json(parsed_json)
                                            try:
                                                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                                            except Exception:
                                                pass
                                        is_valid = result.get("is_valid", False)
                                        error_msg = result.get("error")
                                        
                                        if error_msg:
                                            parsing_errors.append(f"Текст #{i}: ошибка в мультиагентном подходе. Ошибка: {error_msg}")
                                        
                                        if not is_valid and json_part:
                                            # Для API моделей при verbose выводим полный JSON, иначе обрезаем
                                            json_display = json_part if (is_api_model and verbose) else (json_part[:200] if len(json_part) > 200 else json_part)
                                            parsing_errors.append(f"Текст #{i}: невалидный JSON. Ответ: {json_display}")
                                        
                                        # Для мультиагентного режима нужно добавить raw_output, raw_validation и parsed_validation
                                        # response_text содержит сырой ответ модели
                                        raw_output_for_result = response_text
                                        # Валидация raw output через Pydantic
                                        raw_validation_for_result = validate_with_pydantic(raw_output_for_result, stage="raw")
                                        # Валидация после парсинга через Pydantic
                                        parsed_validation_for_result = validate_with_pydantic(parsed_json, stage="parsed")
                                        
                                        results.append({
                                            "text": self.texts[i],
                                            "json": json_part,
                                            "json_parsed": parsed_json,
                                            "is_valid": is_valid,
                                            "raw_output": raw_output_for_result,
                                            "raw_validation": raw_validation_for_result,
                                            "parsed_validation": parsed_validation_for_result,
                                            "prompt": result.get("prompt", ""),
                                            "fix_prompt": result.get("fix_prompt", ""),
                                            "fix_called": result.get("fix_called", False),
                                            "initial_response": result.get("initial_response", ""),
                                            "validation_errors": result.get("validation_errors", ""),
                                        })
                                    except Exception as e:
                                        error_msg = str(e)
                                        import traceback
                                        parsing_errors.append(f"Текст #{i}: критическая ошибка в мультиагентном подходе. Ошибка: {error_msg}. Traceback: {traceback.format_exc()[:200]}")
                                        empty_validation = {"is_valid": False, "errors": [], "validated_data": None}
                                        results.append({
                                            "text": self.texts[i],
                                            "json": "",
                                            "json_parsed": {},
                                            "is_valid": False,
                                            "raw_output": "",
                                            "raw_validation": empty_validation,
                                            "parsed_validation": empty_validation,
                                            "prompt": "",
                                            "initial_response": "",
                                            "validation_errors": "",
                                        })
                                else:
                                    so = hyperparameters.get("structured_output", False)
                                    uo = hyperparameters.get("use_outlines", False)
                                    use_guidance = hyperparameters.get("use_guidance", False)
                                    rs = None
                                    if use_guidance:
                                        from structured_schemas import FertilizerExtractionOutput
                                        rs = FertilizerExtractionOutput
                                    elif uo or so:
                                        from structured_schemas import FertilizerExtractionOutput
                                        rs = FertilizerExtractionOutput
                                    pt_name = hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME
                                    prompt = prompt_template(self.texts[i], structured_output=so or use_guidance, response_schema=rs, prompt_template_name=pt_name, model_key=model_key)
                                    
                                    # Генерируем ответ с повторными попытками
                                    response_text, elapsed, error_msg, outlines_skip = self._generate_response_with_retries(
                                        model, tokenizer, prompt, generate_func,
                                        hyperparameters, max_new_tokens, num_retries,
                                        is_api_model or is_ollama or is_vllm, verbose, i, len(self.texts), self.texts[i],
                                        times, memory_samples, parsing_errors, model_name=model_name
                                    )
                                    if outlines_skip:
                                        if last_outlines_skip:
                                            mode_label = "Guidance" if use_guidance else "Outlines"
                                            raise InferenceCriticalFailure(
                                                f"{mode_label} vocabulary/encoding: второй подряд пример #{i+1} упал. {error_msg}",
                                                i, num_retries
                                            )
                                        last_outlines_skip = True
                                        result = self._handle_no_response(
                                            self.texts[i], i, len(self.texts), error_msg,
                                            is_api_model or is_ollama or is_vllm, verbose, parsing_errors
                                        )
                                        result["prompt"] = prompt
                                        results.append(result)
                                        self._print_progress(i, len(self.texts), results, times, total_start_time, verbose)
                                        continue
                                    last_outlines_skip = False
                                    if is_ollama and response_text:
                                        from gpu_info import get_ollama_inference_vram_gb
                                        memory_samples.append(get_ollama_inference_vram_gb())
                                    elif is_vllm and response_text:
                                        from gpu_info import get_gpu_memory_usage_nvidia_smi
                                        memory_samples.append(get_gpu_memory_usage_nvidia_smi()["used_gb"])
                                    # Выводим исходный текст и полный ответ в консоль (только при verbose)
                                    if verbose and (response_text or initial_response_text):
                                        response_for_print = response_text or initial_response_text
                                        self._print_verbose_output(self.texts[i], response_for_print, is_api_model or is_ollama or is_vllm, i, len(self.texts))
                                    
                                    if not response_text:
                                        result = self._handle_no_response(
                                            self.texts[i], i, len(self.texts), error_msg,
                                            is_api_model or is_ollama or is_vllm, verbose, parsing_errors
                                        )
                                        result["prompt"] = prompt
                                        results.append(result)
                                        continue
                                    
                                    # Обрабатываем ответ: валидация, парсинг JSON
                                    result = self._process_response(
                                        response_text, self.texts[i], i, is_api_model or is_ollama or is_vllm, verbose, parsing_errors
                                    )
                                    result["prompt"] = prompt
                                    results.append(result)
                                
                                # Выводим прогресс
                                self._print_progress(i, len(self.texts), results, times, total_start_time, verbose)
                        except InferenceCriticalFailure as e:
                            print(f"\n\n{'='*80}")
                            print(f"ДОСРОЧНОЕ ЗАВЕРШЕНИЕ ОЦЕНКИ МОДЕЛИ")
                            print(f"{'='*80}")
                            print(f"Не удалось получить ответ после {e.num_retries} попыток на примере #{e.text_index + 1}.")
                            print(f"Ошибка: {e.message}")
                            print(f"{'='*80}\n")
                            self.clear_memory()
                            error_text = f"Досрочное завершение: не удалось получить ответ после {e.num_retries} попыток на примере #{e.text_index + 1}. {e.message}"
                            _append_to_model_errors_log(
                                self.output_dir,
                                title="КРИТИЧЕСКАЯ ОШИБКА ГЕНЕРАЦИИ",
                                model_name=model_name,
                                message=error_text,
                            )
                            parsing_errors.sort(key=lambda e: e.get("text_index", 999999) if isinstance(e, dict) else 999999)
                            return {
                                "status": "error",
                                "error": error_text,
                                "parsing_errors": parsing_errors,
                            }
                        except KeyboardInterrupt:
                            print(f"\n\n⚠️  Повторное прерывание. Завершение без сохранения...")
                            interrupted = True
                            break
                        break
                    elif choice == "3":
                        if stop_all_on_interrupt:
                            print("\n❌ Текущая модель не сохраняется (предыдущие уже сохранены). Переход к следующей модели...")
                            return {
                                "status": "interrupted",
                                "message": "Обработка прервана пользователем без сохранения текущей модели",
                                "processed_count": len(results)
                            }
                        else:
                            print("\n📊 Завершение без сохранения (метрики будут выведены в консоль)...")
                            interrupted = True
                            break
                    elif stop_all_on_interrupt and choice == "4":
                        print("\n❌ Прерывание оценки всех моделей...")
                        raise StopAllModelsInterrupt()
                    else:
                        print("Пожалуйста, введите 1, 2" + (", 3, 4" if stop_all_on_interrupt else " или 3"))
                except StopAllModelsInterrupt:
                    raise
                except KeyboardInterrupt:
                    print("\n\n⚠️  Повторное прерывание. Завершение без сохранения...")
                    interrupted = True
                    break
        finally:
            if _prev_showwarning is not None:
                warnings.showwarning = _prev_showwarning
                _prev_showwarning = None
            _eval_warning_context["model_name"] = None
            _eval_warning_context["output_dir"] = None

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
        elif memory_samples:
            # Для локальных моделей — torch allocated; для Ollama — nvidia-smi used_gb
            memory_during_inference_avg = sum(memory_samples) / len(memory_samples)
        else:
            # Fallback: измеряем сейчас, если не было измерений
            current_memory = get_gpu_memory_usage()
            memory_during_inference_avg = current_memory["allocated"]
        
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
            pt_name = hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME
            full_prompt_example = prompt_template(example_text, prompt_template_name=pt_name, model_key=model_key)
        
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
        elif is_ollama:
            print(f"💾 ИСПОЛЬЗОВАНИЕ РЕСУРСОВ (Ollama, VRAM процессов Ollama / compute-apps):")
            print(f"   • После загрузки: {memory_after_load['allocated']:.2f} GB")
            print(f"   • Во время инференса (средн.): {memory_during_inference_avg:.2f} GB")
            print()
        elif is_vllm:
            print(f"💾 ИСПОЛЬЗОВАНИЕ РЕСУРСОВ (vLLM, nvidia-smi):")
            print(f"   • После «загрузки» (замер GPU): {memory_after_load['allocated']:.2f} GB")
            print(f"   • Во время инференса (средн.): {memory_during_inference_avg:.2f} GB")
            print()
        else:
            print(f"💾 ИСПОЛЬЗОВАНИЕ ПАМЯТИ:")
            print(f"   • После загрузки модели: {memory_after_load['allocated']:.2f} GB")
            print(f"   • Во время инференса: {memory_during_inference_avg:.2f} GB")
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
        raw_output_metrics = None
        validation_stats = None
        pydantic_errors = []

        if self.ground_truths and len(self.ground_truths) == len(results):
            try:
                print(f"🎯 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА...")
                # Фильтруем и нормализуем predictions: должны быть словарями
                predictions = []
                raw_outputs = []
                texts_for_metrics = []
                responses_for_metrics = []
                
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
                
                # Извлекаем тексты, ответы и raw outputs из results
                for r in results:
                    texts_for_metrics.append(r.get("text", ""))
                    responses_for_metrics.append(r.get("json", ""))  # json содержит ответ модели
                    # Для raw output используем только реальный raw_output, без fallback на json
                    # так как json уже прошел через умный парсер
                    raw_output = r.get("raw_output", "")
                    if not raw_output:
                        # Если raw_output отсутствует (например, в мультиагентном режиме),
                        # используем response, который должен содержать сырой ответ
                        raw_output = r.get("response", "")
                    if not raw_output:
                        # Если и response отсутствует, используем пустую строку
                        raw_output = ""
                    raw_outputs.append(raw_output)
                
                quality_metrics = calculate_quality_metrics(
                    predictions, ground_truths_normalized,
                    texts=texts_for_metrics,
                    responses=responses_for_metrics,
                    prompt=full_prompt_example
                )
                
                # Собираем статистику валидации (Pydantic), отдельно от валидности JSON.
                # raw_output: валидируем raw_output (модель могла вернуть текст вокруг JSON)
                # parsed: валидируем распарсенный JSON; если is_valid=False (JSON не распарсился) — это тоже invalid.
                raw_validations = [r.get("raw_validation", {}) for r in results]
                # Для parsed учитываем все ответы, но валидация вычисляется только для успешно распарсенных
                parsed_validations = [r.get("parsed_validation", {}) for r in results if r.get("is_valid", False)]

                # Фильтруем пустые валидации (если они не были вычислены)
                raw_validations = [v for v in raw_validations if v]
                parsed_validations = [v for v in parsed_validations if v]

                # Подсчитываем количество успешно распарсенных ответов (is_valid = True)
                total_parsed_count = sum(1 for r in results if r.get("is_valid", False))

                # Конкретный промпт по примеру: один или сконкатенированный при реальном мультиагентном использовании
                concrete_prompts = []
                for r in results:
                    p1 = r.get("prompt", "")
                    p2 = r.get("fix_prompt", "")
                    fix_called = r.get("fix_called", False)
                    should_concat = (
                        use_multi_agent
                        and multi_agent_mode == "validation_fix_2agents"
                        and fix_called
                    )
                    # Если agent2 реально вызвался, в metrics нужен сконкатенированный промпт,
                    # даже если его response окажется пустым.
                    if should_concat and (not p2):
                        try:
                            text_for_fix = r.get("text", "")
                            validation_errors = r.get("validation_errors", "")
                            original_response = r.get("initial_response", "")
                            invalid_json = extract_json_from_response(original_response) or original_response

                            # original_prompt уже содержит {text} в конце.
                            # Agent2 (validation_fix) НЕ должен видеть текст дважды, поэтому
                            # обрезаем original_prompt по маркеру "Текст для извлечения данных:".
                            original_prompt_for_fix = p1 or ""
                            text_marker = "Текст для извлечения данных:"
                            if text_marker in original_prompt_for_fix:
                                original_prompt_for_fix = original_prompt_for_fix.split(text_marker, 1)[0].rstrip()

                            p2 = prompt_config.VALIDATION_FIX_PROMPT.format(
                                original_prompt=original_prompt_for_fix,
                                text=text_for_fix,
                                invalid_json=invalid_json,
                                validation_errors=validation_errors,
                            )
                        except Exception:
                            p2 = ""

                    if should_concat and p2:
                        concrete_prompts.append(
                            "МУЛЬТИАГЕНТНЫЙ РЕЖИМ (использован для этого примера)\n\n" + p1 + "\n\n---\n\n" + p2
                        )
                    else:
                        concrete_prompts.append(p1 or full_prompt_example)
                # Список ошибок Pydantic по текстам
                for idx, r in enumerate(results):
                    rv = r.get("raw_validation") or {}
                    pv = r.get("parsed_validation") or {}
                    rv_ok = rv.get("is_valid", True)
                    pv_ok = pv.get("is_valid", True)
                    if rv_ok and pv_ok:
                        continue
                    errors = []
                    if not rv_ok:
                        errors.extend(rv.get("errors", []))
                    if not pv_ok:
                        errors.extend(pv.get("errors", []))
                    prompt_for_entry = concrete_prompts[idx] if idx < len(concrete_prompts) else (r.get("prompt") or full_prompt_example)
                    entry = {
                        "text_index": idx,
                        "text": r.get("text", ""),
                        "response": r.get("json", "") or r.get("raw_output", "") or r.get("response", ""),
                        "prompt": prompt_for_entry,
                        "errors": errors,
                    }
                    if r.get("initial_response") is not None:
                        entry["initial_response"] = r.get("initial_response", "")
                    if r.get("validation_errors"):
                        entry["validation_errors"] = r.get("validation_errors", "")
                    pydantic_errors.append(entry)

                if raw_validations or parsed_validations:
                    raw_valid_count = sum(1 for v in raw_validations if v.get("is_valid", False)) if raw_validations else 0
                    parsed_valid_count = sum(1 for v in parsed_validations if v.get("is_valid", False)) if parsed_validations else 0

                    # Для parsed: invalid_count = невалидные среди распарсенных + не распарсенные ответы
                    parsed_invalid_count = (len(parsed_validations) - parsed_valid_count) if parsed_validations else 0
                    # Добавляем количество не распарсенных ответов
                    not_parsed_count = len(results) - total_parsed_count
                    parsed_invalid_count += not_parsed_count

                    validation_stats = {
                        "raw_output": {
                            "invalid_count": len(raw_validations) - raw_valid_count if raw_validations else 0,
                            "validation_rate": raw_valid_count / len(raw_validations) if raw_validations else 0.0
                        },
                        "parsed": {
                            "invalid_count": parsed_invalid_count,
                            "validation_rate": parsed_valid_count / len(results) if len(results) > 0 else 0.0
                        }
                    }
                else:
                    validation_stats = None
                
                # Проверяем, что quality_metrics - это словарь
                if not isinstance(quality_metrics, dict):
                    print(f"   ⚠️ Ошибка: calculate_quality_metrics вернула не словарь, а {type(quality_metrics)}")
                    quality_metrics = None
                else:
                    # Подставляем конкретный промпт по примеру в ошибки quality_metrics (единое поле prompt)
                    for group in ("массовая доля", "прочее"):
                        if group in quality_metrics and "все_ошибки" in quality_metrics[group]:
                            for err in quality_metrics[group]["все_ошибки"]:
                                ti = err.get("text_index")
                                if ti is not None and ti < len(concrete_prompts):
                                    err["prompt"] = concrete_prompts[ti]
                    # Выводим метрики качества через MetricsPrinter (cleaned output)
                    MetricsPrinter.print_quality_metrics(quality_metrics)
                
                # Выводим статистику валидации cleaned output, если она была вычислена
                if validation_stats:
                    MetricsPrinter.print_validation_stats(validation_stats)
                else:
                    print(f"\n   ⚠️ Статистика валидации не была вычислена (raw_validation или parsed_validation отсутствуют в results)")
                
                # Вычисляем метрики для raw output (без допущений, кроме регистра)
                print(f"\n🎯 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА ДЛЯ RAW OUTPUT...")
                print(f"   • Количество raw_outputs: {len(raw_outputs)}")
                print(f"   • Количество непустых raw_outputs: {sum(1 for ro in raw_outputs if ro)}")
                print(f"   • Количество пустых raw_outputs: {sum(1 for ro in raw_outputs if not ro)}")
                if sum(1 for ro in raw_outputs if not ro) > 0:
                    print(f"   ⚠️ ВНИМАНИЕ: {sum(1 for ro in raw_outputs if not ro)} raw_outputs пустые! Это может быть проблемой в мультиагентном режиме.")
                try:
                    raw_output_metrics = calculate_raw_output_metrics(
                        raw_outputs, ground_truths_normalized,
                        texts=texts_for_metrics,
                        responses=raw_outputs,  # Используем raw_outputs как responses
                        prompt=full_prompt_example
                    )
                    print(f"   ✅ Метрики raw output вычислены")
                    if raw_output_metrics:
                        print(f"   • Raw метрики содержат: {list(raw_output_metrics.keys())}")
                        # Выводим raw метрики через MetricsPrinter
                        MetricsPrinter.print_raw_output_metrics(raw_output_metrics)
                    else:
                        print(f"   ⚠️ Raw метрики пустые")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при вычислении метрик raw output: {e}")
                    import traceback
                    traceback.print_exc()
                    raw_output_metrics = None
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

        # Ошибки в JSON по возрастанию text_index
        def _text_index_sort_key(e):
            return e.get("text_index", 999999) if isinstance(e, dict) else 999999
        parsing_errors.sort(key=_text_index_sort_key)

        # Формируем итоговый результат с правильным порядком полей
        # 1. Сначала метрики парсинга и валидации
        # 2. Затем quality_metrics
        # 3. Затем raw_output_metrics
        # 4. Потом все остальное
        evaluation_result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
            "model_name": model_name,
            "model_key": model_key,  # Alias модели из конфигурации (например, "gemma-2-2b")
            "interrupted": interrupted,
            "timeout_reason": timeout_reason,  # Причина прерывания (если было прерывание по времени)
            "total_samples": len(results),  # Количество обработанных результатов (может быть меньше общего количества текстов при прерывании)
            # Метрики парсинга и валидации (первыми)
            "valid_json_count": valid_count,  # Количество валидных JSON (можно вычислить: total_samples - invalid_json_count)
            "invalid_json_count": invalid_count,  # Количество невалидных JSON
            "parsing_error_rate": parsing_error_rate,  # Процент ошибок парсинга (можно вычислить: invalid_json_count / total_samples)
            "parsing_errors_count": len(parsing_errors),  # Количество записей об ошибках (может отличаться от invalid_json_count)
            "validation_stats": validation_stats,  # Статистика валидации через Pydantic
            "pydantic_errors": pydantic_errors,  # Список ошибок валидации Pydantic по текстам
            # Метрики качества (вторыми)
            "quality_metrics": quality_metrics,
            # Метрики raw output (третьими)
            "raw_output_metrics": raw_output_metrics,  # Метрики для raw output
            # Остальные поля
            "multi_agent_mode": multi_agent_mode if use_multi_agent else None,
            "gpu_info": gpu_info_before if not (is_api_model or is_ollama or is_vllm) else (
                {"api": True} if is_api_model else ({"ollama": True} if is_ollama else {"vllm": True})
            ),
            "gpu_memory_after_load_gb": memory_after_load["allocated"] if not is_api_model else 0.0,
            "gpu_memory_during_inference_gb": memory_during_inference_avg if not is_api_model else 0.0,
            "api_model": is_api_model,
            "ollama": is_ollama,
            "vllm": is_vllm,
            "average_response_time_seconds": avg_speed,
            "hyperparameters": hyperparameters_to_save,
            "effective_prompt_template_name": hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME,
            "prompt_template": (hyperparameters.get("prompt_template_name") or PROMPT_TEMPLATE_NAME) if not use_multi_agent else multi_agent_mode,
            "prompt_full_text": full_prompt_example,
            "prompt_info": prompt_info,
            "parsing_errors": parsing_errors,
            "gemini_analysis": gemini_analysis
        }
        
        # При прерывании не сохраняем результаты, только пишем в model_errors.log
        if interrupted:
            log_file = os.path.join(self.output_dir, "model_errors.log")
            error_msg = (
                f"\n{'='*80}\n"
                f"ПРЕРЫВАНИЕ ОБРАБОТКИ МОДЕЛИ\n"
                f"{'='*80}\n"
                f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Модель: {model_name}\n"
                f"Причина: {timeout_reason or 'Прервано пользователем'}\n"
                f"Обработано текстов: {len(results)}/{len(self.texts)}\n"
                f"{'='*80}\n"
            )
            logging.error(error_msg)
            print(f"\nРезультаты не сохранены (прерывание). Запись в {log_file}")
        else:
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
        """Сохраняет результаты в файлы с организованной структурой папок"""
        timestamp = evaluation_result["timestamp"]
        
        print(f"\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
        
        # Используем высокоуровневый метод FileManager для сохранения всех результатов
        saved_files = self.file_manager.save_evaluation_results(
            evaluation_result=evaluation_result,
            results=results,
            output_dir=self.output_dir,
            timestamp=timestamp
        )
        
        print(f"✅ Все результаты сохранены успешно!")
    
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
        
        # Извлекаем тексты и ответы из CSV для передачи в метрики
        texts_for_metrics = []
        responses_for_metrics = []
        if "text" in df_results.columns:
            texts_for_metrics = df_results["text"].tolist()
        if "json" in df_results.columns:
            responses_for_metrics = df_results["json"].tolist()
        
        # Пересчитываем метрики качества
        try:
            # Передаем тексты и ответы из CSV, если они доступны
            quality_metrics = calculate_quality_metrics(
                predictions, 
                ground_truths, 
                texts=texts_for_metrics if texts_for_metrics else None, 
                responses=responses_for_metrics if responses_for_metrics else None
            )
            
            # Выводим детальные метрики качества через MetricsPrinter (cleaned output)
            if quality_metrics:
                MetricsPrinter.print_quality_metrics(quality_metrics)
        except Exception as e:
            print(f"⚠️  Ошибка при вычислении метрик качества: {e}")
            import traceback
            traceback.print_exc()
            quality_metrics = None
        
        # Вычисляем статистику валидации (если есть колонки raw_validation и parsed_validation)
        validation_stats = None
        if "raw_validation" in df_results.columns or "parsed_validation" in df_results.columns:
            print(f"\n📊 ВЫЧИСЛЕНИЕ СТАТИСТИКИ ВАЛИДАЦИИ...")
            try:
                # Загружаем данные валидации из CSV
                raw_validations = []
                parsed_validations = []
                
                if "raw_validation" in df_results.columns:
                    for idx, row in df_results.iterrows():
                        raw_val = row.get("raw_validation", "")
                        # Пытаемся распарсить, если это строка
                        if isinstance(raw_val, str) and raw_val:
                            # Убираем пробелы и проверяем, не пустая ли строка
                            raw_val = raw_val.strip()
                            if raw_val and raw_val != "nan" and raw_val != "None":
                                try:
                                    raw_val = json.loads(raw_val)
                                except (json.JSONDecodeError, ValueError):
                                    # Если не JSON, пытаемся распарсить как Python dict literal
                                    try:
                                        raw_val = eval(raw_val) if raw_val else {}
                                    except:
                                        raw_val = {}
                            else:
                                raw_val = {}
                        elif not isinstance(raw_val, dict):
                            raw_val = {}
                        raw_validations.append(raw_val)
                else:
                    raw_validations = [{}] * len(df_results)
                
                if "parsed_validation" in df_results.columns:
                    for idx, row in df_results.iterrows():
                        # Для parsed учитываем только успешно распарсенные ответы (is_valid = True)
                        is_valid = row.get("is_valid", False)
                        if not is_valid:
                            continue  # Пропускаем невалидные ответы
                        
                        parsed_val = row.get("parsed_validation", "")
                        # Пытаемся распарсить, если это строка
                        if isinstance(parsed_val, str) and parsed_val:
                            # Убираем пробелы и проверяем, не пустая ли строка
                            parsed_val = parsed_val.strip()
                            if parsed_val and parsed_val != "nan" and parsed_val != "None":
                                try:
                                    parsed_val = json.loads(parsed_val)
                                except (json.JSONDecodeError, ValueError):
                                    # Если не JSON, пытаемся распарсить как Python dict literal
                                    try:
                                        parsed_val = eval(parsed_val) if parsed_val else {}
                                    except:
                                        parsed_val = {}
                            else:
                                parsed_val = {}
                        elif not isinstance(parsed_val, dict):
                            parsed_val = {}
                        parsed_validations.append(parsed_val)
                else:
                    # Если колонки parsed_validation нет, но есть is_valid, учитываем только валидные
                    if "is_valid" in df_results.columns:
                        parsed_validations = [{}] * sum(1 for _, row in df_results.iterrows() if row.get("is_valid", False))
                    else:
                        parsed_validations = []
                
                # Подсчитываем статистику (Pydantic), отдельно от валидности JSON.
                raw_total_count = sum(1 for v in raw_validations if v)
                raw_valid_count = sum(1 for v in raw_validations if v and v.get("is_valid", False))
                raw_invalid_indices = [i for i, v in enumerate(raw_validations) if v and not v.get("is_valid", False)]

                total_results_count = len(df_results)
                parsed_total_count = total_results_count
                parsed_valid_count = 0
                parsed_invalid_indices = []
                parsed_not_parsed_indices = []

                for idx, row in df_results.iterrows():
                    if not row.get("is_valid", False):
                        parsed_invalid_indices.append(int(idx))
                        parsed_not_parsed_indices.append(int(idx))
                        continue
                    pv = row.get("parsed_validation", {}) or {}
                    if isinstance(pv, str) and pv:
                        # На всякий случай: если не распарсили выше
                        try:
                            import json as _json
                            pv = _json.loads(pv)
                        except Exception:
                            pv = {}
                    if isinstance(pv, dict) and pv.get("is_valid", False):
                        parsed_valid_count += 1
                    else:
                        parsed_invalid_indices.append(int(idx))

                validation_stats = {
                    "raw_output": {
                        "total_count": raw_total_count,
                        "valid_count": raw_valid_count,
                        "invalid_count": raw_total_count - raw_valid_count,
                        "validation_rate": (raw_valid_count / raw_total_count) if raw_total_count > 0 else 0.0,
                        "invalid_indices": raw_invalid_indices,
                    },
                    "parsed": {
                        "total_count": parsed_total_count,
                        "valid_count": parsed_valid_count,
                        "invalid_count": len(parsed_invalid_indices),
                        "validation_rate": (parsed_valid_count / parsed_total_count) if parsed_total_count > 0 else 0.0,
                        "invalid_indices": parsed_invalid_indices,
                        "not_parsed_indices": parsed_not_parsed_indices,
                    },
                }
                
                print(f"   ✅ Статистика валидации вычислена")
                # Выводим статистику валидации cleaned output через MetricsPrinter
                MetricsPrinter.print_validation_stats(validation_stats)
            except Exception as e:
                print(f"   ⚠️ Ошибка при вычислении статистики валидации: {e}")
                import traceback
                traceback.print_exc()
                validation_stats = None
        else:
            print(f"\n⚠️  Колонки 'raw_validation' и 'parsed_validation' не найдены в CSV, пропускаем вычисление статистики валидации")
        
        # Вычисляем метрики для raw output (если есть колонка raw_output)
        raw_output_metrics = None
        
        if "raw_output" in df_results.columns:
            print(f"\n🎯 ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА ДЛЯ RAW OUTPUT...")
            try:
                # Заменяем NaN и None на пустые строки, затем конвертируем в список
                raw_outputs = df_results["raw_output"].fillna("").astype(str).tolist()
                # Убираем строки "nan" и "None"
                raw_outputs = [ro if ro not in ["nan", "None", ""] else "" for ro in raw_outputs]
                print(f"   • Количество raw_outputs: {len(raw_outputs)}")
                print(f"   • Количество непустых raw_outputs: {sum(1 for ro in raw_outputs if ro)}")
                
                # Нормализуем ground_truths для raw метрик
                ground_truths_normalized = []
                for gt in ground_truths:
                    if isinstance(gt, list):
                        ground_truths_normalized.append({})
                    elif isinstance(gt, dict):
                        ground_truths_normalized.append(gt)
                    else:
                        ground_truths_normalized.append({})
                
                # Обрезаем до минимальной длины
                min_len = min(len(raw_outputs), len(ground_truths_normalized))
                raw_outputs = raw_outputs[:min_len]
                ground_truths_normalized = ground_truths_normalized[:min_len]
                
                raw_output_metrics = calculate_raw_output_metrics(
                    raw_outputs, ground_truths_normalized,
                    texts=texts_for_metrics[:min_len] if texts_for_metrics else None,
                    responses=raw_outputs  # Используем raw_outputs как responses
                )
                print(f"   ✅ Метрики raw output вычислены")
                if raw_output_metrics:
                    print(f"   • Raw метрики содержат: {list(raw_output_metrics.keys())}")
                    # Выводим raw метрики через MetricsPrinter
                    MetricsPrinter.print_raw_output_metrics(raw_output_metrics)
                else:
                    print(f"   ⚠️ Raw метрики пустые")
            except Exception as e:
                print(f"   ⚠️ Ошибка при вычислении метрик raw output: {e}")
                import traceback
                traceback.print_exc()
                raw_output_metrics = None
        else:
            print(f"\n⚠️  Колонка 'raw_output' не найдена в CSV, пропускаем вычисление raw метрик")
        
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
            file_manager = FileManager()
            filename = file_manager.get_basename(results_csv_path)
            # Формат: results_model_name_timestamp.csv или results_timestamp.csv (новая структура)
            # Убираем префикс "results_" и расширение ".csv"
            name_without_ext = filename.replace("results_", "").replace(".csv", "")
            # Убираем timestamp в конце (формат: _HHMMSS или _YYYYMMDD_HHMMSS)
            import re
            # Убираем паттерны типа _123456 или _20260123_123456
            name_without_timestamp = re.sub(r'_\d{4}$|_\d{8}_\d{4}$', '', name_without_ext)
            if name_without_timestamp:
                model_name = name_without_timestamp
            else:
                # Если после удаления timestamp ничего не осталось, используем исходное имя
                parts = name_without_ext.split("_")
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
                    metrics_file_pattern = f"metrics_{FileManager.sanitize_filename(model_name)}_*.json"
                    file_manager = FileManager()
                    metrics_files = file_manager.find_files(metrics_file_pattern, file_manager.get_dirname(results_csv_path))
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        evaluation_result = {
            "timestamp": timestamp,
            "model_name": model_name,
            "reevaluated_from": results_csv_path,
            "parsing_error_rate": parsing_error_rate,
            "parsing_errors_count": len(parsing_errors),
            "quality_metrics": quality_metrics,
            "raw_output_metrics": raw_output_metrics,  # Метрики для raw output
            "validation_stats": validation_stats,  # Статистика валидации через Pydantic
            "parsing_errors": parsing_errors,
            "total_samples": len(predictions),
            "valid_json_count": valid_count,
            "invalid_json_count": invalid_count,
            "gemini_analysis": gemini_analysis
        }
        
        # Сохраняем обновленные метрики
        print(f"\n💾 СОХРАНЕНИЕ ОБНОВЛЕННЫХ РЕЗУЛЬТАТОВ...")
        
        # Используем высокоуровневый метод FileManager для сохранения всех результатов переоценки
        file_manager = FileManager()
        saved_files = file_manager.save_reevaluation_results(
            evaluation_result=evaluation_result,
            results_csv_path=results_csv_path,
            df_results=df_results,
            predictions=predictions,
            quality_metrics=quality_metrics,
            raw_output_metrics=raw_output_metrics,
            timestamp=timestamp,
            model_name=model_name
        )
        
        print(f"✅ Все результаты переоценки сохранены успешно!")
        
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

