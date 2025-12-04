"""
Основной класс для оценки моделей LLM
"""
import torch
import gc
import time
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import os

from utils import build_prompt3, parse_json_safe, is_valid_json, extract_json_from_response
from metrics import calculate_quality_metrics
from gpu_info import get_gpu_info, get_gpu_memory_usage


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
                      num_retries: int = 2) -> Dict[str, Any]:
        """
        Оценивает модель на датасете
        
        Args:
            model_name: название модели
            load_model_func: функция для загрузки модели (должна возвращать (model, tokenizer))
            generate_func: функция генерации (model, tokenizer, prompt) -> response_text
            hyperparameters: словарь с гиперпараметрами
            prompt_template: шаблон промпта (если None, используется build_prompt3)
            max_new_tokens: максимальное количество новых токенов
            num_retries: количество попыток при ошибке
        
        Returns:
            словарь с результатами оценки
        """
        print(f"\n{'='*80}")
        print(f"🚀 НАЧАЛО ОЦЕНКИ МОДЕЛИ")
        print(f"{'='*80}")
        print(f"📌 Модель: {model_name}")
        print(f"📌 Датасет: {len(self.texts)} текстов")
        print(f"📌 Гиперпараметры:")
        for key, value in hyperparameters.items():
            print(f"   • {key}: {value}")
        print(f"{'='*80}\n")
        
        # Информация о GPU до загрузки модели
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
        
        # Информация о GPU после загрузки
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
        parsing_errors = []
        times = []
        memory_samples = []  # Для сбора измерений памяти во время инференса
        total_start_time = time.time()
        
        model.eval()
        
        print(f"🔄 ОБРАБОТКА ДАТАСЕТА")
        print(f"{'='*80}")
        print(f"Всего текстов: {len(self.texts)}")
        print(f"{'='*80}\n")
        
        for i, text in enumerate(self.texts):
            prompt = prompt_template(text)
            response_text = ""
            error_msg = None
            
            # Попытки генерации
            for attempt in range(num_retries):
                try:
                    start_time = time.time()
                    # Передаем repetition_penalty из гиперпараметров, если есть
                    repetition_penalty = hyperparameters.get("repetition_penalty")
                    if repetition_penalty is not None:
                        response_text = generate_func(model, tokenizer, prompt, max_new_tokens, repetition_penalty=repetition_penalty)
                    else:
                        response_text = generate_func(model, tokenizer, prompt, max_new_tokens)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    
                    # Измеряем память во время инференса (после каждого запроса)
                    memory_sample = get_gpu_memory_usage()
                    memory_samples.append(memory_sample["allocated"])
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ⚠️ [{i+1}/{len(self.texts)}] Ошибка при генерации (попытка {attempt+1}/{num_retries}): {error_msg[:100]}")
                    if attempt < num_retries - 1:
                        time.sleep(4 + attempt * 2)
                    else:
                        # Если все попытки исчерпаны, сохраняем детальную информацию об ошибке
                        import traceback
                        parsing_errors.append(f"Текст #{i}: критическая ошибка генерации после {num_retries} попыток. Ошибка: {error_msg}. Traceback: {traceback.format_exc()[:200]}")
            
            if not response_text:
                print(f"  ❌ [{i+1}/{len(self.texts)}] Ответ не получен — пропуск")
                if error_msg:
                    print(f"     Последняя ошибка: {error_msg[:200]}")
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
                parsing_errors.append(f"Текст #{i}: невалидный JSON. Ответ: {json_part}")
            
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
            
            # Выводим компактный статус после каждого запроса
            status_line = (
                f"  [{i + 1}/{len(self.texts)}] "
                f"✓: {valid_count} ✗: {invalid_count} | "
                f"Скорость: {avg_time:.2f}с/ответ | "
                f"Осталось: ~{eta_str}"
            )
            print(f"\r{status_line}", end="", flush=True)
            
            # Подробный прогресс каждые 10 текстов или в конце
            if (i + 1) % 10 == 0 or (i + 1) == len(self.texts):
                print()  # Новая строка для подробного вывода
                print(f"     📊 Детальная статистика:")
                print(f"        • Прогресс: {progress_pct:.1f}% ({i + 1}/{len(self.texts)})")
                print(f"        • Валидных JSON: {valid_count} | Невалидных: {invalid_count}")
                print(f"        • Средняя скорость: {avg_time:.3f} сек/ответ")
                print(f"        • Прошло времени: {elapsed_total/60:.1f} мин | Осталось: ~{eta_minutes:.1f} мин")
                print()
        
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
        if memory_samples:
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
        
        # Выводим информацию о промпте
        print(f"📝 ИСПОЛЬЗОВАННЫЙ ПРОМПТ:")
        print(f"   • Шаблон: {prompt_template.__name__ if hasattr(prompt_template, '__name__') else str(prompt_template)}")
        print(f"   • Полный текст промпта (пример с первым текстом):")
        print(f"{'─'*80}")
        example_text = self.texts[0] if self.texts else "Пример текста"
        full_prompt_example = prompt_template(example_text)
        # Выводим промпт с отступами для читаемости
        prompt_lines = full_prompt_example.split('\n')
        for line in prompt_lines[:30]:  # Первые 30 строк, чтобы не перегружать консоль
            print(f"   {line}")
        if len(prompt_lines) > 30:
            print(f"   ... (ещё {len(prompt_lines) - 30} строк, полный текст сохранён в отчёте)")
        print(f"{'─'*80}")
        print()
        
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
                predictions = [r["json_parsed"] for r in results]
                quality_metrics = calculate_quality_metrics(predictions, self.ground_truths)
                
                mass_dolya = quality_metrics.get('массовая доля', {})
                prochee = quality_metrics.get('прочее', {})
                
                print(f"   ✅ Метрики качества вычислены:")
                print(f"   📊 Группа 'массовая доля':")
                print(f"      • Точность (Accuracy): {mass_dolya.get('средняя_точность', 0):.2%}")
                print(f"      • Precision: {mass_dolya.get('precision', 0):.2%}")
                print(f"      • Recall: {mass_dolya.get('recall', 0):.2%}")
                print(f"      • F1-score: {mass_dolya.get('f1', 0):.2%}")
                print(f"      • TP: {mass_dolya.get('tp', 0)}, FP: {mass_dolya.get('fp', 0)}, FN: {mass_dolya.get('fn', 0)}")
                print(f"      • Количество сравнений: {mass_dolya.get('количество_сравнений', 0)}")
                print(f"      • Примеры ошибок: {len(mass_dolya.get('ошибки', []))}")
                print(f"   📊 Группа 'прочее':")
                print(f"      • Точность (Accuracy): {prochee.get('средняя_точность', 0):.2%}")
                print(f"      • Precision: {prochee.get('precision', 0):.2%}")
                print(f"      • Recall: {prochee.get('recall', 0):.2%}")
                print(f"      • F1-score: {prochee.get('f1', 0):.2%}")
                print(f"      • TP: {prochee.get('tp', 0)}, FP: {prochee.get('fp', 0)}, FN: {prochee.get('fn', 0)}")
                print(f"      • Количество сравнений: {prochee.get('количество_сравнений', 0)}")
                print(f"      • Примеры ошибок: {len(prochee.get('ошибки', []))}")
            except Exception as e:
                print(f"   ⚠️ Ошибка при вычислении метрик качества: {e}")
        else:
            print(f"   ⚠️ Ground truth не загружен или не совпадает по размеру с результатами")
            if not self.ground_truths:
                print(f"      (Ground truth не найден в датасете)")
            elif len(self.ground_truths) != len(results):
                print(f"      (Размеры не совпадают: GT={len(self.ground_truths)}, Results={len(results)})")
        print()
        
        # Генерируем пример полного промпта для сохранения в отчёт
        # Используем первый текст из датасета как пример
        example_text = self.texts[0] if self.texts else "Пример текста для анализа"
        full_prompt_example = prompt_template(example_text)
        
        # Формируем итоговый результат
        evaluation_result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_name": model_name,
            "gpu_info": gpu_info_before,
            "gpu_memory_after_load_gb": memory_after_load["allocated"],
            "gpu_memory_during_inference_gb": memory_during_inference_avg,
            "gpu_memory_during_inference_max_gb": memory_during_inference_max,
            "gpu_memory_during_inference_min_gb": memory_during_inference_min,
            "average_response_time_seconds": avg_speed,
            "parsing_error_rate": parsing_error_rate,
            "parsing_errors_count": len(parsing_errors),
            "quality_metrics": quality_metrics,
            "hyperparameters": hyperparameters,
            "prompt_template": prompt_template.__name__ if hasattr(prompt_template, '__name__') else str(prompt_template),
            "prompt_full_text": full_prompt_example,
            "parsing_errors": parsing_errors,
            "total_samples": len(results),
            "valid_json_count": len(results) - invalid_count,
            "invalid_json_count": invalid_count
        }
        
        # Сохраняем результаты
        print(f"💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
        self._save_results(evaluation_result, results)
        
        print(f"\n{'='*80}")
        print(f"✅ ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"{'='*80}")
        print(f"📌 Итоговая сводка:")
        print(f"   • Модель: {model_name}")
        print(f"   • Обработано текстов: {len(results)}")
        print(f"   • Общее время: {total_time/60:.2f} минут")
        print(f"   • Средняя скорость: {avg_speed:.3f} сек/ответ")
        print(f"   • Ошибки парсинга: {parsing_error_rate:.2%} ({invalid_count}/{len(results)})")
        print(f"   • Использование памяти (среднее во время инференса): {memory_during_inference_avg:.2f} GB")
        if quality_metrics:
            mass_acc = quality_metrics.get('массовая доля', {}).get('средняя_точность', 0)
            prochee_acc = quality_metrics.get('прочее', {}).get('средняя_точность', 0)
            print(f"   • Качество 'массовая доля': {mass_acc:.2%}")
            print(f"   • Качество 'прочее': {prochee_acc:.2%}")
        print(f"{'='*80}\n")
        
        return evaluation_result
    
    def _save_results(self, evaluation_result: Dict[str, Any], results: List[Dict[str, Any]]):
        """Сохраняет результаты в файлы"""
        timestamp = evaluation_result["timestamp"]
        model_name_safe = evaluation_result["model_name"].replace("/", "_").replace("\\", "_")
        
        # Сохраняем детальные результаты
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, f"results_{model_name_safe}_{timestamp}.csv")
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 Детальные результаты сохранены: {csv_path}")
        
        # Сохраняем метрики
        metrics_path = os.path.join(self.output_dir, f"metrics_{model_name_safe}_{timestamp}.json")
        import json
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        print(f"💾 Метрики сохранены: {metrics_path}")
        
        # Обновляем общий файл со всеми прогонами
        summary_path = os.path.join(self.output_dir, "evaluation_summary.jsonl")
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation_result, ensure_ascii=False) + '\n')
        print(f"💾 Результат добавлен в общий файл: {summary_path}")
    
    @staticmethod
    def reevaluate_from_file(
        results_csv_path: str,
        dataset_path: str,
        output_dir: str = "results",
        model_name: str = None
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
        predictions = []
        for idx, row in df_results.iterrows():
            pred = row.get("json_parsed", {})
            if isinstance(pred, str):
                try:
                    pred = json.loads(pred)
                except:
                    pred = parse_json_safe(pred)
            predictions.append(pred if isinstance(pred, dict) else {})
        
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
            quality_metrics = calculate_quality_metrics(predictions, ground_truths)
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
        
        # Извлекаем parsing errors из CSV, если есть
        parsing_errors = []
        if "json" in df_results.columns:
            for idx, row in df_results.iterrows():
                json_str = row.get("json", "")
                is_valid = row.get("is_valid", False)
                if not is_valid and json_str:
                    parsing_errors.append(f"Текст #{idx}: невалидный JSON. Ответ: {json_str}")
        
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
            "invalid_json_count": invalid_count
        }
        
        # Сохраняем обновленные метрики
        print(f"\n💾 СОХРАНЕНИЕ ОБНОВЛЕННЫХ РЕЗУЛЬТАТОВ...")
        os.makedirs(output_dir, exist_ok=True)
        
        model_name_safe = model_name.replace("/", "_").replace("\\", "_")
        metrics_path = os.path.join(output_dir, f"metrics_{model_name_safe}_{timestamp}_reevaluated.json")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        print(f"💾 Обновленные метрики сохранены: {metrics_path}")
        
        # Выводим сводку
        print(f"\n{'='*80}")
        print(f"✅ ПЕРЕОЦЕНКА ЗАВЕРШЕНА!")
        print(f"{'='*80}")
        print(f"📌 Итоговая сводка:")
        print(f"   • Модель: {model_name}")
        print(f"   • Обработано текстов: {len(predictions)}")
        print(f"   • Ошибки парсинга: {parsing_error_rate:.2%} ({invalid_count}/{len(predictions)})")
        if quality_metrics:
            mass_acc = quality_metrics.get('массовая доля', {}).get('средняя_точность', 0)
            prochee_acc = quality_metrics.get('прочее', {}).get('средняя_точность', 0)
            mass_f1 = quality_metrics.get('массовая доля', {}).get('f1', 0)
            prochee_f1 = quality_metrics.get('прочее', {}).get('f1', 0)
            print(f"   • Качество 'массовая доля': Accuracy={mass_acc:.2%}, F1={mass_f1:.2%}")
            print(f"   • Качество 'прочее': Accuracy={prochee_acc:.2%}, F1={prochee_f1:.2%}")
        print(f"{'='*80}\n")
        
        return evaluation_result

