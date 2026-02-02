"""
Главный файл для запуска оценки моделей
"""
import os
import sys
import logging
from datetime import datetime
from model_evaluator import ModelEvaluator
from gemini_analyzer import analyze_errors_with_gemini, check_gemini_api
from config import GROUND_TRUTH_PATH, OUTPUT_DIR, GEMINI_API_KEY, MODEL_CONFIGS
from utils import find_dataset_path

# Настройка логирования
log_file = os.path.join(OUTPUT_DIR, "model_errors.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)


def run_evaluation(model_config: dict, model_key: str = None, use_gemini: bool = True, verbose: bool = False):
    """
    Запускает оценку модели
    
    Args:
        model_config: словарь с конфигурацией модели:
            - name: название модели
            - load_func: функция загрузки модели
            - generate_func: функция генерации
            - hyperparameters: гиперпараметры (может содержать multi_agent_mode)
        model_key: ключ модели из конфигурации (alias, используется для имен файлов)
        use_gemini: использовать ли анализ через Gemini API
        verbose: если True, выводит подробную информацию (текст и ответы) в консоль
    """
    evaluator = ModelEvaluator(
        dataset_path=find_dataset_path(),
        ground_truth_path=GROUND_TRUTH_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Очищаем память перед загрузкой новой модели
    evaluator.clear_memory()
    
    # Запускаем оценку
    # Для API моделей используем больше попыток (10 вместо 2)
    num_retries = 10 if model_config["hyperparameters"].get("api_model", False) else 2
    result = evaluator.evaluate_model(
        model_name=model_config["name"],
        load_model_func=model_config["load_func"],
        generate_func=model_config["generate_func"],
        hyperparameters=model_config["hyperparameters"],
        num_retries=num_retries,
        verbose=verbose,  # Передаем флаг verbose
        use_gemini_analysis=use_gemini,
        gemini_api_key=GEMINI_API_KEY if use_gemini else None,
        model_key=model_key  # Передаем alias модели для структуры папок
    )
    
    if result.get("status") == "error":
        print(f"Ошибка при оценке модели: {result.get('error')}")
        
        # Записываем ошибку в log файл
        error_msg = f"\n{'='*80}\n"
        error_msg += f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ\n"
        error_msg += f"{'='*80}\n"
        error_msg += f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        error_msg += f"Модель: {model_config['name']}\n"
        error_msg += f"Ошибка: {result.get('error')}\n"
        error_msg += f"\nГиперпараметры: {model_config.get('hyperparameters', {})}\n"
        error_msg += f"\nПолный traceback:\n{result.get('error_traceback', 'Не указан')}\n"
        error_msg += f"{'='*80}\n"
        
        logging.error(error_msg)
        print(f"Ошибка записана в log файл: {log_file}")
        return result
    
    # Анализ через Gemini (если включен)
    if use_gemini:
        print(f"\n{'='*80}")
        print(f"АНАЛИЗ ОШИБОК ЧЕРЕЗ GEMINI API")
        print(f"{'='*80}")
        
        parsing_errors = result.get("parsing_errors", [])
        quality_metrics = result.get("quality_metrics", {})
        hyperparameters = result.get("hyperparameters", {})
        
        print(f"Статистика для анализа:")
        print(f"   • Ошибок парсинга: {len(parsing_errors)}")
        if quality_metrics:
            # Используем 'все_ошибки' для правильного подсчета всех ошибок
            mass_dolya = quality_metrics.get('массовая доля', {})
            mass_errors = len(mass_dolya.get('все_ошибки', mass_dolya.get('ошибки', [])))
            prochee = quality_metrics.get('прочее', {})
            prochee_errors = len(prochee.get('все_ошибки', prochee.get('ошибки', [])))
            print(f"   • Ошибок качества 'массовая доля': {mass_errors}")
            print(f"   • Ошибок качества 'прочее': {prochee_errors}")
        print(f"   • Гиперпараметры: {len(hyperparameters)} параметров")
        print()
        
        # Анализ через Gemini теперь выполняется внутри evaluate_model
        # Сохраняем анализ в JSON, если он был выполнен
        gemini_analysis = result.get("gemini_analysis")
        if gemini_analysis and gemini_analysis.get("status") == "success":
            timestamp = result.get("timestamp", datetime.now().strftime("%H%M%S"))
            from model_evaluator import sanitize_filename
            
            # Определяем структуру папок (такая же, как в _save_results)
            model_key = result.get("model_key")
            if not model_key:
                model_key = sanitize_filename(model_config["name"])
            else:
                model_key = sanitize_filename(model_key)
            
            multi_agent_mode = result.get("multi_agent_mode")
            prompt_template_name = result.get("prompt_template", "unknown")
            
            if multi_agent_mode:
                prompt_folder_name = sanitize_filename(multi_agent_mode)
            else:
                prompt_folder_name = sanitize_filename(prompt_template_name)
            
            # Создаем структуру папок
            model_dir = os.path.join(OUTPUT_DIR, model_key)
            prompt_dir = os.path.join(model_dir, prompt_folder_name)
            os.makedirs(prompt_dir, exist_ok=True)
            
            analysis_path = os.path.join(prompt_dir, f"gemini_analysis_{timestamp}.json")
            
            analysis_text = gemini_analysis.get("analysis", "")
            
            # Получаем дополнительную информацию из результата оценки
            hyperparameters = result.get("hyperparameters", {})
            gpu_info = result.get("gpu_info", {})
            average_response_time = result.get("average_response_time_seconds", 0)
            gpu_memory_during_inference = result.get("gpu_memory_during_inference_gb", 0)
            api_model = result.get("api_model", False)
            multi_agent_mode = result.get("multi_agent_mode")
            prompt_full_text = result.get("prompt_full_text")
            prompt_info = result.get("prompt_info")
            prompt_template = result.get("prompt_template")
            
            analysis_data = {
                "model_name": model_config["name"],
                "timestamp": timestamp,
                "analysis": analysis_text,
                "model_used": gemini_analysis.get("model_used", "gemini-2.5-flash"),
                "parsing_errors_count": len(parsing_errors),
                "hyperparameters": hyperparameters,
                "prompts": {
                    "prompt_template": prompt_template,
                    "prompt_full_text": prompt_full_text,
                    "prompt_info": prompt_info
                },
                "system_info": {
                    "api_model": api_model,
                    "multi_agent_mode": multi_agent_mode,
                    "gpu_info": gpu_info,
                    "gpu_memory_during_inference_gb": gpu_memory_during_inference,
                    "average_response_time_seconds": average_response_time
                },
                "quality_metrics_summary": {
                    "массовая доля": {
                        "accuracy": quality_metrics.get('массовая доля', {}).get('accuracy', 0) if quality_metrics else 0,
                        "precision": quality_metrics.get('массовая доля', {}).get('precision', 0) if quality_metrics else 0,
                        "recall": quality_metrics.get('массовая доля', {}).get('recall', 0) if quality_metrics else 0,
                        "f1": quality_metrics.get('массовая доля', {}).get('f1', 0) if quality_metrics else 0
                    },
                    "прочее": {
                        "accuracy": quality_metrics.get('прочее', {}).get('accuracy', 0) if quality_metrics else 0,
                        "precision": quality_metrics.get('прочее', {}).get('precision', 0) if quality_metrics else 0,
                        "recall": quality_metrics.get('прочее', {}).get('recall', 0) if quality_metrics else 0,
                        "f1": quality_metrics.get('прочее', {}).get('f1', 0) if quality_metrics else 0
                    }
                } if quality_metrics else None
            }
            
            import json
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Анализ Gemini сохранен в JSON: {analysis_path}\n")
    else:
        if not use_gemini:
            print(f"\nАнализ через Gemini API пропущен (отключен пользователем)\n")
    
    return result


def main():
    """Главная функция"""
    # Проверяем работоспособность Gemini API в самом начале
    print(f"\n{'='*80}")
    print(f"ПРОВЕРКА СИСТЕМЫ")
    print(f"{'='*80}")
    # GEMINI_API_KEY загружается из config.py (который берет его из config_secrets.py или переменных окружения)
    
    if GEMINI_API_KEY:
        print(f"Проверка работоспособности Gemini API...")
        gemini_working, gemini_message = check_gemini_api(GEMINI_API_KEY)
        print(f"   {gemini_message}\n")
    else:
        print(f"GEMINI_API_KEY не установлен, пропускаем проверку API")
        gemini_working = False
        print()
    
    use_gemini = True
    if not gemini_working:
        print(f"{'='*80}")
        print(f"ВНИМАНИЕ: Gemini API недоступен")
        print(f"{'='*80}")
        print(f"Оценка модели будет выполнена, но анализ ошибок через Gemini будет пропущен.")
        print(f"Вы можете продолжить без анализатора ошибок или исправить проблему и запустить заново.\n")
        
        while True:
            response = input("Продолжить без анализатора ошибок? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'да', 'д']:
                use_gemini = False
                print("Продолжаем без анализатора ошибок...\n")
                break
            elif response in ['n', 'no', 'нет', 'н']:
                print("Запуск отменён. Исправьте проблему с Gemini API и попробуйте снова.")
                return
            else:
                print("Пожалуйста, введите 'y' (да) или 'n' (нет)")
    
    # Теперь проверяем аргументы командной строки
    if len(sys.argv) < 2:
        print("Использование: python main.py <model_name> [model_name2 ...] [--multi-agent MODE] [--structured-output] [--outlines] [--no-gemini] [--verbose] [--no-verbose]")
        print("\nАргументы:")
        print("  <model_name>        - ключ модели из конфигурации (можно указать несколько через запятую или пробел)")
        print("  --multi-agent       - (опционально) режим мультиагентного подхода")
        print("                        Доступные режимы: simple_4agents, critic_3agents, qa_workflow")
        print("  --structured-output - (опционально) использовать structured output через Pydantic")
        print("                        Работает только с API моделями, поддерживающими structured output")
        print("  --outlines          - (опционально) использовать библиотеку outlines для структурированной генерации JSON")
        print("                        Работает для локальных моделей вместе с --structured-output и Pydantic схемой")
        print("  --no-gemini         - (опционально) отключить анализ ошибок через Gemini API")
        print("  --verbose           - (опционально) включить подробный вывод (включен по умолчанию)")
        print("  --no-verbose        - (опционально) отключить подробный вывод")
        print("\nПримеры:")
        print("  python main.py qwen-2.5-3b")
        print("  python main.py qwen-2.5-3b,qwen-2.5-4b  # Несколько моделей через запятую")
        print("  python main.py qwen-2.5-3b qwen-2.5-4b  # Несколько моделей через пробел")
        print("  python main.py qwen-2.5-3b --multi-agent simple_4agents")
        print("  python main.py qwen-2.5-3b qwen-2.5-4b --no-gemini  # Несколько моделей с флагами")
        print("  python main.py gemma-3-27b-api --structured-output")
        print("  python main.py qwen-3-32b-api --structured-output")
        print("  python main.py qwen-2.5-3b --no-gemini")
        print("  python main.py qwen-2.5-3b --structured-output --outlines")
        print("\nДоступные модели:")
        for key in MODEL_CONFIGS.keys():
            print(f"  - {key}")
        return
    
    # Парсим аргументы командной строки
    # Сначала собираем все модели и флаги
    model_keys = []
    multi_agent_mode = None
    structured_output = False
    use_outlines = False
    use_gemini = True  # По умолчанию включен
    verbose = True  # По умолчанию включен для main.py
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Если это флаг, обрабатываем его
        if arg.startswith("--"):
            if arg == "--multi-agent":
                if i + 1 < len(sys.argv):
                    multi_agent_mode = sys.argv[i + 1]
                    i += 2
                else:
                    print("Ошибка: после --multi-agent должен быть указан режим (например, simple_4agents)")
                    return
            elif arg == "--structured-output":
                structured_output = True
                i += 1
            elif arg == "--outlines":
                use_outlines = True
                i += 1
            elif arg == "--no-gemini" or arg == "--skip-gemini":
                use_gemini = False
                i += 1
            elif arg == "--verbose":
                verbose = True
                i += 1
            elif arg == "--no-verbose" or arg == "--quiet":
                verbose = False
                i += 1
            else:
                print(f"Неизвестный аргумент: {arg}")
                print("Использование: python main.py <model_name> [model_name2 ...] [--multi-agent MODE] [--structured-output] [--outlines] [--no-gemini] [--verbose] [--no-verbose]")
                return
        else:
            # Это модель или список моделей через запятую
            if "," in arg:
                # Разбиваем по запятой
                models = [m.strip() for m in arg.split(",") if m.strip()]
                model_keys.extend(models)
            else:
                model_keys.append(arg)
            i += 1
    
    # Проверяем, что указаны модели
    if not model_keys:
        print("Ошибка: не указаны модели для оценки")
        print("Использование: python main.py <model_name> [model_name2 ...] [--multi-agent MODE] [--structured-output] [--outlines] [--no-gemini] [--verbose] [--no-verbose]")
        return
    
    # Проверяем, что все модели существуют
    invalid_models = [m for m in model_keys if m not in MODEL_CONFIGS]
    if invalid_models:
        print(f"Ошибка: следующие модели не найдены: {', '.join(invalid_models)}")
        print("Доступные модели:", ", ".join(MODEL_CONFIGS.keys()))
        return
    
    # Проверяем существование датасета
    dataset_path = find_dataset_path()
    if not os.path.exists(dataset_path):
        print(f"Датасет не найден: {dataset_path}")
        print("Убедитесь, что файл results_var3.xlsx находится в папке data/")
        return
    
    # Если одна модель, выводим информацию о ней
    if len(model_keys) == 1:
        model_key = model_keys[0]
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ МОДЕЛИ")
        print(f"{'='*80}")
        print(f"📌 Модель: {model_key}")
        print(f"📌 Полное название: {MODEL_CONFIGS[model_key]['name']}")
    else:
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ НЕСКОЛЬКИХ МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"📌 Количество моделей: {len(model_keys)}")
        print(f"📌 Модели: {', '.join(model_keys)}")
    
    if multi_agent_mode:
        print(f"📌 Режим: Мультиагентный ({multi_agent_mode})")
    else:
        print(f"📌 Режим: Одноагентный")
    if structured_output:
        print(f"📌 Structured Output: Включен (Pydantic валидация)")
    if use_outlines:
        print(f"📌 Outlines: Включен")
    if verbose:
        print(f"📌 Verbose: Включен (подробный вывод)")
    else:
        print(f"📌 Verbose: Отключен (краткий вывод)")
    if use_gemini:
        print(f"📌 Анализ через Gemini API: Включен")
    else:
        print(f"📌 Анализ через Gemini API: Отключен")
    print(f"📁 Датасет: {find_dataset_path()}")
    print(f"📁 Результаты: {OUTPUT_DIR}")
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Запускаем оценку для каждой модели
    import copy
    results_summary = []
    
    for idx, model_key in enumerate(model_keys, 1):
        if len(model_keys) > 1:
            print(f"\n{'='*80}")
            print(f"МОДЕЛЬ {idx}/{len(model_keys)}: {model_key}")
            print(f"{'='*80}\n")
        
        # Создаем копию конфигурации и добавляем параметры если указаны
        config = copy.deepcopy(MODEL_CONFIGS[model_key])
        if multi_agent_mode:
            config["hyperparameters"]["multi_agent_mode"] = multi_agent_mode
        if structured_output:
            config["hyperparameters"]["structured_output"] = True
        if use_outlines:
            config["hyperparameters"]["use_outlines"] = True
        
        try:
            result = run_evaluation(config, model_key=model_key, use_gemini=use_gemini, verbose=verbose)
            
            if result.get("status") != "error":
                results_summary.append({
                    "model_key": model_key,
                    "status": "success",
                    "result": result
                })
                
                if len(model_keys) == 1:
                    # Для одной модели выводим полную сводку
                    print(f"\n{'='*80}")
                    print(f"🎉 ФИНАЛЬНАЯ СВОДКА")
                    print(f"{'='*80}")
                    print(f"Оценка модели '{model_key}' завершена успешно!")
                    print(f"\nОсновные результаты:")
                    print(f"   • Модель: {result.get('model_name', 'N/A')}")
                    print(f"   • Время выполнения: {result.get('average_response_time_seconds', 0) * result.get('total_samples', 0) / 60:.2f} минут")
                    print(f"   • Средняя скорость: {result.get('average_response_time_seconds', 0):.3f} сек/ответ")
                    print(f"   • Ошибки парсинга: {result.get('parsing_error_rate', 0):.2%} ({result.get('invalid_json_count', 0)}/{result.get('total_samples', 0)})")
                    print(f"   • Использование памяти: {result.get('gpu_memory_during_inference_gb', 0):.2f} GB")
                    
                    quality = result.get('quality_metrics')
                    if quality:
                        print(f"\n🎯 Метрики качества:")
                        mass = quality.get('массовая доля', {})
                        prochee = quality.get('прочее', {})
                        print(f"   • 'массовая доля':")
                        print(f"     - Accuracy: {mass.get('accuracy', 0):.2%}")
                        print(f"     - Precision: {mass.get('precision', 0):.2%}, Recall: {mass.get('recall', 0):.2%}, F1: {mass.get('f1', 0):.2%}")
                        print(f"   • 'прочее':")
                        print(f"     - Accuracy: {prochee.get('accuracy', 0):.2%}")
                        print(f"     - Precision: {prochee.get('precision', 0):.2%}, Recall: {prochee.get('recall', 0):.2%}, F1: {prochee.get('f1', 0):.2%}")
                    
                    print(f"\n📁 Результаты сохранены в директории: {OUTPUT_DIR}")
                    print(f"{'='*80}\n")
            else:
                results_summary.append({
                    "model_key": model_key,
                    "status": "error",
                    "error": result.get("error", "Unknown error")
                })
                print(f"❌ Ошибка при оценке модели '{model_key}': {result.get('error', 'Unknown error')}\n")
        except KeyboardInterrupt:
            print(f"\n⚠️ Прервано пользователем. Остановка оценки моделей.")
            break
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"❌ Критическая ошибка при оценке '{model_key}': {error_msg}")
            print(f"   Детали: {traceback.format_exc()[:500]}...\n")
            results_summary.append({
                "model_key": model_key,
                "status": "error",
                "error": error_msg
            })
    
    # Выводим итоговую сводку для нескольких моделей
    if len(model_keys) > 1:
        print(f"\n{'='*80}")
        print(f"🎉 ИТОГОВАЯ СВОДКА ПО ВСЕМ МОДЕЛЯМ")
        print(f"{'='*80}\n")
        
        successful = [s for s in results_summary if s['status'] == 'success']
        failed = [s for s in results_summary if s['status'] == 'error']
        
        print(f"Общая статистика:")
        print(f"   • Всего моделей: {len(results_summary)}")
        print(f"   • Успешно оценено: {len(successful)}")
        print(f"   • Пропущено из-за ошибок: {len(failed)}")
        print()
        
        if successful:
            print(f"УСПЕШНО ОЦЕНЕННЫЕ МОДЕЛИ:")
            for summary in successful:
                result = summary['result']
                print(f"   • {summary['model_key']}")
                print(f"     - Модель: {result.get('model_name', 'N/A')}")
                print(f"     - Скорость: {result.get('average_response_time_seconds', 0):.3f} сек/ответ")
                print(f"     - Ошибки парсинга: {result.get('parsing_error_rate', 0):.2%}")
                print(f"     - Память: {result.get('gpu_memory_during_inference_gb', 0):.2f} GB")
                
                quality = result.get('quality_metrics')
                if quality:
                    mass = quality.get('массовая доля', {})
                    prochee = quality.get('прочее', {})
                    print(f"     - Метрики 'массовая доля': Accuracy={mass.get('accuracy', 0):.2%}, F1={mass.get('f1', 0):.2%}")
                    print(f"     - Метрики 'прочее': Accuracy={prochee.get('accuracy', 0):.2%}, F1={prochee.get('f1', 0):.2%}")
                print()
        
        if failed:
            print(f"ПРОПУЩЕННЫЕ МОДЕЛИ:")
            for summary in failed:
                print(f"   • {summary['model_key']}: {summary.get('error', 'Unknown error')[:100]}")
            print()
        
        print(f"📁 Результаты сохранены в директории: {OUTPUT_DIR}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

