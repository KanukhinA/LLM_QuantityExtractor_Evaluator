"""
Главный файл для запуска оценки моделей
"""
import os
import sys
import logging
from datetime import datetime
from model_evaluator import ModelEvaluator
import model_loaders as ml
import model_loaders_api as ml_api
from gemini_analyzer import analyze_errors_with_gemini, check_gemini_api
from config import DATASET_PATH, GROUND_TRUTH_PATH, OUTPUT_DIR, GEMINI_API_KEY

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


def run_evaluation(model_config: dict, use_gemini: bool = True, verbose: bool = False):
    """
    Запускает оценку модели
    
    Args:
        model_config: словарь с конфигурацией модели:
            - name: название модели
            - load_func: функция загрузки модели
            - generate_func: функция генерации
            - hyperparameters: гиперпараметры (может содержать multi_agent_mode)
        use_gemini: использовать ли анализ через Gemini API
        verbose: если True, выводит подробную информацию (текст и ответы) в консоль
    """
    evaluator = ModelEvaluator(
        dataset_path=DATASET_PATH,
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
        gemini_api_key=GEMINI_API_KEY if use_gemini else None
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
            mass_errors = len(quality_metrics.get('массовая доля', {}).get('ошибки', []))
            prochee_errors = len(quality_metrics.get('прочее', {}).get('ошибки', []))
            print(f"   • Ошибок качества 'массовая доля': {mass_errors}")
            print(f"   • Ошибок качества 'прочее': {prochee_errors}")
        print(f"   • Гиперпараметры: {len(hyperparameters)} параметров")
        print()
        
        # Анализ через Gemini теперь выполняется внутри evaluate_model
        # Сохраняем анализ в JSON, если он был выполнен
        gemini_analysis = result.get("gemini_analysis")
        if gemini_analysis and gemini_analysis.get("status") == "success":
            timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name_safe = model_config["name"].replace("/", "_").replace("\\", "_")
            analysis_path = os.path.join(OUTPUT_DIR, f"gemini_analysis_{model_name_safe}_{timestamp}.json")
            
            analysis_text = gemini_analysis.get("analysis", "")
            
            # Получаем дополнительную информацию из результата оценки
            hyperparameters = result.get("hyperparameters", {})
            gpu_info = result.get("gpu_info", {})
            average_response_time = result.get("average_response_time_seconds", 0)
            gpu_memory_during_inference = result.get("gpu_memory_during_inference_gb", 0)
            api_model = result.get("api_model", False)
            multi_agent_mode = result.get("multi_agent_mode")
            
            analysis_data = {
                "model_name": model_config["name"],
                "timestamp": timestamp,
                "analysis": analysis_text,
                "model_used": gemini_analysis.get("model_used", "gemini-2.5-flash"),
                "parsing_errors_count": len(parsing_errors),
                "hyperparameters": hyperparameters,
                "system_info": {
                    "api_model": api_model,
                    "multi_agent_mode": multi_agent_mode,
                    "gpu_info": gpu_info,
                    "gpu_memory_during_inference_gb": gpu_memory_during_inference,
                    "average_response_time_seconds": average_response_time
                },
                "quality_metrics_summary": {
                    "массовая доля": {
                        "accuracy": quality_metrics.get('массовая доля', {}).get('средняя_точность', 0) if quality_metrics else 0,
                        "precision": quality_metrics.get('массовая доля', {}).get('precision', 0) if quality_metrics else 0,
                        "recall": quality_metrics.get('массовая доля', {}).get('recall', 0) if quality_metrics else 0,
                        "f1": quality_metrics.get('массовая доля', {}).get('f1', 0) if quality_metrics else 0
                    },
                    "прочее": {
                        "accuracy": quality_metrics.get('прочее', {}).get('средняя_точность', 0) if quality_metrics else 0,
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


# Конфигурации моделей
MODEL_CONFIGS = {
    "gemma-2-2b": {
        "name": "google/gemma-2-2b-it",
        "load_func": ml.load_gemma_2_2b,
        "generate_func": ml.generate_standard,
        "hyperparameters": {
            "max_new_tokens": 512,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "qwen-2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "load_func": ml.load_qwen_2_5_1_5b,
        "generate_func": ml.generate_qwen,
        "hyperparameters": {
            "max_new_tokens": 512,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "qwen-2.5-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "load_func": ml.load_qwen_2_5_3b,
        "generate_func": ml.generate_qwen,
        "hyperparameters": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "dtype": "bfloat16"
        }
    },
    "qwen-2.5-4b": {
        "name": "Qwen/Qwen2.5-4B-Instruct",
        "load_func": ml.load_qwen_2_5_4b,
        "generate_func": ml.generate_qwen,
        "hyperparameters": {
            "max_new_tokens": 512,
            "do_sample": False,
            "dtype": "bfloat16"
        }
    },
    "gemma-3-1b": {
        "name": "google/gemma-3-1b-it",
        "load_func": ml.load_gemma_3_1b,
        "generate_func": ml.generate_gemma,
        "hyperparameters": {
            "max_new_tokens": 512,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "gemma-3-4b": {
        "name": "google/gemma-3-4b-it",
        "load_func": ml.load_gemma_3_4b,
        "generate_func": ml.generate_gemma,
        "hyperparameters": {
            "max_new_tokens": 512,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "gemma-3-4b-api": {
        "name": "gemma-3-4b-it",
        "load_func": ml_api.load_gemma_3_4b_api,
        "generate_func": ml_api.generate_gemma_api,
        "hyperparameters": {
            "max_new_tokens": 512,
            "model_name": "gemma-3-4b-it",
            "api_model": True
        }
    },
    "gemma-3-12b-api": {
        "name": "gemma-3-12b-it",
        "load_func": ml_api.load_gemma_3_12b_api,
        "generate_func": ml_api.generate_gemma_api,
        "hyperparameters": {
            "max_new_tokens": 512,
            "model_name": "gemma-3-12b-it",
            "api_model": True
        }
    },
    "gemma-3-27b-api": {
        "name": "gemma-3-27b-it",
        "load_func": ml_api.load_gemma_3_27b_api,
        "generate_func": ml_api.generate_gemma_api,
        "hyperparameters": {
            "max_new_tokens": 512,
            "model_name": "gemma-3-27b-it",
            "api_model": True
        }
    },
    "Ministral-3-3B-Reasoning-2512": {
        "name": "mistralai/Ministral-3-3B-Reasoning-2512",
        "load_func": ml.load_ministral_3_3b_reasoning_2512,
        "generate_func": ml.generate_standard,
        "hyperparameters": {
            "max_new_tokens": 512,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "CHEMLLM-2b-1_5": {
        "name": "AI4Chem/CHEMLLM-2b-1_5",
        "load_func": ml.load_chemllm_2b_1_5,
        "generate_func": ml.generate_standard,
        "hyperparameters": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "Phi-3.5-mini-instruct": {
        "name": "microsoft/Phi-3.5-mini-instruct",
        "load_func": ml.load_phi_3_5_mini_instruct,
        "generate_func": ml.generate_phi_3_5,  # Используем специальную функцию для Phi-3.5
        "hyperparameters": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "torch_dtype": "bfloat16"
        }
    },
    "phi-4-mini-instruct": {
        "name": "microsoft/Phi-4-mini-instruct",
        "load_func": ml.load_phi_4_mini_instruct,
        "generate_func": ml.generate_standard,
        "hyperparameters": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "dtype": "bfloat16"
        }
    },
    "mistral-7b-v0.3-bnb-4bit": {
        "name": "unsloth/mistral-7b-v0.3-bnb-4bit",
        "load_func": ml.load_mistral_7b_v0_3_bnb_4bit,
        "generate_func": ml.generate_standard,
        "hyperparameters": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "quantization": "4-bit (pre-quantized)"
        }
    }
}


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
        print("Использование: python main.py <model_name> [--multi-agent MODE]")
        print("\nАргументы:")
        print("  <model_name>     - ключ модели из конфигурации")
        print("  --multi-agent     - (опционально) режим мультиагентного подхода")
        print("                      Доступные режимы: simple_4agents")
        print("\nПримеры:")
        print("  python main.py qwen-2.5-3b")
        print("  python main.py qwen-2.5-3b --multi-agent simple_4agents")
        print("\nДоступные модели:")
        for key in MODEL_CONFIGS.keys():
            print(f"  - {key}")
        return
    
    model_key = sys.argv[1]
    
    if model_key not in MODEL_CONFIGS:
        print(f"Модель '{model_key}' не найдена.")
        print("Доступные модели:", ", ".join(MODEL_CONFIGS.keys()))
        return
    
    # Парсим аргументы командной строки для мультиагентного режима
    multi_agent_mode = None
    if len(sys.argv) > 2:
        if "--multi-agent" in sys.argv:
            idx = sys.argv.index("--multi-agent")
            if idx + 1 < len(sys.argv):
                multi_agent_mode = sys.argv[idx + 1]
            else:
                print("Ошибка: после --multi-agent должен быть указан режим (например, simple_4agents)")
                return
        else:
            print(f"Неизвестный аргумент: {sys.argv[2]}")
            print("Использование: python main.py <model_name> [--multi-agent MODE]")
            return
    
    # Проверяем существование датасета
    if not os.path.exists(DATASET_PATH):
        print(f"Датасет не найден: {DATASET_PATH}")
        print("Убедитесь, что файл results_var3.xlsx находится в папке data/")
        return
    
    print(f"\n{'='*80}")
    print(f"ЗАПУСК ОЦЕНКИ МОДЕЛИ")
    print(f"{'='*80}")
    print(f"📌 Модель: {model_key}")
    print(f"📌 Полное название: {MODEL_CONFIGS[model_key]['name']}")
    if multi_agent_mode:
        print(f"📌 Режим: Мультиагентный ({multi_agent_mode})")
    else:
        print(f"📌 Режим: Одноагентный")
    print(f"📁 Датасет: {DATASET_PATH}")
    print(f"📁 Результаты: {OUTPUT_DIR}")
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Создаем копию конфигурации и добавляем multi_agent_mode если указан
    import copy
    config = copy.deepcopy(MODEL_CONFIGS[model_key])
    if multi_agent_mode:
        config["hyperparameters"]["multi_agent_mode"] = multi_agent_mode
    
    result = run_evaluation(config, use_gemini=use_gemini, verbose=True)  # Подробный вывод для main.py
    
    if result.get("status") != "error":
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
            print(f"     - Accuracy: {mass.get('средняя_точность', 0):.2%}")
            print(f"     - Precision: {mass.get('precision', 0):.2%}, Recall: {mass.get('recall', 0):.2%}, F1: {mass.get('f1', 0):.2%}")
            print(f"   • 'прочее':")
            print(f"     - Accuracy: {prochee.get('средняя_точность', 0):.2%}")
            print(f"     - Precision: {prochee.get('precision', 0):.2%}, Recall: {prochee.get('recall', 0):.2%}, F1: {prochee.get('f1', 0):.2%}")
        
        print(f"\n📁 Результаты сохранены в директории: {OUTPUT_DIR}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

