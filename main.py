"""
Главный файл для запуска оценки моделей
"""
import os
import sys
from datetime import datetime
from model_evaluator import ModelEvaluator
import model_loaders as ml
from gemini_analyzer import analyze_errors_with_gemini, check_gemini_api
from config import DATASET_PATH, GROUND_TRUTH_PATH, OUTPUT_DIR, GEMINI_API_KEY


def run_evaluation(model_config: dict, use_gemini: bool = True):
    """
    Запускает оценку модели
    
    Args:
        model_config: словарь с конфигурацией модели:
            - name: название модели
            - load_func: функция загрузки модели
            - generate_func: функция генерации
            - hyperparameters: гиперпараметры
        use_gemini: использовать ли анализ через Gemini API
    """
    evaluator = ModelEvaluator(
        dataset_path=DATASET_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Очищаем память перед загрузкой новой модели
    evaluator.clear_memory()
    
    # Запускаем оценку
    result = evaluator.evaluate_model(
        model_name=model_config["name"],
        load_model_func=model_config["load_func"],
        generate_func=model_config["generate_func"],
        hyperparameters=model_config["hyperparameters"]
    )
    
    if result.get("status") == "error":
        print(f"Ошибка при оценке модели: {result.get('error')}")
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
        
        if not GEMINI_API_KEY:
            print("GEMINI_API_KEY не установлен, пропускаем анализ через Gemini")
        else:
            print("Отправка запроса к Gemini...")
            prompt_full_text = result.get("prompt_full_text")
            gemini_analysis = analyze_errors_with_gemini(
                model_name=model_config["name"],
                parsing_errors=parsing_errors,
                quality_metrics=quality_metrics or {},
                hyperparameters=hyperparameters,
                prompt_full_text=prompt_full_text,
                gemini_api_key=GEMINI_API_KEY
            )
            
            if gemini_analysis.get("status") == "success":
                print("Анализ от Gemini получен успешно!")
                print(f"\n{'─'*80}")
                print("📝 АНАЛИЗ И РЕКОМЕНДАЦИИ:")
                print(f"{'─'*80}")
                analysis_text = gemini_analysis.get("analysis", "")
                print(analysis_text)
                print(f"{'─'*80}\n")
                
                # Сохраняем анализ в JSON
                timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
                model_name_safe = model_config["name"].replace("/", "_").replace("\\", "_")
                analysis_path = os.path.join(OUTPUT_DIR, f"gemini_analysis_{model_name_safe}_{timestamp}.json")
                
                analysis_data = {
                    "model_name": model_config["name"],
                    "timestamp": timestamp,
                    "analysis": analysis_text,
                    "model_used": gemini_analysis.get("model_used", "gemini-2.5-flash"),
                    "parsing_errors_count": len(parsing_errors),
                    "quality_metrics_summary": {
                        "массовая доля": {
                            "precision": quality_metrics.get('массовая доля', {}).get('precision', 0) if quality_metrics else 0,
                            "recall": quality_metrics.get('массовая доля', {}).get('recall', 0) if quality_metrics else 0,
                            "f1": quality_metrics.get('массовая доля', {}).get('f1', 0) if quality_metrics else 0
                        },
                        "прочее": {
                            "precision": quality_metrics.get('прочее', {}).get('precision', 0) if quality_metrics else 0,
                            "recall": quality_metrics.get('прочее', {}).get('recall', 0) if quality_metrics else 0,
                            "f1": quality_metrics.get('прочее', {}).get('f1', 0) if quality_metrics else 0
                        }
                    } if quality_metrics else None
                }
                
                import json
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                print(f"Анализ сохранен в JSON: {analysis_path}\n")
            else:
                print(f"Не удалось получить анализ от Gemini")
                print(f"   Причина: {gemini_analysis.get('message', 'Unknown error')}\n")
    else:
        print(f"\nАнализ через Gemini API пропущен (API недоступен или отключен пользователем)\n")
    
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
    "Ministral-3-3B-Instruct-2512": {
        "name": "mistralai/Ministral-3-3B-Instruct-2512",
        "load_func": ml.load_ministral_3_3b_instruct_2512,
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
        "generate_func": ml.generate_standard,
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
        print("Использование: python main.py <model_name>")
        print("\nДоступные модели:")
        for key in MODEL_CONFIGS.keys():
            print(f"  - {key}")
        return
    
    model_key = sys.argv[1]
    
    if model_key not in MODEL_CONFIGS:
        print(f"Модель '{model_key}' не найдена.")
        print("Доступные модели:", ", ".join(MODEL_CONFIGS.keys()))
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
    print(f"📁 Датасет: {DATASET_PATH}")
    print(f"📁 Результаты: {OUTPUT_DIR}")
    print(f"📅 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    config = MODEL_CONFIGS[model_key]
    result = run_evaluation(config, use_gemini=use_gemini)
    
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

