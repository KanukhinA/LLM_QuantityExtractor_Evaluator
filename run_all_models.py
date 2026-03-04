"""
Скрипт для запуска оценки всех моделей подряд
"""
import os
import sys
import argparse
import logging
from main import run_evaluation
from gemini_analyzer import check_gemini_api
from config import GEMINI_API_KEY, MODEL_CONFIGS, OUTPUT_DIR
from utils import ConsoleLogCapture
from model_evaluator import StopAllModelsInterrupt, _append_to_model_errors_log

def run_all_models(local_only: bool = False, multi_agent_mode: str = None,
                   structured_output: bool = False, use_outlines: bool = False,
                   prompt_template_name: str = None, pydantic_outlines: bool = False,
                   use_guidance: bool = False):
    """Запускает оценку всех моделей из конфигурации"""
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
        print(f"Оценка будет выполнена без анализа ошибок через Gemini.\n")
        use_gemini = False
    
    # Выводим информацию о режимах
    if multi_agent_mode:
        print(f"📌 Режим: Мультиагентный ({multi_agent_mode})")
    else:
        print(f"📌 Режим: Одноагентный")
    if structured_output:
        print(f"📌 Structured Output: Включен (Pydantic валидация)")
    if use_outlines or pydantic_outlines:
        print(f"📌 Outlines: Включен" + (" (схема из Pydantic)" if pydantic_outlines else " (outlines_schema.py)"))
    if use_guidance:
        effective_prompt = prompt_template_name or "DETAILED_INSTR_ZEROSHOT_BASELINE_OUTLINES_RUS"
        print(f"📌 Guidance (llguidance): Включен, промпт: {effective_prompt}")
    if prompt_template_name:
        print(f"📌 Промпт: {prompt_template_name}")
    print()
    
    # Фильтруем модели, если указан флаг local_only
    all_models = list(MODEL_CONFIGS.keys())
    if local_only:
        # Локальные модели - это те, у которых нет "-api" в ключе
        models = [model_key for model_key in all_models if "-api" not in model_key]
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ ЛОКАЛЬНЫХ МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"Всего моделей в конфигурации: {len(all_models)}")
        print(f"Локальных моделей: {len(models)}")
        print(f"API моделей (пропущено): {len(all_models) - len(models)}")
    else:
        models = all_models
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ ВСЕХ МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"Количество моделей: {len(models)}")
    
    if not models:
        print(f"⚠️  Не найдено моделей для оценки.")
        if local_only:
            print(f"   Попробуйте запустить без флага --local-only для оценки всех моделей.")
        return
    
    print(f"Модели: {', '.join(models)}\n")
    
    results_summary = []
    
    for i, model_key in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"Модель {i}/{len(models)}: {model_key}")
        print(f"{'='*80}\n")
        
        try:
            # Создаем копию конфигурации и добавляем параметры если указаны
            import copy
            config = copy.deepcopy(MODEL_CONFIGS[model_key])
            if multi_agent_mode:
                config["hyperparameters"]["multi_agent_mode"] = multi_agent_mode
            if structured_output:
                config["hyperparameters"]["structured_output"] = True
            if use_outlines or pydantic_outlines:
                config["hyperparameters"]["use_outlines"] = True
                config["hyperparameters"]["pydantic_outlines"] = pydantic_outlines
            if use_guidance:
                config["hyperparameters"]["use_guidance"] = True
                config["hyperparameters"]["prompt_template_name"] = (
                    prompt_template_name or "DETAILED_INSTR_ZEROSHOT_BASELINE_OUTLINES_RUS"
                )
            elif prompt_template_name is not None:
                config["hyperparameters"]["prompt_template_name"] = prompt_template_name
            
            result = run_evaluation(config, model_key=model_key, use_gemini=use_gemini, verbose=False, stop_all_on_interrupt=True)  # Короткий вывод; при Ctrl+C — опция прервать все модели
            
            if result.get("status") == "interrupted":
                print(f"Модель {model_key} прервана (без сохранения)\n")
                results_summary.append({
                    "model": model_key,
                    "status": "interrupted",
                    "message": result.get("message", "Прервано пользователем")
                })
            elif result.get("status") != "error":
                # Проверяем, была ли модель прервана по времени
                if result.get("interrupted") and result.get("timeout_reason"):
                    timeout_reason = result.get("timeout_reason")
                    print(f"Модель {model_key} прервана: {timeout_reason}\n")
                    results_summary.append({
                        "model": model_key,
                        "status": "timeout",
                        "timeout_reason": timeout_reason
                    })
                elif result.get("interrupted"):
                    print(f"Модель {model_key} прервана (без сохранения)\n")
                    results_summary.append({
                        "model": model_key,
                        "status": "interrupted",
                        "message": "Прервано пользователем"
                    })
                else:
                    results_summary.append({
                        "model": model_key,
                        "status": "success",
                        "multi_agent_mode": result.get("multi_agent_mode"),
                        "avg_speed": result.get("average_response_time_seconds"),
                        "parsing_error_rate": result.get("parsing_error_rate"),
                        "memory_gb": result.get("gpu_memory_during_inference_gb")
                    })
                    print(f"Модель {model_key} успешно оценена\n")
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"Модель {model_key} пропущена из-за ошибки: {error_msg}\n")
                results_summary.append({
                    "model": model_key,
                    "status": "error",
                    "error": error_msg
                })
        except (KeyboardInterrupt, StopAllModelsInterrupt):
            print(f"\nПрервано пользователем. Остановка оценки моделей.")
            break
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb_str = traceback.format_exc()
            print(f"Критическая ошибка при оценке {model_key}: {error_msg}")
            print(f"   Детали: {tb_str[:500]}...")
            print(f"Модель {model_key} пропущена\n")
            full_msg = f"{error_msg}\n\n{tb_str}"
            _append_to_model_errors_log(
                OUTPUT_DIR,
                "Критическая ошибка при оценке модели (исключение в run_evaluation)",
                model_key,
                full_msg,
            )
            logging.error("Оценка модели %s: %s", model_key, error_msg)
            results_summary.append({
                "model": model_key,
                "status": "error",
                "error": error_msg
            })
    
    # Выводим итоговую сводку
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СВОДКА")
    print(f"{'='*80}\n")
    
    successful = [s for s in results_summary if s['status'] == 'success']
    failed = [s for s in results_summary if s['status'] == 'error']
    timeout_models = [s for s in results_summary if s['status'] == 'timeout']
    interrupted_models = [s for s in results_summary if s['status'] == 'interrupted']
    
    print(f"Общая статистика:")
    print(f"   • Всего моделей: {len(results_summary)}")
    print(f"   • Успешно оценено: {len(successful)}")
    print(f"   • Пропущено из-за ошибок: {len(failed)}")
    print(f"   • Прервано по времени: {len(timeout_models)}")
    print(f"   • Прервано пользователем: {len(interrupted_models)}")
    print()
    
    if timeout_models:
        print(f"МОДЕЛИ, ПРЕРВАННЫЕ ИЗ-ЗА ПРЕВЫШЕНИЯ ВРЕМЕНИ ИНФЕРЕНСА:")
        for summary in timeout_models:
            print(f"   • {summary['model']}: {summary.get('timeout_reason', 'Превышен лимит времени')}")
        print()
    
    if interrupted_models:
        print(f"МОДЕЛИ, ПРЕРВАННЫЕ ПОЛЬЗОВАТЕЛЕМ (без сохранения):")
        for summary in interrupted_models:
            print(f"   • {summary['model']}: {summary.get('message', 'Прервано')}")
        print()
    
    if successful:
        print(f"УСПЕШНО ОЦЕНЕННЫЕ МОДЕЛИ:")
        for summary in successful:
            print(f"   • {summary['model']}")
            mode = summary.get('multi_agent_mode') or 'Одноагентный'
            print(f"     - Режим: {mode}")
            print(f"     - Скорость: {summary['avg_speed']:.3f} сек/ответ")
            print(f"     - Ошибки парсинга: {summary['parsing_error_rate']:.2%}")
            print(f"     - Память: {summary['memory_gb']:.2f} GB")
        print()
    
    if failed:
        print(f"ПРОПУЩЕННЫЕ МОДЕЛИ:")
        for summary in failed:
            print(f"   • {summary['model']}: {summary.get('error', 'Unknown error')[:100]}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск оценки всех моделей")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Запустить оценку только для локальных моделей (исключить API модели)"
    )
    parser.add_argument(
        "--multi-agent",
        type=str,
        metavar="MODE",
        help="Режим мультиагентного подхода (simple_4agents, critic_3agents, qa_workflow)"
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Использовать structured output через Pydantic"
    )
    outlines_group = parser.add_mutually_exclusive_group()
    outlines_group.add_argument(
        "--outlines",
        action="store_true",
        help="Использовать outlines со схемой из outlines_schema.py"
    )
    outlines_group.add_argument(
        "--pydantic-outlines",
        action="store_true",
        help="Использовать outlines со схемой из Pydantic (model_json_schema) вместо outlines_schema.py"
    )
    parser.add_argument(
        "--guidance",
        action="store_true",
        help="Constrained decoding через llguidance (по умолчанию схема RUS)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        metavar="NAME",
        help="Название промпта из prompt_config.py (например, DETAILED_INSTR_ZEROSHOT_BASELINE_OUTLINES)"
    )
    args = parser.parse_args()

    log_path = os.path.join(OUTPUT_DIR, "evaluation_summary.log")
    with ConsoleLogCapture(log_path):
        run_all_models(
            local_only=args.local_only,
            multi_agent_mode=args.multi_agent,
            structured_output=args.structured_output,
            use_outlines=args.outlines,
            prompt_template_name=args.prompt,
            pydantic_outlines=args.pydantic_outlines,
            use_guidance=args.guidance
        )

