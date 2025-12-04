"""
Скрипт для запуска оценки всех моделей подряд
"""
import sys
import os
from main import MODEL_CONFIGS, run_evaluation
from gemini_analyzer import check_gemini_api
from config import GEMINI_API_KEY

def run_all_models():
    """Запускает оценку всех моделей из конфигурации"""
    # Проверяем работоспособность Gemini API в самом начале
    print(f"\n{'='*80}")
    print(f"🔍 ПРОВЕРКА СИСТЕМЫ")
    print(f"{'='*80}")
    print(f"🔍 Проверка работоспособности Gemini API...")
    gemini_working, gemini_message = check_gemini_api(GEMINI_API_KEY)
    print(f"   {gemini_message}\n")
    
    use_gemini = True
    if not gemini_working:
        print(f"{'='*80}")
        print(f"⚠️  ВНИМАНИЕ: Gemini API недоступен")
        print(f"{'='*80}")
        print(f"Оценка моделей будет выполнена, но анализ ошибок через Gemini будет пропущен.")
        print(f"Вы можете продолжить без анализатора ошибок или исправить проблему и запустить заново.\n")
        
        while True:
            response = input("Продолжить без анализатора ошибок? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'да', 'д']:
                use_gemini = False
                print("✅ Продолжаем без анализатора ошибок...\n")
                break
            elif response in ['n', 'no', 'нет', 'н']:
                print("❌ Запуск отменён. Исправьте проблему с Gemini API и попробуйте снова.")
                return
            else:
                print("Пожалуйста, введите 'y' (да) или 'n' (нет)")
    
    models = list(MODEL_CONFIGS.keys())
    
    print(f"\n{'='*80}")
    print(f"🚀 ЗАПУСК ОЦЕНКИ ВСЕХ МОДЕЛЕЙ")
    print(f"{'='*80}")
    print(f"📊 Количество моделей: {len(models)}")
    print(f"📋 Модели: {', '.join(models)}\n")
    
    results_summary = []
    
    for i, model_key in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"Модель {i}/{len(models)}: {model_key}")
        print(f"{'='*80}\n")
        
        try:
            config = MODEL_CONFIGS[model_key]
            result = run_evaluation(config, use_gemini=use_gemini)
            
            if result.get("status") != "error":
                results_summary.append({
                    "model": model_key,
                    "status": "success",
                    "avg_speed": result.get("average_response_time_seconds"),
                    "parsing_error_rate": result.get("parsing_error_rate"),
                    "memory_gb": result.get("gpu_memory_during_inference_gb")
                })
                print(f"✅ Модель {model_key} успешно оценена\n")
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"⚠️ Модель {model_key} пропущена из-за ошибки: {error_msg}\n")
                results_summary.append({
                    "model": model_key,
                    "status": "error",
                    "error": error_msg
                })
        except KeyboardInterrupt:
            print(f"\n⚠️ Прервано пользователем. Остановка оценки моделей.")
            break
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"❌ Критическая ошибка при оценке {model_key}: {error_msg}")
            print(f"   Детали: {traceback.format_exc()[:500]}...")
            print(f"⚠️ Модель {model_key} пропущена\n")
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
    
    print(f"📊 Общая статистика:")
    print(f"   • Всего моделей: {len(results_summary)}")
    print(f"   • Успешно оценено: {len(successful)}")
    print(f"   • Пропущено из-за ошибок: {len(failed)}")
    print()
    
    if successful:
        print(f"✅ УСПЕШНО ОЦЕНЕННЫЕ МОДЕЛИ:")
        for summary in successful:
            print(f"   • {summary['model']}")
            print(f"     - Скорость: {summary['avg_speed']:.3f} сек/ответ")
            print(f"     - Ошибки парсинга: {summary['parsing_error_rate']:.2%}")
            print(f"     - Память: {summary['memory_gb']:.2f} GB")
        print()
    
    if failed:
        print(f"❌ ПРОПУЩЕННЫЕ МОДЕЛИ:")
        for summary in failed:
            print(f"   • {summary['model']}: {summary.get('error', 'Unknown error')[:100]}")
        print()

if __name__ == "__main__":
    run_all_models()

