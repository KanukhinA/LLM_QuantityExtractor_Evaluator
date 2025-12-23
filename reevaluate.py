"""
Скрипт для переоценки результатов из сохраненного файла без повторного запуска модели
"""
import os
import sys
import json
import glob
from model_evaluator import ModelEvaluator
from config import DATASET_PATH, GEMINI_API_KEY

def main():
    if len(sys.argv) < 2:
        print("Использование: python reevaluate.py <путь_к_csv_файлу_с_результатами> [имя_модели] [--gemini]")
        print("\nПример:")
        print("  python reevaluate.py results/results_google_gemma-2-2b-it_20251203_123456.csv")
        print("  python reevaluate.py results/results_google_gemma-2-2b-it_20251203_123456.csv 'google/gemma-2-2b-it'")
        print("  python reevaluate.py results/results_google_gemma-2-2b-it_20251203_123456.csv 'google/gemma-2-2b-it' --gemini")
        sys.exit(1)
    
    results_csv_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    use_gemini = '--gemini' in sys.argv
    
    # Проверяем существование файла
    if not os.path.exists(results_csv_path):
        print(f"❌ Ошибка: файл не найден: {results_csv_path}")
        sys.exit(1)
    
    # Проверяем существование датасета
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Ошибка: файл датасета не найден: {DATASET_PATH}")
        sys.exit(1)
    
    # Определяем директорию для сохранения результатов
    output_dir = os.path.dirname(results_csv_path) or "results"
    
    try:
        # Выполняем переоценку
        result = ModelEvaluator.reevaluate_from_file(
            results_csv_path=results_csv_path,
            dataset_path=DATASET_PATH,
            output_dir=output_dir,
            model_name=model_name,
            use_gemini_analysis=use_gemini,
            gemini_api_key=GEMINI_API_KEY if use_gemini else None
        )
        
        print(f"\n✅ Переоценка успешно завершена!")
        print(f"📁 Результаты сохранены в: {output_dir}")
        
        # Сохраняем анализ Gemini в JSON, если он был выполнен
        gemini_analysis = result.get("gemini_analysis")
        if gemini_analysis and gemini_analysis.get("status") == "success":
            from datetime import datetime
            import json
            timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_name_safe = result.get("model_name", "unknown").replace("/", "_").replace("\\", "_")
            analysis_path = os.path.join(output_dir, f"gemini_analysis_{model_name_safe}_{timestamp}_reevaluated.json")
            
            analysis_text = gemini_analysis.get("analysis", "")
            quality_metrics = result.get("quality_metrics", {})
            
            # Пытаемся загрузить гиперпараметры и системную информацию из исходного файла метрик
            hyperparameters = result.get("hyperparameters", {})
            system_info = None
            
            # Ищем исходный файл метрик (без _reevaluated)
            model_name_for_pattern = result.get("model_name", "unknown").replace("/", "_").replace("\\", "_")
            metrics_file_pattern = os.path.join(os.path.dirname(results_csv_path), f"metrics_{model_name_for_pattern}_*.json")
            metrics_files = glob.glob(metrics_file_pattern)
            # Исключаем файлы с _reevaluated
            original_metrics_files = [f for f in metrics_files if "_reevaluated" not in f]
            
            if original_metrics_files:
                try:
                    # Берем последний исходный файл метрик
                    with open(original_metrics_files[-1], 'r', encoding='utf-8') as f:
                        original_metrics = json.load(f)
                    
                    # Извлекаем информацию из исходного файла
                    hyperparameters = original_metrics.get("hyperparameters", hyperparameters)
                    gpu_info = original_metrics.get("gpu_info", {})
                    average_response_time = original_metrics.get("average_response_time_seconds", None)
                    gpu_memory_during_inference = original_metrics.get("gpu_memory_during_inference_gb", None)
                    api_model = original_metrics.get("api_model", False)
                    multi_agent_mode = original_metrics.get("multi_agent_mode")
                    
                    system_info = {
                        "api_model": api_model,
                        "multi_agent_mode": multi_agent_mode,
                        "gpu_info": gpu_info,
                        "gpu_memory_during_inference_gb": gpu_memory_during_inference,
                        "average_response_time_seconds": average_response_time,
                        "source": "from_original_metrics_file"
                    }
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить информацию из исходного файла метрик: {e}")
            
            # Если системная информация не была загружена, создаем минимальную версию
            if system_info is None:
                system_info = {
                    "note": "system_info_not_available_for_reevaluation",
                    "source": "reevaluated_from_file"
                }
            
            analysis_data = {
                "model_name": result.get("model_name"),
                "timestamp": timestamp,
                "reevaluated_from": results_csv_path,
                "analysis": analysis_text,
                "model_used": gemini_analysis.get("model_used", "gemini-2.5-flash"),
                "parsing_errors_count": len(result.get("parsing_errors", [])),
                "hyperparameters": hyperparameters,
                "system_info": system_info,
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
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Анализ Gemini сохранен в JSON: {analysis_path}\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка при переоценке: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

