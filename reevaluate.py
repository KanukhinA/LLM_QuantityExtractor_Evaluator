"""
Скрипт для переоценки результатов из сохраненного файла без повторного запуска модели
"""
import os
import sys
import json
import glob
from model_evaluator import ModelEvaluator
from config import GEMINI_API_KEY
from utils import find_dataset_path

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
    dataset_path = find_dataset_path()
    if not os.path.exists(dataset_path):
        print(f"❌ Ошибка: файл датасета не найден: {dataset_path}")
        sys.exit(1)
    
    # Определяем директорию для сохранения результатов
    output_dir = os.path.dirname(results_csv_path) or "results"
    
    try:
        # Выполняем переоценку
        result = ModelEvaluator.reevaluate_from_file(
            results_csv_path=results_csv_path,
            dataset_path=dataset_path,
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
            timestamp = result.get("timestamp", datetime.now().strftime("%H%M%S"))
            from model_evaluator import sanitize_filename
            model_name_safe = sanitize_filename(result.get("model_name", "unknown"))
            
            analysis_text = gemini_analysis.get("analysis", "")
            quality_metrics = result.get("quality_metrics", {})
            
            # Пытаемся загрузить гиперпараметры и системную информацию из исходного файла метрик
            hyperparameters = result.get("hyperparameters", {})
            system_info = None
            multi_agent_mode = None
            prompt_full_text = None
            prompt_info = None
            prompt_template = None
            
            # Ищем исходный файл метрик (без _reevaluated)
            # Пытаемся найти в старой структуре (плоской) и новой (вложенной)
            model_name_for_pattern = sanitize_filename(result.get("model_name", "unknown"))
            csv_dir = os.path.dirname(results_csv_path)
            
            # Ищем в старой структуре (плоской)
            metrics_file_pattern = os.path.join(csv_dir, f"metrics_{model_name_for_pattern}_*.json")
            metrics_files = glob.glob(metrics_file_pattern)
            # Исключаем файлы с _reevaluated
            original_metrics_files = [f for f in metrics_files if "_reevaluated" not in f]
            
            # Если не нашли, ищем в новой структуре папок (model_key/prompt_name/)
            if not original_metrics_files:
                for root, dirs, files in os.walk(csv_dir):
                    for file in files:
                        if file.startswith("metrics_") and file.endswith(".json") and "_reevaluated" not in file:
                            file_path = os.path.join(root, file)
                            original_metrics_files.append(file_path)
            
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
                    prompt_full_text = original_metrics.get("prompt_full_text")
                    prompt_info = original_metrics.get("prompt_info")
                    prompt_template = original_metrics.get("prompt_template")
                    
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
                "prompts": {
                    "prompt_template": prompt_template,
                    "prompt_full_text": prompt_full_text,
                    "prompt_info": prompt_info,
                    "note": "prompts_loaded_from_metrics_file" if prompt_full_text else "prompts_not_available_for_reevaluation"
                },
                "system_info": system_info,
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
            
            # Определяем структуру папок (такая же, как в reevaluate_from_file)
            from config import PROMPT_TEMPLATE_NAME
            from model_evaluator import sanitize_filename
            import re
            
            model_key = result.get("model_key")
            if not model_key:
                # Убираем дату из model_name_safe, если она там есть (формат: name_YYYYMMDD)
                model_key = model_name_safe
                # Убираем паттерны типа _20260123 или _20260123_123456
                model_key = re.sub(r'_\d{8}(_\d{6})?$', '', model_key)
                if not model_key:  # Если после удаления даты ничего не осталось
                    model_key = model_name_safe
            else:
                model_key = sanitize_filename(model_key)
            
            prompt_template_name = result.get("prompt_template") or prompt_template
            # Если в старом файле сохранено название функции (build_prompt3), заменяем на PROMPT_TEMPLATE_NAME
            if prompt_template_name == "build_prompt3" or prompt_template_name == "build_prompt":
                prompt_template_name = PROMPT_TEMPLATE_NAME
            
            if multi_agent_mode:
                prompt_folder_name = sanitize_filename(multi_agent_mode)
            elif prompt_template_name:
                prompt_folder_name = sanitize_filename(prompt_template_name)
            else:
                prompt_folder_name = sanitize_filename(PROMPT_TEMPLATE_NAME)  # Используем текущий из config
            
            # Создаем структуру папок
            model_dir = os.path.join(output_dir, model_key)
            prompt_dir = os.path.join(model_dir, prompt_folder_name)
            os.makedirs(prompt_dir, exist_ok=True)
            
            analysis_path = os.path.join(prompt_dir, f"gemini_analysis_{timestamp}_reevaluated.json")
            
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

