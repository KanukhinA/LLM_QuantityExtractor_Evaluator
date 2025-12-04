"""
Скрипт для переоценки результатов из сохраненного файла без повторного запуска модели
"""
import os
import sys
from model_evaluator import ModelEvaluator
from config import DATASET_PATH

def main():
    if len(sys.argv) < 2:
        print("Использование: python reevaluate.py <путь_к_csv_файлу_с_результатами> [имя_модели]")
        print("\nПример:")
        print("  python reevaluate.py results/results_google_gemma-2-2b-it_20251203_123456.csv")
        print("  python reevaluate.py results/results_google_gemma-2-2b-it_20251203_123456.csv 'google/gemma-2-2b-it'")
        sys.exit(1)
    
    results_csv_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
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
            model_name=model_name
        )
        
        print(f"\n✅ Переоценка успешно завершена!")
        print(f"📁 Результаты сохранены в: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при переоценке: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

