"""
Конфигурация проекта
"""
import os

# Путь к датасету
# По умолчанию ищем в родительской директории (где находится папка data)
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(_base_dir, "data", "results_var3.xlsx")

# Если файл не найден, попробуем относительный путь
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join("data", "results_var3.xlsx")

# Путь к ground truth (опционально, устаревший параметр)
# По умолчанию ground truth загружается из колонки "json_parsed" того же файла (DATASET_PATH).
# Если в датасете нет колонки "json_parsed", можно указать отдельный файл здесь.
# Если None и нет json_parsed, метрики качества (пункты 5-6) не будут вычисляться.
GROUND_TRUTH_PATH = None

# Директория для результатов
OUTPUT_DIR = "results"

# HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCMTsCnbXIAcwOqWNuyTGM-a7TdGpGUjno")
#os.environ["GEMINI_API_KEY"] = "AIzaSyDUtC9E6388RYJde5hMC3jI8MDn6z4Tc5s"
#os.environ["GEMINI_API_KEY"] = "AIzaSyCMTsCnbXIAcwOqWNuyTGM-a7TdGpGUjno"

# Настройки по умолчанию для генерации
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_NUM_RETRIES = 2

