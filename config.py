"""
Конфигурация проекта
"""
import os

# Подавляем предупреждение о symlinks на Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Название файла датасета
DATASET_FILENAME = "results_var3.xlsx"

# Путь к ground truth (опционально)
# По умолчанию ground truth загружается из колонки "json_parsed" того же файла датасета.
# Если в датасете нет колонки "json_parsed", можно указать отдельный файл здесь.
# Если None и нет json_parsed, метрики качества не будут вычисляться.
GROUND_TRUTH_PATH = None

# Директория для результатов
OUTPUT_DIR = "results"

# Импортируем API ключи из utils
from utils import HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY

# Конфигурация базового промпта для одноагентного подхода
# Укажите название переменной промпта из prompt_config.py
# Доступные варианты:
# - "FERTILIZER_EXTRACTION_PROMPT_WITH_EXAMPLE" - промпт с примером текста и ответа (по умолчанию) One-shot prompt
# - "FERTILIZER_EXTRACTION_PROMPT_TEMPLATE" - baseline промпт без примера
# - "MINIMAL_FIVESHOT_PROMPT" - промпт few-shot с 5 примерами
PROMPT_TEMPLATE_NAME = "FERTILIZER_EXTRACTION_PROMPT_TEMPLATE"

# Пути для few-shot extractor
# Путь к неразмеченному корпусу (Excel файл)
UNLABELED_CORPUS_PATH = "data/udobrenia_unlabeled.xlsx"  # Укажите путь к неразмеченным данным (используется для подбора few-shot примеров при помощи few-shot_extractor.py)

