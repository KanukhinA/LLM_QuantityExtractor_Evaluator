"""
Конфигурация проекта
"""
import os

# Убеждаемся, что есть доступная временная директория (избегаем FileNotFoundError в tempfile).
# Ошибка "No usable temporary directory" чаще из-за отсутствия/прав каталога; реже — из-за нехватки места на диске или (если /tmp в RAM) нехватки RAM.
_dirlist = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
if not os.path.isdir(_dirlist):
    try:
        os.makedirs(_dirlist, exist_ok=True)
    except OSError:
        pass
if os.path.isdir(_dirlist) and os.access(_dirlist, os.W_OK):
    if "TMPDIR" not in os.environ:
        os.environ["TMPDIR"] = _dirlist
    if "TEMP" not in os.environ:
        os.environ["TEMP"] = _dirlist
    if "TMP" not in os.environ:
        os.environ["TMP"] = _dirlist

# Подавляем предупреждение о symlinks на Windows
# Должно быть установлено до импорта model_loaders
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Загружаем API ключи из config_loader
from config_loader import load_api_keys
HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY = load_api_keys()

# Конфигурации моделей загружаются через model_config_loader.py
# Импортируем MODEL_CONFIGS из модуля загрузчика
from model_config_loader import MODEL_CONFIGS

# ============================================================================
# Настройки датасета
# ============================================================================

# Название файла датасета
DATASET_FILENAME = "results_var3.xlsx"

# Путь к ground truth (опционально)
# По умолчанию ground truth загружается из колонки "json_parsed"
# того же файла датасета.
# Если в датасете нет колонки "json_parsed", можно указать отдельный
# файл здесь.
# Если None и нет json_parsed, метрики качества не будут вычисляться.
GROUND_TRUTH_PATH = None

# ============================================================================
# Настройки вывода
# ============================================================================

# Директория для результатов
OUTPUT_DIR = "results"

# ============================================================================
# Настройки промптов
# ============================================================================

# Конфигурация базового промпта для одноагентного подхода
# Укажите название переменной промпта из prompt_config.py
#
# Доступные варианты:
# - "DETAILED_INSTR_ZEROSHOT_BASELINE" - детальный zero-shot промпт без примера (baseline)
# - "DETAILED_INSTR_ONESHOT" - детальный промпт с примером текста и ответа (One-shot prompt)
# - "MINIMAL_FIVESHOT_PROMPT" - минималистичный few-shot промпт с 5 примерами
# - "MINIMAL_FIVESHOT_APIE_PROMPT" - few-shot промпт с 5 примерами (версия APIE)
PROMPT_TEMPLATE_NAME = "DETAILED_INSTR_ZEROSHOT_BASELINE"

# ============================================================================
# Настройки few-shot extractor
# ============================================================================

# Путь к неразмеченному корпусу (Excel файл)
# Используется для подбора few-shot примеров при помощи
# few-shot_extractor.py
UNLABELED_CORPUS_PATH = "data/udobrenia_unlabeled.xlsx"

# ============================================================================
# Конфигурации моделей
# ============================================================================

# Путь к файлу конфигурации моделей
MODELS_CONFIG_PATH = "models.yaml"

# ============================================================================
# Лимиты времени инференса
# ============================================================================

# Максимальное время инференса на одну модель (в минутах)
# Если время превышено, инференс прерывается досрочно
MAX_INFERENCE_TIME_MINUTES = 20

# ============================================================================
# Flash Attention 2 для локальных моделей
# ============================================================================

# Использовать Flash Attention 2 при загрузке локальных моделей (экономия VRAM, ускорение).
# По умолчанию включено; применяется только если установлен flash-attn (иначе предупреждение и обычная загрузка).
# Требуется: pip install flash-attn --no-build-isolation (CUDA, Linux/WSL; на Windows может быть недоступен).
# Отключить: set USE_FLASH_ATTENTION_2=0
USE_FLASH_ATTENTION_2 = os.environ.get("USE_FLASH_ATTENTION_2", "1").lower() in ("1", "true", "yes")
