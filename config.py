"""
Конфигурация проекта
"""
import os

# Подавляем предупреждение о symlinks на Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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

# Загрузка ключей с приоритетом файла config_secrets.py
HF_TOKEN = None
GEMINI_API_KEY = None

# Приоритет 1: Пытаемся загрузить из config_secrets.py
try:
    from config_secrets import HF_TOKEN as _hf_token, GEMINI_API_KEY as _gemini_key
    if _hf_token and _hf_token.strip():
        HF_TOKEN = _hf_token.strip()
    if _gemini_key and _gemini_key.strip():
        GEMINI_API_KEY = _gemini_key.strip()
except ImportError:
    # Файл config_secrets.py не найден - это нормально, используем переменные окружения
    pass
except Exception as e:
    # Если файл есть, но есть ошибка при импорте - выводим предупреждение
    print(f"Предупреждение: не удалось загрузить config_secrets.py: {e}")
    print("Будет использована переменная окружения, если она установлена.")

# Приоритет 2: Если не загрузили из файла, пробуем переменные окружения
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        HF_TOKEN = HF_TOKEN.strip()

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        GEMINI_API_KEY = GEMINI_API_KEY.strip()

# Проверка обязательного HF_TOKEN
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN не установлен!\n"
        "Создайте файл config_secrets.py на основе config_secrets.py.example и заполните свои ключи,\n"
        "или установите переменную окружения:\n"
        "  Windows: set HF_TOKEN=your_token_here\n"
        "  Linux/Mac: export HF_TOKEN=your_token_here"
    )

# Нормализуем GEMINI_API_KEY (пустая строка = None)
if GEMINI_API_KEY == "":
    GEMINI_API_KEY = None

# Настройки по умолчанию для генерации
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_NUM_RETRIES = 2

