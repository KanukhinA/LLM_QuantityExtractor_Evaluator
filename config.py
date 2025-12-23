"""
Конфигурация проекта
"""
import os

# Подавляем предупреждение о symlinks на Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Путь к датасету
# Сначала проверяем переменную окружения (приоритет 1)
if os.environ.get("DATASET_PATH"):
    DATASET_PATH = os.path.abspath(os.environ.get("DATASET_PATH"))
else:
    # Пробуем несколько вариантов расположения файла
    _config_dir = os.path.dirname(os.path.abspath(__file__))
    _base_dir = os.path.dirname(_config_dir)  # Родительская директория SmallLLMEvaluator
    _cwd = os.getcwd()  # Текущая рабочая директория

    # Определяем директорию запуска скрипта (если запускается через %run в Jupyter)
    # Пытаемся найти директорию, где находится run_all_models.py или main.py
    _script_dir = _cwd
    try:
        import inspect
        # Пытаемся найти директорию вызывающего скрипта
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if frame and frame.f_globals.get('__file__'):
                _script_file = frame.f_globals['__file__']
                _script_dir = os.path.dirname(os.path.abspath(_script_file))
                break
    except:
        pass

    # Список возможных путей для поиска
    _possible_paths = [
        # Вариант 1: В родительской директории SmallLLMEvaluator (../data/)
        os.path.join(_base_dir, "data", "results_var3.xlsx"),
        # Вариант 2: На уровень выше от родительской директории (../../data/)
        os.path.join(_base_dir, "..", "data", "results_var3.xlsx"),
        # Вариант 3: В директории запуска скрипта (если data рядом со скриптом)
        os.path.join(_script_dir, "data", "results_var3.xlsx"),
        # Вариант 4: На уровень выше от директории запуска скрипта
        os.path.join(_script_dir, "..", "data", "results_var3.xlsx"),
        # Вариант 5: Относительно текущей рабочей директории
        os.path.join(_cwd, "data", "results_var3.xlsx"),
        # Вариант 6: На уровень выше от текущей рабочей директории
        os.path.join(_cwd, "..", "data", "results_var3.xlsx"),
        # Вариант 7: В директории config (на случай, если data рядом с SmallLLMEvaluator)
        os.path.join(_config_dir, "..", "data", "results_var3.xlsx"),
        # Вариант 8: Относительный путь (data/results_var3.xlsx)
        os.path.join("data", "results_var3.xlsx"),
    ]

    # Ищем первый существующий файл
    DATASET_PATH = None
    _checked_paths = []
    for path in _possible_paths:
        abs_path = os.path.abspath(path)
        _checked_paths.append(abs_path)
        if os.path.exists(abs_path):
            DATASET_PATH = abs_path
            break

    # Если не нашли, используем первый вариант как дефолтный (для вывода ошибки)
    if DATASET_PATH is None:
        DATASET_PATH = os.path.abspath(_possible_paths[0])

    # Проверяем, что файл найден, и выводим предупреждение, если нет
    if not os.path.exists(DATASET_PATH):
        import warnings
        paths_list = "\n".join([f"   {i+1}. {path}" for i, path in enumerate(_checked_paths)])
        warnings.warn(
            f"⚠️  Датасет не найден по пути: {DATASET_PATH}\n"
            f"   Проверенные пути:\n{paths_list}\n"
            f"   Текущая рабочая директория: {_cwd}\n"
            f"   Директория config.py: {_config_dir}\n"
            f"   Родительская директория: {_base_dir}\n"
            f"   Директория запуска скрипта: {_script_dir}\n"
            f"   Убедитесь, что файл results_var3.xlsx находится в папке data/\n"
            f"   Или установите переменную окружения: export DATASET_PATH=/path/to/data/results_var3.xlsx",
            UserWarning
        )

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

