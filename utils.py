"""
Утилиты для парсинга JSON и построения промптов
"""


def local_cache_path_for_model(repo_id):
    """
    Если репозиторий уже в кэше HF, возвращает путь к снапшоту (директория).
    Сначала ищет config.json, затем tokenizer.json (репо только с токенизатором). К сети не обращается.
    """
    import os
    if not repo_id or not isinstance(repo_id, str) or "/" not in repo_id or os.path.isabs(repo_id):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        for filename in ("config.json", "tokenizer.json"):
            path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
            if path and os.path.isfile(path):
                return os.path.dirname(path)
    except Exception:
        pass
    return None


def from_pretrained_local_first(loader, *args, **kwargs):
    """
    Сначала пробует загрузить из локального кэша: по repo_id ищет снапшот и передаёт путь в loader.
    Если кэша нет — вызов loader с исходным именем (загрузка с HF).
    """
    model_id = args[0] if args else None
    local_dir = local_cache_path_for_model(model_id)
    if local_dir:
        try:
            return loader(local_dir, *args[1:], **kwargs)
        except Exception:
            pass
    return loader(*args, **kwargs)


import io
import json
import re
import ast
import codecs
import csv
import glob
import os
import inspect
import sys
import warnings
from typing import Dict, Any, Optional
import prompt_config
from config import PROMPT_TEMPLATE_NAME, DATASET_FILENAME, OUTPUT_DIR


class _TeeWriter:
    """Пишет в оригинальный поток и в sink (буфер и/или файл)."""
    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, data):
        self._stream.write(data)
        self._sink.write(data)

    def flush(self):
        self._stream.flush()
        self._sink.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _BufferAndFileSink:
    """Пишет в буфер и в файл при каждом write; flush файла для записи на диск в реальном времени."""
    def __init__(self, buffer: io.StringIO, file_handle):
        self._buffer = buffer
        self._file = file_handle

    def write(self, data: str):
        self._buffer.write(data)
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass

    def flush(self):
        self._buffer.flush()
        try:
            self._file.flush()
        except Exception:
            pass


class ConsoleLogCapture:
    """
    Контекстный менеджер: перехватывает stdout/stderr, дублирует вывод в консоль и
    в файл evaluation_summary.log. Запись в файл идёт в реальном времени (при каждом write),
    а не только при выходе из контекста.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._buffer = None
        self._log_file = None
        self._orig_stdout = None
        self._orig_stderr = None

    def __enter__(self):
        self._buffer = io.StringIO()
        d = os.path.dirname(self.log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._log_file = open(self.log_path, "w", encoding="utf-8")
        sink = _BufferAndFileSink(self._buffer, self._log_file)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeWriter(sys.stdout, sink)
        sys.stderr = _TeeWriter(sys.stderr, sink)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        try:
            if self._log_file is not None:
                self._log_file.close()
        except Exception:
            pass
        self._log_file = None
        return None


def find_dataset_path(dataset_filename: str = None) -> str:
    """
    Ищет файл датасета в различных возможных местах.
    
    Args:
        dataset_filename: имя файла датасета (по умолчанию из config.DATASET_FILENAME)
    
    Returns:
        Абсолютный путь к файлу датасета
    """
    if dataset_filename is None:
        dataset_filename = DATASET_FILENAME
    
    # Сначала проверяем переменную окружения (приоритет 1)
    if os.environ.get("DATASET_PATH"):
        return os.path.abspath(os.environ.get("DATASET_PATH"))
    
    # Пробуем несколько вариантов расположения файла
    config_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(config_dir)  # Родительская директория SmallLLMEvaluator
    cwd = os.getcwd()  # Текущая рабочая директория
    
    # Определяем директорию запуска скрипта (если запускается через %run в Jupyter)
    # Пытаемся найти директорию, где находится run_all_models.py или main.py
    script_dir = cwd
    try:
        # Пытаемся найти директорию вызывающего скрипта
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if frame and frame.f_globals.get('__file__'):
                script_file = frame.f_globals['__file__']
                script_dir = os.path.dirname(os.path.abspath(script_file))
                break
    except:
        pass
    
    # Список возможных путей для поиска
    possible_paths = [
        # Вариант 1: В родительской директории SmallLLMEvaluator (../data/)
        os.path.join(base_dir, "data", dataset_filename),
        # Вариант 2: На уровень выше от родительской директории (../../data/)
        os.path.join(base_dir, "..", "data", dataset_filename),
        # Вариант 3: В директории запуска скрипта (если data рядом со скриптом)
        os.path.join(script_dir, "data", dataset_filename),
        # Вариант 4: На уровень выше от директории запуска скрипта
        os.path.join(script_dir, "..", "data", dataset_filename),
        # Вариант 5: Относительно текущей рабочей директории
        os.path.join(cwd, "data", dataset_filename),
        # Вариант 6: На уровень выше от текущей рабочей директории
        os.path.join(cwd, "..", "data", dataset_filename),
        # Вариант 7: В директории config (на случай, если data рядом с SmallLLMEvaluator)
        os.path.join(config_dir, "..", "data", dataset_filename),
        # Вариант 8: Относительный путь (data/results_var3.xlsx)
        os.path.join("data", dataset_filename),
    ]
    
    # Ищем первый существующий файл
    dataset_path = None
    checked_paths = []
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        checked_paths.append(abs_path)
        if os.path.exists(abs_path):
            dataset_path = abs_path
            break
    
    # Если не нашли, используем первый вариант как дефолтный (для вывода ошибки)
    if dataset_path is None:
        dataset_path = os.path.abspath(possible_paths[0])
    
    # Проверяем, что файл найден, и выводим предупреждение, если нет
    if not os.path.exists(dataset_path):
        paths_list = "\n".join([f"   {i+1}. {path}" for i, path in enumerate(checked_paths)])
        warnings.warn(
            f"⚠️  Датасет не найден по пути: {dataset_path}\n"
            f"   Проверенные пути:\n{paths_list}\n"
            f"   Текущая рабочая директория: {cwd}\n"
            f"   Директория config.py: {config_dir}\n"
            f"   Родительская директория: {base_dir}\n"
            f"   Директория запуска скрипта: {script_dir}\n"
            f"   Убедитесь, что файл {dataset_filename} находится в папке data/\n"
            f"   Или установите переменную окружения: export DATASET_PATH=/path/to/data/{dataset_filename}",
            UserWarning
        )
    
    return dataset_path


def find_file_path(relative_path: str) -> str:
    """
    Ищет файл по относительному пути (например, data/udobrenia_unlabeled.xlsx)
    в нескольких возможных базовых директориях.
    Используется для UNLABELED_CORPUS_PATH в few_shot_extractor.

    Args:
        relative_path: относительный путь к файлу (например, "data/udobrenia_unlabeled.xlsx")

    Returns:
        Абсолютный путь к найденному файлу

    Raises:
        FileNotFoundError: если файл не найден ни по одному из путей
    """
    if not relative_path or not relative_path.strip():
        raise FileNotFoundError("Путь к файлу не указан")
    path = relative_path.strip()
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Файл не найден: {path}")

    config_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(config_dir)
    cwd = os.getcwd()
    script_dir = cwd
    try:
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if frame and frame.f_globals.get("__file__"):
                script_file = frame.f_globals["__file__"]
                script_dir = os.path.dirname(os.path.abspath(script_file))
                break
    except Exception:
        pass

    possible_paths = [
        os.path.join(cwd, path),
        os.path.join(base_dir, path),
        os.path.join(script_dir, path),
        os.path.join(base_dir, "..", path),
        os.path.join(script_dir, "..", path),
        os.path.join(config_dir, "..", path),
    ]

    for p in possible_paths:
        abs_path = os.path.abspath(p)
        if os.path.exists(abs_path):
            return abs_path

    checked = "\n".join(f"   {i+1}. {os.path.abspath(p)}" for i, p in enumerate(possible_paths))
    raise FileNotFoundError(
        f"Файл не найден: {path}\nПроверенные пути:\n{checked}\n"
        f"Текущая рабочая директория: {cwd}"
    )


def get_few_shot_csv_path(model_key: str, output_dir: str = None) -> Optional[str]:
    """
    Возвращает путь к последнему по времени файлу few_shot_examples_{model_key}_*.csv в output_dir,
    или None, если такого файла нет. Используется для проверки возможности запуска MINIMAL_INSTR_FIVESHOT_APIE.
    """
    if not model_key or not output_dir:
        return None
    pattern = os.path.join(output_dir, f"few_shot_examples_{model_key}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_few_shot_examples_block(model_key: str, output_dir: str = None, max_examples: int = 5) -> str:
    """
    Собирает блок примеров "Текст примера N: ... Ответ примера N: ..." из последнего по времени
    CSV few_shot_examples_{model_key}_*.csv в output_dir. Используется для MINIMAL_INSTR_FIVESHOT_APIE.
    """
    latest = get_few_shot_csv_path(model_key, output_dir)
    if not latest:
        return ""
    try:
        with open(latest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:max_examples]
    except Exception:
        return ""
    block = []
    for i, row in enumerate(rows, 1):
        text = (row.get("text") or "").strip()
        json_val = (row.get("json") or "").strip()
        block.append(f"Текст примера {i}: {text}")
        block.append(f"Ответ примера {i}: {json_val}")
    return "\n\n".join(block)


def build_prompt3(
    text: str,
    structured_output: bool = False,
    response_schema: Any = None,
    prompt_template_name: str = None,
    model_key: Optional[str] = None,
) -> str:
    """
    Генерация промпта для конкретного текста

    Использует промпт из prompt_config.py. Имя шаблона задаётся:
    - prompt_template_name (приоритет), иначе config.PROMPT_TEMPLATE_NAME.
    Для MINIMAL_INSTR_FIVESHOT_APIE при указании model_key примеры подставляются из CSV,
    сгенерированного few_shot_extractor для этой модели.
    При structured_output=True и наличии response_schema — добавляет JSON-схему в промпт.
    """
    name = prompt_template_name if prompt_template_name is not None else PROMPT_TEMPLATE_NAME
    prompt_template = getattr(prompt_config, name)
    if name == "MINIMAL_INSTR_FIVESHOT_APIE":
        few_shot_examples = load_few_shot_examples_block(model_key or "", OUTPUT_DIR, max_examples=5)
        prompt = prompt_template.format(few_shot_examples=few_shot_examples, text=text)
    else:
        prompt = prompt_template.format(text=text)
    if structured_output and response_schema is not None and hasattr(response_schema, "model_json_schema"):
        import json as _json
        schema_dict = response_schema.model_json_schema()
        schema_str = _json.dumps(schema_dict, ensure_ascii=False, indent=2)
        prompt += f"\n\nТочная JSON-схема ожидаемого ответа:\n```json\n{schema_str}\n```"
    return prompt


def _extract_json_like(s: str) -> str:
    """
    Извлекает JSON-подстроку:
    1) fenced ```json ... ```
    2) если нет — ищет первый '{' и возвращает оттуда до конца (фрагмент, возможно обрезанный)
    """
    if not isinstance(s, str):
        return ""

    # 1) fenced block ```json ... ```
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # убрать ведущие markdown-символы и пробелы
    s_stripped = re.sub(r"^[\s\*\-#>]+", "", s.lstrip())

    # 2) найти первую фигурную скобку и вернуть фрагмент от неё до конца (включая возможную обрезку)
    idx = s_stripped.find("{")
    if idx != -1:
        return s_stripped[idx:].strip()

    # если нет '{', пробуем '['
    idx = s_stripped.find("[")
    if idx != -1:
        return s_stripped[idx:].strip()

    return ""


def _autofix_commas(s: str) -> str:
    """
    Вставляет пропущенные запятые между соседними объектами/элементами,
    удаляет лишние запятые перед закрывающими скобками.
    """
    # "}{", "}\n{" -> "}, {"
    s = re.sub(r"}\s*{", "}, {", s)
    s = re.sub(r"}\s*\n\s*{", "},\n{", s)

    # "] {" -> "], {"
    s = re.sub(r"]\s*{", "], {", s)

    # удалить запятую перед закрывающей скобкой
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*\]", "]", s)

    return s


def _balance_and_close(s: str) -> str:
    """
    Считает незакрытые скобки и дозакрывает их в конце: сначала ']' затем '}'.
    Также обрабатывает обрывы внутри строк и незавершенные значения.
    """
    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        # Обрабатываем escape-последовательности
        if ch == "\\" and not escape:
            escape = True
            continue
        elif escape:
            # Если это экранированная кавычка, не переключаем состояние строки
            if ch == '"':
                escape = False
                continue
            # Если это другой escape-символ, просто сбрасываем флаг
            escape = False
        
        # Переключаем состояние строки только на неэкранированных кавычках
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        
        # Считаем скобки только вне строк
        if not in_string:
            if ch == "{":
                depth_obj += 1
            elif ch == "}":
                if depth_obj > 0:
                    depth_obj -= 1
            elif ch == "[":
                depth_arr += 1
            elif ch == "]":
                if depth_arr > 0:
                    depth_arr -= 1

    # Если строка обрывается внутри незавершенной строки, закрываем её
    if in_string:
        s = s + '"'
        # После закрытия строки: если это был ключ в объекте ("," или "{" перед ключом),
        # нужен : null для валидного JSON
        if re.search(r'[,{]\s*"[^"]*"$', s.rstrip()):
            s = s.rstrip() + ': null'

    # Обрабатываем незавершенные значения после запятой или двоеточия
    s_stripped = s.rstrip()
    if s_stripped:
        last_non_ws = s_stripped[-1]
        # Если обрыв после двоеточия (ожидается значение), добавляем null
        if last_non_ws == ':':
            s = s.rstrip() + ' null'
        # Если обрыв после запятой, удаляем запятую (она будет лишней при дозакрытии)
        elif last_non_ws == ',':
            # Удаляем последнюю запятую, чтобы не было синтаксической ошибки
            s = s_stripped[:-1].rstrip()

    # дозакрываем: сначала массивы, потом объекты
    closing = ""
    if depth_arr > 0:
        closing += "]" * depth_arr
    if depth_obj > 0:
        closing += "}" * depth_obj

    if closing:
        s = s + closing

    return s


def parse_json_safe(s: str) -> Any:
    """
    Умный парсер JSON с автопочинкой и автодозакрытием скобок.
    Возвращает dict (успешно распарсенный объект), list или {}.
    """
    if not isinstance(s, str) or not s.strip():
        return {}

    fragment = _extract_json_like(s)
    if not fragment:
        return {}

    # Обработка escape-последовательностей
    s_clean = fragment
    try:
        # ВАЖНО: НЕ заменяем \" на " - это сломает строки с экранированными кавычками внутри!
        # В валидном JSON кавычки внутри строк должны оставаться экранированными как \"
        # Например: "значение": "ТОВАРНЫЙ ЗНАК \"БУЙСКИЕ УДОБРЕНИЯ\"" - это валидный JSON
        # Если заменить \" на ", получится невалидный JSON: "значение": "ТОВАРНЫЙ ЗНАК "БУЙСКИЕ УДОБРЕНИЯ""
        
        # Заменяем буквальные \n, \t, \r на настоящие символы (для форматирования JSON)
        # В валидном JSON внутри строк они должны быть экранированы как \\n (двойной слэш)
        # Но модели иногда выводят буквальные \n в форматировании JSON (вне строк)
        # Заменяем их аккуратно: заменяем только одиночные \n, \t, \r (не \\n)
        # Используем регулярное выражение для замены только одиночных escape-последовательностей
        # Но это сложно, поэтому просто заменяем все \n, \t, \r (в валидном JSON внутри строк должны быть \\n)
        s_clean = s_clean.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    except Exception:
        pass
    
    s_clean = s_clean.replace("\r", "").strip()

    # замены типографских кавычек на обычные (безопасно, так как типографские кавычки не валидны в JSON)
    s_clean = s_clean.replace(""", '"').replace(""", '"').replace("‚", "'").replace("'", "'")
    
    # НЕ заменяем двойные кавычки "" на одинарные " - это может сломать валидный JSON
    # В валидном JSON не должно быть двойных кавычек подряд, но если они есть в строке,
    # то они должны быть экранированы как \"

    # нормализуем None/null
    s_clean = re.sub(r"\bNone\b", "null", s_clean)

    # автопочинки запятых
    s_clean = _autofix_commas(s_clean)

    # дозакрываем скобки (обрабатывает обрывы внутри строк и незавершенные значения)
    s_clean = _balance_and_close(s_clean)

    # удалить двойные запятые
    s_clean = re.sub(r",\s*,+", ",", s_clean)
    
    # Удаляем запятые перед закрывающими скобками (после дозакрытия могут появиться лишние)
    s_clean = re.sub(r",\s*}", "}", s_clean)
    s_clean = re.sub(r",\s*\]", "]", s_clean)

    # Попытка парсинга
    try:
        parsed = json.loads(s_clean)
        if isinstance(parsed, (dict, list)):
            return parsed
        return {}
    except json.JSONDecodeError:
        pass

    # fallback: безопасный eval
    s_eval = re.sub(r"\bnull\b", "None", s_clean)
    try:
        data = ast.literal_eval(s_eval)
        if isinstance(data, (dict, list)):
            return data
        return {}
    except Exception:
        pass

    return {}


def is_valid_json(s: str) -> bool:
    """
    Проверка валидности JSON
    """
    try:
        parsed = parse_json_safe(s)
        if not parsed:
            return False
        return isinstance(parsed, dict)
    except Exception:
        return False


def extract_json_from_response(response_text: str) -> str:
    """
    Извлекает JSON часть из ответа модели.
    Ищет:
    1. Последний "ОТВЕТ:" или "Ответ:" (любой регистр) и JSON после него
    2. Последний markdown блок ```json ... ```
    3. Прямой JSON в тексте
    """
    if not response_text:
        return ""
    
    # Предварительная обработка: заменяем буквальные \n, \t, \r на настоящие символы
    # Это нужно для правильной работы регулярных выражений
    # ВАЖНО: НЕ заменяем \" на " - это сломает строки с экранированными кавычками!
    # В валидном JSON внутри строк кавычки должны быть экранированы как \"
    try:
        # Заменяем только \n, \t, \r (не трогаем \")
        # В валидном JSON внутри строк должны быть \\n (двойной слэш), а не \n
        # Но модели иногда выводят буквальные \n в форматировании JSON (вне строк)
        # Простой подход: заменяем все \n, \t, \r, но это может сломать строки с этими символами
        # Более безопасно: не заменять вообще, но тогда модели, которые выводят буквальные \n, не будут работать
        # Компромисс: заменяем только если они не экранированы двойным слэшем
        # Но это сложно определить без парсинга, поэтому оставляем как есть
        # Стандартный JSON парсер должен справиться с буквальными \n в форматировании
        pass
    except Exception:
        pass
    
    # Нормализуем регистр для поиска
    response_lower = response_text.lower()
    
    # 1. Ищем последний "ОТВЕТ:" или "Ответ:" (любой регистр)
    answer_markers = ["ответ:", "answer:"]
    for marker in answer_markers:
        # Находим все вхождения маркера
        last_idx = -1
        search_pos = 0
        while True:
            idx = response_lower.find(marker, search_pos)
            if idx == -1:
                break
            last_idx = idx
            search_pos = idx + 1
        
        if last_idx != -1:
            # Берем текст после последнего маркера
            json_part = response_text[last_idx + len(marker):].strip()
            # Убираем возможные переносы строк и пробелы в начале
            json_part = json_part.lstrip("\n\r\t ")
            
            # Ищем последний JSON блок в этом тексте
            # Сначала пробуем найти последний markdown блок ```json ... ```
            json_blocks = list(re.finditer(r"```(?:json)?\s*(.*?)\s*```", json_part, flags=re.IGNORECASE | re.DOTALL))
            if json_blocks:
                # Берем последний блок
                last_block = json_blocks[-1]
                extracted = last_block.group(1).strip()
                if extracted:
                    return extracted
            
            # Если нет markdown блоков, ищем JSON (первый { или [ — для вложенных структур)
            first_brace = json_part.find("{")
            first_bracket = json_part.find("[")
            start = -1
            if first_brace != -1 and first_bracket != -1:
                start = min(first_brace, first_bracket)
            elif first_brace != -1:
                start = first_brace
            elif first_bracket != -1:
                start = first_bracket
            if start != -1:
                extracted = json_part[start:].strip()
                if extracted:
                    return extracted
            # Если после маркера пусто (например, модель вывела ```json ... ``` до «ОТВЕТ:») — ищем по всему тексту
            if json_part.strip():
                return json_part
    
    # 2. Если нет маркера "Ответ:" или после него ничего нет, ищем последний markdown блок ```json ... ```
    json_blocks = list(re.finditer(r"```(?:json)?\s*(.*?)\s*```", response_text, flags=re.IGNORECASE | re.DOTALL))
    if json_blocks:
        # Берем последний блок
        last_block = json_blocks[-1]
        extracted = last_block.group(1).strip()
        if extracted:
            return extracted
    
    # 3. Ищем JSON объект или массив в тексте
    # Важно: используем ПЕРВЫЙ { или [, т.к. при вложенных структурах {"a": [...], "b": [...]}
    # последний [ указывает на середину объекта, что даёт невалидный фрагмент
    first_brace = response_text.find("{")
    first_bracket = response_text.find("[")
    start = -1
    if first_brace != -1 and first_bracket != -1:
        start = min(first_brace, first_bracket)
    elif first_brace != -1:
        start = first_brace
    elif first_bracket != -1:
        start = first_bracket
    if start != -1:
        extracted = response_text[start:].strip()
        if extracted:
            return extracted
    
    # 4. Иначе возвращаем весь текст (будет обработано в parse_json_safe)
    return response_text.strip()
