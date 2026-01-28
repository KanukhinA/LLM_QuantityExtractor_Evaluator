"""
Утилиты для парсинга JSON и построения промптов
"""
import json
import re
import ast
import codecs
import os
import inspect
import warnings
from typing import Dict, Any, Optional
import prompt_config


def load_api_keys():
    """
    Загружает API ключи с приоритетом:
    1. Из config_secrets.py (если файл существует)
    2. Из переменных окружения
    
    Returns:
        tuple: (HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY)
    """
    HF_TOKEN = None
    GEMINI_API_KEY = None
    OPENAI_API_KEY = None
    
    # Приоритет 1: Пытаемся загрузить из config_secrets.py
    try:
        from config_secrets import (
            HF_TOKEN as _hf_token, 
            GEMINI_API_KEY as _gemini_key,
            OPENAI_API_KEY as _openai_key
        )
        if _hf_token and _hf_token.strip():
            HF_TOKEN = _hf_token.strip()
        if _gemini_key and _gemini_key.strip():
            GEMINI_API_KEY = _gemini_key.strip()
        if _openai_key and _openai_key.strip():
            OPENAI_API_KEY = _openai_key.strip()
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
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            OPENAI_API_KEY = OPENAI_API_KEY.strip()
    
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
    
    # Нормализуем OPENAI_API_KEY (пустая строка = None)
    if OPENAI_API_KEY == "":
        OPENAI_API_KEY = None
    
    return HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY


# Загружаем ключи при импорте модуля
HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY = load_api_keys()


def find_file_path(file_path: str, search_in_parent: bool = True) -> str:
    """
    Ищет файл в различных возможных местах.
    
    Args:
        file_path: относительный путь к файлу (например, "data/udobrenia.xlsx")
        search_in_parent: если True, ищет в родительских директориях
    
    Returns:
        Абсолютный путь к файлу, если найден
    
    Raises:
        FileNotFoundError: если файл не найден
    """
    # Если путь абсолютный и файл существует, возвращаем его
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return os.path.abspath(file_path)
    
    # Если путь относительный и файл существует в текущей директории
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    
    if not search_in_parent:
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    # Пробуем несколько вариантов расположения файла
    config_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(config_dir)  # Родительская директория SmallLLMEvaluator
    cwd = os.getcwd()  # Текущая рабочая директория
    
    # Определяем директорию запуска скрипта
    script_dir = cwd
    try:
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
        os.path.join(base_dir, file_path),
        # Вариант 2: На уровень выше от родительской директории (../../data/)
        os.path.join(base_dir, "..", file_path),
        # Вариант 3: В директории запуска скрипта
        os.path.join(script_dir, file_path),
        # Вариант 4: На уровень выше от директории запуска скрипта
        os.path.join(script_dir, "..", file_path),
        # Вариант 5: Относительно текущей рабочей директории
        os.path.join(cwd, file_path),
        # Вариант 6: На уровень выше от текущей рабочей директории
        os.path.join(cwd, "..", file_path),
        # Вариант 7: В директории config (на случай, если data рядом с SmallLLMEvaluator)
        os.path.join(config_dir, "..", file_path),
        # Вариант 8: Относительный путь
        file_path
    ]
    
    # Проверяем каждый путь
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    # Если файл не найден, выводим все проверенные пути для отладки
    error_msg = f"Файл не найден: {file_path}\n"
    error_msg += "Проверены следующие пути:\n"
    for path in possible_paths:
        error_msg += f"  - {os.path.abspath(path)}\n"
    raise FileNotFoundError(error_msg)


def find_dataset_path(dataset_filename: str = None) -> str:
    """
    Ищет файл датасета в различных возможных местах.
    
    Args:
        dataset_filename: имя файла датасета (по умолчанию из config.DATASET_FILENAME)
    
    Returns:
        Абсолютный путь к файлу датасета
    """
    if dataset_filename is None:
        from config import DATASET_FILENAME
        dataset_filename = DATASET_FILENAME
    
    # Сначала проверяем переменную окружения (приоритет 1)
    if os.environ.get("DATASET_PATH"):
        return os.path.abspath(os.environ.get("DATASET_PATH"))
    
    # Используем общую функцию поиска
    return find_file_path(os.path.join("data", dataset_filename))


def build_prompt3(text: str, structured_output: bool = False, response_schema: Any = None) -> str:
    """
    Генерация промпта для конкретного текста
    
    Использует промпт, указанный в config.PROMPT_TEMPLATE_NAME (название переменной из prompt_config.py)
    Если structured_output=True, использует вариант промпта для structured output
    Если response_schema передан, добавляет JSON schema в промпт для локальных моделей
    """
    from config import PROMPT_TEMPLATE_NAME
    import json
    
    if structured_output:
        # Используем вариант промпта для structured output
        prompt_name = PROMPT_TEMPLATE_NAME.replace("_TEMPLATE", "_STRUCTURED").replace("_WITH_EXAMPLE", "_STRUCTURED")
        if not hasattr(prompt_config, prompt_name):
            # Если специального промпта нет, используем обычный
            prompt_name = PROMPT_TEMPLATE_NAME
        prompt_template = getattr(prompt_config, prompt_name)
    else:
        prompt_template = getattr(prompt_config, PROMPT_TEMPLATE_NAME)
    
    prompt_text = prompt_template.format(text=text)
    
    # Для локальных моделей добавляем JSON schema в промпт, если передан response_schema
    if structured_output and response_schema is not None:
        try:
            # Конвертируем Pydantic схему в JSON Schema
            if hasattr(response_schema, 'model_json_schema'):
                json_schema = response_schema.model_json_schema()
                # Добавляем JSON schema в конец промпта
                schema_text = json.dumps(json_schema, ensure_ascii=False, indent=2)
                prompt_text += f"\n\nТребуемая структура JSON (JSON Schema):\n```json\n{schema_text}\n```"
        except Exception as e:
            # Если не удалось добавить schema, просто используем промпт без него
            pass
    
    return prompt_text


def _find_last_valid_json(text: str) -> str:
    """
    Находит последний валидный JSON объект или массив в тексте.
    Учитывает баланс скобок для правильного определения границ JSON.
    
    Args:
        text: текст для поиска
        
    Returns:
        строка с последним JSON объектом/массивом или пустая строка
    """
    if not text:
        return ""
    
    # Ищем все возможные начала JSON объектов и массивов
    candidates = []
    
    # Ищем все открывающие скобки
    for i, char in enumerate(text):
        if char == '{':
            candidates.append(('object', i))
        elif char == '[':
            candidates.append(('array', i))
    
    if not candidates:
        return ""
    
    # Проверяем кандидатов с конца, чтобы найти последний валидный JSON
    for json_type, start_idx in reversed(candidates):
        # Извлекаем фрагмент от начала до конца текста
        fragment = text[start_idx:]
        
        # Пытаемся найти конец JSON объекта/массива, учитывая баланс скобок
        depth_obj = 0
        depth_arr = 0
        in_string = False
        escape = False
        
        end_idx = -1
        for i, ch in enumerate(fragment):
            # Обрабатываем escape-последовательности
            if ch == "\\" and not escape:
                escape = True
                continue
            elif escape:
                escape = False
                continue
            
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
                        # Для объекта проверяем только depth_obj (массив может быть внутри)
                        if json_type == 'object' and depth_obj == 0:
                            # Нашли конец объекта
                            end_idx = i + 1
                            break
                elif ch == "[":
                    depth_arr += 1
                elif ch == "]":
                    if depth_arr > 0:
                        depth_arr -= 1
                        # Для массива проверяем только depth_arr (объект может быть внутри)
                        if json_type == 'array' and depth_arr == 0:
                            # Нашли конец массива
                            end_idx = i + 1
                            break
        
        if end_idx != -1:
            # Нашли валидный JSON
            extracted = fragment[:end_idx].strip()
            if extracted:
                return extracted
        else:
            # JSON обрезан, но все равно возвращаем (будет обработано в parse_json_safe)
            extracted = fragment.strip()
            if extracted:
                return extracted
    
    return ""


def _extract_json_like(s: str) -> str:
    """
    Извлекает JSON-подстроку:
    1) fenced ```json ... ``` (берет последний блок, если их несколько)
    2) если нет — ищет последний '{' и возвращает оттуда до конца (фрагмент, возможно обрезанный)
    """
    if not isinstance(s, str):
        return ""

    # 1) fenced block ```json ... ``` - ищем все блоки и берем последний
    json_blocks = list(re.finditer(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.IGNORECASE | re.DOTALL))
    if json_blocks:
        # Берем последний блок
        last_block = json_blocks[-1]
        return last_block.group(1).strip()

    # убрать ведущие markdown-символы и пробелы
    s_stripped = re.sub(r"^[\s\*\-#>]+", "", s.lstrip())

    # 2) найти последнюю фигурную скобку и вернуть фрагмент от неё до конца (включая возможную обрезку)
    idx = s_stripped.rfind("{")
    if idx != -1:
        return s_stripped[idx:].strip()

    # если нет '{', пробуем последнюю '['
    idx = s_stripped.rfind("[")
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
            
            # Если нет markdown блоков, ищем последний валидный JSON объект или массив
            # Используем более умный подход: находим последний JSON, учитывая баланс скобок
            extracted = _find_last_valid_json(json_part)
            if extracted:
                return extracted
            
            # Если ничего не найдено, возвращаем как есть
            return json_part
    
    # 2. Если нет маркера "Ответ:", ищем последний markdown блок ```json ... ```
    json_blocks = list(re.finditer(r"```(?:json)?\s*(.*?)\s*```", response_text, flags=re.IGNORECASE | re.DOTALL))
    if json_blocks:
        # Берем последний блок
        last_block = json_blocks[-1]
        extracted = last_block.group(1).strip()
        if extracted:
            return extracted
    
    # 3. Ищем последний валидный JSON объект или массив в тексте
    extracted = _find_last_valid_json(response_text)
    if extracted:
        return extracted
    
    # 4. Иначе возвращаем весь текст (будет обработано в parse_json_safe)
    return response_text.strip()
