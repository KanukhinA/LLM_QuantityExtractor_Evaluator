"""
Утилиты для парсинга JSON и построения промптов
"""
import json
import re
import ast
import codecs
from typing import Dict, Any
from prompt_config import FERTILIZER_EXTRACTION_PROMPT_WITH_EXAMPLE


def build_prompt3(text: str) -> str:
    """
    Генерация промпта для конкретного текста
    """
    return FERTILIZER_EXTRACTION_PROMPT_WITH_EXAMPLE.format(text=text)


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
            
            # Если нет markdown блоков, ищем последний JSON объект или массив
            # Ищем все вхождения открывающих фигурных скобок (для объектов)
            brace_positions = []
            bracket_positions = []  # Для массивов
            
            for i, char in enumerate(json_part):
                if char == '{':
                    brace_positions.append(i)
                elif char == '[':
                    bracket_positions.append(i)
            
            # Берем последний JSON (объект или массив)
            last_obj_idx = brace_positions[-1] if brace_positions else -1
            last_arr_idx = bracket_positions[-1] if bracket_positions else -1
            
            if last_obj_idx > last_arr_idx:
                # Последний JSON - объект
                extracted = json_part[last_obj_idx:].strip()
                if extracted:
                    return extracted
            elif last_arr_idx != -1:
                # Последний JSON - массив
                extracted = json_part[last_arr_idx:].strip()
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
    
    # 3. Ищем последний JSON объект или массив в тексте
    brace_positions = []  # Для объектов
    bracket_positions = []  # Для массивов
    
    for i, char in enumerate(response_text):
        if char == '{':
            brace_positions.append(i)
        elif char == '[':
            bracket_positions.append(i)
    
    # Берем последний JSON (объект или массив)
    last_obj_idx = brace_positions[-1] if brace_positions else -1
    last_arr_idx = bracket_positions[-1] if bracket_positions else -1
    
    if last_obj_idx > last_arr_idx:
        # Последний JSON - объект
        extracted = response_text[last_obj_idx:].strip()
        if extracted:
            return extracted
    elif last_arr_idx != -1:
        # Последний JSON - массив
        extracted = response_text[last_arr_idx:].strip()
        if extracted:
            return extracted
    
    # 4. Иначе возвращаем весь текст (будет обработано в parse_json_safe)
    return response_text.strip()
