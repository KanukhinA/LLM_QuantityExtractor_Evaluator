"""
Модуль для интеграции результатов оценки моделей с Google Таблицами.

Извлекает F1 метрики из metrics.json файлов и загружает их в Google Таблицу.
Структура таблицы:
- По вертикали: alias моделей (model_key)
- По горизонтали: методы (название папки = prompt_template или multi_agent_mode)

Запуск без аргументов: python google_sheets_integration.py
  — обход папки results/, для каждой пары (модель, метод) берётся последний запуск,
  — загрузка в Google Таблицу (нужны google_sheets_credentials.json и
  GOOGLE_SHEETS_SPREADSHEET_ID в config.py или в переменной окружения).

Где взять JSON для Google API:
  console.cloud.google.com -> Ваш проект -> API и сервисы -> Учётные данные ->
  Создать учётные данные -> Сервисный аккаунт -> создать -> в карточке аккаунта
  вкладка "Ключи" -> Добавить ключ -> JSON -> скачается файл. Положите его
  в корень проекта как google_sheets_credentials.json (подробно в README).
"""
import os
import json
import glob
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

# Пауза (сек) между загрузками листов, чтобы не превысить лимит Write requests per minute (Sheets API)
SHEETS_UPLOAD_DELAY_SEC = 1.5

BASELINE_KEYWORD = "BASELINE"

# Таблица алиасов методов: (полное_название, alias, описание).
# Нумерация: 1–3 базовые, 4 minimal outlines, 5–10 CD-методы подряд, затем мультиагент и прочие.
# Если метода нет в таблице, alias = акроним из первых букв частей (разделитель _), описание = полное название.
METHOD_ALIAS_TABLE: List[Tuple[str, str, str]] = [
    ("DETAILED_INSTR_ZEROSHOT_BASELINE", "1.DIZB", "Детальный zero-shot промпт (baseline)"),
    ("DETAILED_INSTR_ONESHOT", "2.DIO", "Детальный one-shot промпт"),
    ("MINIMAL_INSTR_FIVESHOT", "3.MIF", "Минимальный инструктивный few-shot (5 примеров)"),
    ("MINIMAL_INSTR_FIVESHOT_OUTLINES", "4.MIFO", "Минимальный few-shot (5 примеров) с constrained decoding (outlines)"),
    ("DETAILED_INSTR_ZEROSHOT_CD_RUS_OUTLINES", "5.DIZCRO", "Zero-shot CD с кириллической схемой (outlines)"),
    ("DETAILED_INSTR_ONESHOT_CD_RUS_OUTLINES", "6.DIOCRO", "One-shot CD с кириллической схемой (outlines)"),
    ("DETAILED_INSTR_ZEROSHOT_CD_RUS_GUIDANCE", "7.DIZCRG", "Zero-shot CD с кириллической схемой (guidance)"),
    ("DETAILED_INSTR_ONESHOT_CD_RUS_GUIDANCE", "8.DIOCRG", "One-shot CD с кириллической схемой (guidance)"),
    ("MINIMAL_INSTR_FIVESHOT_CD_RUS_OUTLINES", "9.MIFCRO", "Минимальный few-shot CD с кириллической схемой (outlines)"),
    ("MINIMAL_INSTR_FIVESHOT_CD_RUS_GUIDANCE", "10.MIFCRG", "Минимальный few-shot CD с кириллической схемой (guidance)"),
    ("MA_SIMPLE_4AGENTS", "11.MS4", "Рабочий процесс \"Разделение обязанностей\" (4 агента)"),
    ("MA_CRITIC_3AGENTS", "12.MC3", "Рабочий процесс critic_3agents (3 агента: генератор, критик, исправитель)"),
    ("MINIMAL_INSTR_FIVESHOT_APIE", "13.MIFA", "Few-shot с 5 примерами (APIE)"),
]


def _acronym(name: str) -> str:
    """Акроним из первых букв частей названия (разделитель _)."""
    parts = [p.strip() for p in name.split("_") if p.strip()]
    return "".join(p[0].upper() for p in parts if p) if parts else name.upper()[:4]


def _method_alias_and_description(full_name: str) -> Tuple[str, str]:
    """Возвращает (alias, описание) для метода: из METHOD_ALIAS_TABLE или акроним и полное название."""
    for f, alias, desc in METHOD_ALIAS_TABLE:
        if f == full_name:
            return (alias, desc)
    return (_acronym(full_name), full_name)


def get_method_alias(full_name: str) -> str:
    """Алиас метода для заголовка таблицы (короткий)."""
    return _method_alias_and_description(full_name)[0]


def get_method_description(full_name: str) -> str:
    """Описание метода для блока примечаний."""
    return _method_alias_and_description(full_name)[1]


def _alias_order_number(alias: str) -> int:
    """Из алиаса вида 'N.XXX' извлекает N; иначе возвращает большое число (в конец сортировки)."""
    match = re.match(r"^(\d+)\.", alias)
    return int(match.group(1)) if match else 9999


def _method_sort_key(full_name: str) -> Tuple[int, str]:
    """Ключ сортировки метода: сначала по номеру в алиасе из METHOD_ALIAS_TABLE, затем по имени."""
    alias = get_method_alias(full_name)
    return (_alias_order_number(alias), full_name)


def _build_notes_rows(methods: List[str]) -> List[List[str]]:
    """Строки для блока примечаний под таблицей: заголовок и строки «ALIAS — определение»."""
    if not methods:
        return []
    rows = [["Примечания"]]
    for m in methods:
        alias = get_method_alias(m)
        desc = get_method_description(m)
        rows.append([f"{alias} — {desc}"])
    return rows


def _table_header(methods: List[str]) -> List[str]:
    """Заголовок таблицы: Модель + алиасы методов."""
    return ["Модель"] + [get_method_alias(m) for m in methods]

AVG_ROW_LABEL = "Средняя разница"


def _sort_methods_by_alias_number(methods) -> List[str]:
    """Список методов: сортировка по номеру в сокращении (METHOD_ALIAS_TABLE), затем по имени."""
    methods_list = sorted(set(methods), key=_method_sort_key)
    return methods_list


def _col_letter_1based(col_1based: int) -> str:
    """Буква(ы) столбца для 1-based индекса: 1=A, 2=B, ..., 26=Z, 27=AA."""
    s = ""
    n = col_1based
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _data_cell_a1(row_idx: int, col_idx: int) -> str:
    """A1-нотация ячейки: row_idx 0 = первая строка данных (вторая строка листа), col_idx 0 = первый столбец метрик (B)."""
    col_1based = col_idx + 2
    return f"{_col_letter_1based(col_1based)}{row_idx + 2}"


def _a1_range_to_grid_range(sheet_id: int, a1_range: str) -> dict:
    """Преобразует диапазон в A1-нотации (например 'A12:J12') в grid range для Sheets API (0-based индексы)."""
    part = a1_range.replace("$", "")
    if ":" in part:
        left, right = part.split(":", 1)
    else:
        left = right = part
    # Столбец: буквы -> 0-based индекс (A=0, ..., Z=25, AA=26, ...)
    def _col_letter_to_index(letters: str) -> int:
        letters = letters.upper().strip()
        idx = 0
        for c in letters:
            idx = idx * 26 + (ord(c) - ord("A") + 1)
        return idx - 1
    # Строка: цифры в конце
    def _row_from_cell(cell: str) -> int:
        i = 0
        while i < len(cell) and cell[i].isalpha():
            i += 1
        return int(cell[i:]) if cell[i:] else 1
    def _col_from_cell(cell: str) -> int:
        i = 0
        while i < len(cell) and cell[i].isalpha():
            i += 1
        return _col_letter_to_index(cell[:i])
    start_row = _row_from_cell(left)
    start_col = _col_from_cell(left)
    end_row = _row_from_cell(right)
    end_col = _col_from_cell(right)
    return {
        "sheetId": sheet_id,
        "startRowIndex": start_row - 1,
        "endRowIndex": end_row,
        "startColumnIndex": start_col,
        "endColumnIndex": end_col + 1,
    }


try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("⚠️ Библиотеки gspread и google-auth не установлены. Установите: pip install gspread google-auth")


class GoogleSheetsIntegration:
    """Класс для интеграции результатов с Google Таблицами"""
    
    def __init__(self, results_dir: str = "results", credentials_path: Optional[str] = None):
        """
        Args:
            results_dir: путь к директории с результатами
            credentials_path: путь к JSON файлу с credentials для Google API
        """
        self.results_dir = results_dir
        self.credentials_path = credentials_path
        self.client = None
        
        if GSPREAD_AVAILABLE and credentials_path:
            self._initialize_client()
    
    def _initialize_client(self):
        """Инициализирует клиент Google Sheets"""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Файл credentials не найден: {self.credentials_path}")
        
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file(self.credentials_path, scopes=scope)
            self.client = gspread.authorize(creds)
        except Exception as e:
            raise Exception(f"Ошибка инициализации Google Sheets клиента: {e}")

    def _ensure_client(self) -> None:
        """Проверяет, что клиент инициализирован; при возможности инициализирует лениво."""
        if self.client:
            return
        if not GSPREAD_AVAILABLE:
            raise Exception(
                "Библиотеки Google Sheets не установлены. Установите: pip install gspread google-auth"
            )
        if not self.credentials_path:
            raise Exception(
                "Google Sheets клиент не инициализирован: не указан путь к credentials JSON. "
                f"Положите {DEFAULT_CREDENTIALS_FILENAME} в корень проекта."
            )
        self._initialize_client()
    
    def find_metrics_files(self) -> List[str]:
        """
        Находит все metrics.json файлы в директории результатов

        Returns:
            список путей к metrics.json файлам
        """
        pattern = os.path.join(self.results_dir, "**", "metrics_*.json")
        metrics_files = glob.glob(pattern, recursive=True)
        return sorted(metrics_files)

    def _model_method_from_path(self, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Извлекает (model_key, method) из пути к metrics файлу без чтения файла.
        Ожидаемая структура: results_dir / model_key / method_folder / metrics_*.json
        """
        norm_path = os.path.normpath(os.path.abspath(file_path))
        norm_results = os.path.normpath(os.path.abspath(self.results_dir))
        if not norm_path.startswith(norm_results):
            return None
        prefix = norm_results.rstrip(os.sep) + os.sep
        rel = norm_path[len(prefix) :].split(os.sep) if norm_path.startswith(prefix) else []
        if len(rel) < 3:
            return None
        return (rel[0], rel[1])

    def get_latest_metrics_files(self) -> List[str]:
        """
        Для каждой пары (model_key, method) оставляет только один файл — с самой
        поздней датой изменения. Чтение содержимого не выполняется.

        Returns:
            список путей к последним metrics_*.json по каждой паре (модель, метод)
        """
        all_files = self.find_metrics_files()
        by_key: Dict[Tuple[str, str], List[Tuple[float, str]]] = defaultdict(list)
        for file_path in all_files:
            key = self._model_method_from_path(file_path)
            if not key:
                continue
            try:
                mtime = os.path.getmtime(file_path)
            except OSError:
                mtime = 0.0
            by_key[key].append((mtime, file_path))
        latest = []
        for key, items in by_key.items():
            items.sort(key=lambda x: x[0], reverse=True)
            latest.append(items[0][1])
        return sorted(latest)

    def parse_metrics_file(self, file_path: str) -> Optional[Dict]:
        """
        Парсит metrics.json файл и извлекает нужные данные
        
        Args:
            file_path: путь к metrics.json файлу
            
        Returns:
            словарь с данными или None, если файл невалидный
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_key = data.get("model_key")
            if not model_key:
                return None
            
            # Определяем метод из пути к файлу
            # Структура: results/{model_key}/{method_folder}/metrics_*.json
            path_parts = file_path.replace("\\", "/").split("/")
            if len(path_parts) < 3:
                return None
            
            # Находим индекс model_key в пути
            try:
                model_key_index = path_parts.index(model_key)
                if model_key_index + 1 < len(path_parts):
                    method_folder = path_parts[model_key_index + 1]
                else:
                    return None
            except ValueError:
                return None
            
            # Извлекаем F1 метрики
            quality_metrics = data.get("quality_metrics", {})
            f1_scores = {}
            for grp in ["массовая доля", "прочее"]:
                if grp in quality_metrics:
                    f1_scores[grp] = quality_metrics[grp].get("f1", None)

            # Формат validation: "0.99(0.88)" = parsed_validation_rate(raw_validation_rate)
            validation_str = None
            vs = data.get("validation_stats")
            if vs and isinstance(vs, dict):
                parsed_rate = vs.get("parsed", {}).get("validation_rate")
                raw_rate = vs.get("raw_output", {}).get("validation_rate")
                if parsed_rate is not None and raw_rate is not None:
                    validation_str = f"{parsed_rate:.2f}({raw_rate:.2f})"
            avg_inference_sec = data.get("average_response_time_seconds")
            if avg_inference_sec is not None and not isinstance(avg_inference_sec, (int, float)):
                avg_inference_sec = None
            gpu_mem_gb = data.get("gpu_memory_during_inference_gb")
            if gpu_mem_gb is not None and not isinstance(gpu_mem_gb, (int, float)):
                gpu_mem_gb = None
            
            return {
                "model_key": model_key,
                "method": method_folder,
                "f1_scores": f1_scores,
                "validation_str": validation_str,
                "average_response_time_seconds": avg_inference_sec,
                "gpu_memory_during_inference_gb": gpu_mem_gb,
                "file_path": file_path,
                "timestamp": data.get("timestamp")
            }
        except Exception as e:
            print(f"⚠️ Ошибка при парсинге {file_path}: {e}")
            return None
    
    def collect_all_metrics(
        self, latest_only: bool = True
    ) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, str]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Собирает метрики из metrics.json. Читает только последний файл по каждой
        паре (модель, метод) — по дате изменения файла (без разбора остальных).

        Args:
            latest_only: если True, для каждой пары (model_key, method) читается
                только один самый свежий metrics.json.

        Returns:
            (all_data, validation_data, inference_time_data, gpu_memory_data):
            - all_data: {model_key: {method: {group: f1_score}}}
            - validation_data: {model_key: {method: "0.99(0.88)"}} (parsed(raw))
            - inference_time_data: {model_key: {method: avg_seconds}}
            - gpu_memory_data: {model_key: {method: gpu_memory_during_inference_gb}}
        """
        if latest_only:
            metrics_files = self.get_latest_metrics_files()
        else:
            metrics_files = self.find_metrics_files()
        if not metrics_files:
            print(f"📊 В {self.results_dir} не найдено файлов metrics_*.json")
            return {}, {}, {}, {}

        all_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        validation_data: Dict[str, Dict[str, str]] = defaultdict(dict)
        inference_time_data: Dict[str, Dict[str, float]] = defaultdict(dict)
        gpu_memory_data: Dict[str, Dict[str, float]] = defaultdict(dict)
        for file_path in metrics_files:
            parsed = self.parse_metrics_file(file_path)
            if not parsed:
                continue
            model_key = parsed["model_key"]
            method = parsed["method"]
            for group, f1_score in parsed["f1_scores"].items():
                if f1_score is not None:
                    all_data[model_key][method][group] = f1_score
            if parsed.get("validation_str"):
                validation_data[model_key][method] = parsed["validation_str"]
            if parsed.get("average_response_time_seconds") is not None:
                inference_time_data[model_key][method] = float(parsed["average_response_time_seconds"])
            if parsed.get("gpu_memory_during_inference_gb") is not None:
                gpu_memory_data[model_key][method] = float(parsed["gpu_memory_during_inference_gb"])

        total = len(metrics_files)
        print(f"📊 Прочитано {total} файлов metrics.json (последний запуск по каждой паре модель+метод)")
        return dict(all_data), dict(validation_data), dict(inference_time_data), dict(gpu_memory_data)
    
    def create_table_data(
        self, group: str = "массовая доля", all_metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    ) -> Tuple[List[List], List[str], List[str], Dict[str, List[str]]]:
        """
        Создает данные для таблицы F1: значение и разница с baseline в скобках.
        Лучшее значение по столбцу выделяется жирным; + зелёный, - красный.

        Returns:
            (данные таблицы, список моделей, список методов, format_info: {"green", "red", "bold"} -> список A1 ячеек)
        """
        if all_metrics is None:
            all_metrics, _, _, _ = self.collect_all_metrics()
        
        models = sorted(set(all_metrics.keys()))
        methods_set = set()
        for model_data in all_metrics.values():
            methods_set.update(k for k in model_data.keys() if k != "_timestamp")
        methods = _sort_methods_by_alias_number(methods_set)
        baseline_method = next((m for m in methods if BASELINE_KEYWORD.upper() in m.upper()), methods[0] if methods else None)
        
        table_data = [_table_header(methods)]
        green_cells: List[str] = []
        red_cells: List[str] = []
        bold_cells: List[str] = []
        
        for row_idx, model in enumerate(models):
            row = [model]
            baseline_val = None
            if baseline_method:
                baseline_val = all_metrics.get(model, {}).get(baseline_method, {}).get(group)
            for col_idx, method in enumerate(methods):
                method_data = all_metrics.get(model, {}).get(method, {})
                f1_score = method_data.get(group) if group in method_data else None
                if f1_score is not None and (not isinstance(f1_score, (int, float)) or abs(float(f1_score)) >= 1e-6):
                    if method == baseline_method or baseline_val is None:
                        row.append(f"{f1_score:.4f}")
                    else:
                        diff = f1_score - baseline_val
                        sign = "+" if diff >= 0 else ""
                        cell_val = f"{f1_score:.4f}\n({sign}{diff:.2f})"
                        row.append(cell_val)
                        if cell_val.strip() != "-":
                            cell_a1 = _data_cell_a1(row_idx, col_idx)
                            if diff > 0:
                                green_cells.append(cell_a1)
                            elif diff < 0:
                                red_cells.append(cell_a1)
                else:
                    row.append("-")
            table_data.append(row)
        
        for col_idx in range(len(methods)):
            col_values = []
            for row_idx, model in enumerate(models):
                v = all_metrics.get(model, {}).get(methods[col_idx], {}).get(group)
                col_values.append((row_idx, v))
            valid = [(ri, v) for ri, v in col_values if v is not None and (not isinstance(v, (int, float)) or abs(float(v)) >= 1e-6)]
            if valid:
                best_row_idx = max(valid, key=lambda x: x[1])[0]
                bold_cells.append(_data_cell_a1(best_row_idx, col_idx))

        avg_row = [AVG_ROW_LABEL]
        for col_idx, method in enumerate(methods):
            diffs = []
            for model in models:
                baseline_val = all_metrics.get(model, {}).get(baseline_method, {}).get(group) if baseline_method else None
                f1_score = all_metrics.get(model, {}).get(method, {}).get(group)
                if (baseline_val is not None and (isinstance(baseline_val, (int, float)) and abs(float(baseline_val)) >= 1e-6)
                    and f1_score is not None and (isinstance(f1_score, (int, float)) and abs(float(f1_score)) >= 1e-6)
                    and method != baseline_method):
                    diffs.append(float(f1_score) - float(baseline_val))
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                sign = "+" if avg_diff >= 0 else ""
                avg_row.append(f"{sign}{avg_diff:.2f}")
            else:
                avg_row.append("-")
        table_data.append(avg_row)

        format_info = {"green": green_cells, "red": red_cells, "bold": bold_cells}
        return table_data, models, methods, format_info

    def _apply_cell_format(self, worksheet, format_info: Dict[str, List[str]]):
        """Применяет к ячейкам зелёный/красный фон и жирный шрифт по спискам A1."""
        if not format_info:
            return
        green = format_info.get("green") or []
        red = format_info.get("red") or []
        bold = format_info.get("bold") or []
        if green:
            worksheet.format(green, {"backgroundColor": {"red": 0.7, "green": 1.0, "blue": 0.7}})
        if red:
            worksheet.format(red, {"backgroundColor": {"red": 1.0, "green": 0.7, "blue": 0.7}})
        if bold:
            worksheet.format(bold, {"textFormat": {"bold": True}})

    def _get_or_create_worksheet(self, spreadsheet, worksheet_name: str):
        """Возвращает лист по имени; создаёт с rows=100, cols=20, если не найден."""
        try:
            return spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=worksheet_name, rows=100, cols=20)

    def _upload_table_to_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        table_data: List[List],
        models: List[str],
        methods: List[str],
        format_info: Dict[str, List[str]],
        clear_existing: bool,
        success_prefix: str,
        extra_lines: Optional[List[str]] = None,
    ) -> None:
        """Общая загрузка таблицы на лист: данные, форматирование заголовка и ячеек, примечания, вывод в консоль."""
        spreadsheet = self.client.open_by_key(spreadsheet_id)
        worksheet = self._get_or_create_worksheet(spreadsheet, worksheet_name)
        if clear_existing:
            worksheet.clear()
        worksheet.update(values=table_data, range_name="A1")
        worksheet.format("A1:Z1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
        })
        num_cols = len(methods)
        if num_cols > 0 and len(table_data) > 1:
            metrics_range = f"B2:{_col_letter_1based(1 + num_cols)}{len(table_data)}"
            worksheet.format(metrics_range, {
                "horizontalAlignment": "CENTER",
                "backgroundColor": {"red": 1, "green": 1, "blue": 1},
                "textFormat": {"bold": False},
            })
        self._apply_cell_format(worksheet, format_info)
        notes = _build_notes_rows(methods)
        if notes:
            start_row = len(table_data) + 1
            worksheet.update(values=notes, range_name=f"A{start_row}")
            num_table_cols = 1 + len(methods)
            last_col_letter = _col_letter_1based(num_table_cols)
            merge_requests = []
            for i in range(len(notes)):
                row = start_row + i
                merge_range = f"A{row}:{last_col_letter}{row}"
                grid = _a1_range_to_grid_range(worksheet.id, merge_range)
                merge_requests.append({"mergeCells": {"range": grid, "mergeType": "MERGE_ALL"}})
            if merge_requests:
                spreadsheet.batch_update({"requests": merge_requests})
        print(success_prefix)
        print(f"   • Моделей: {len(models)}")
        print(f"   • Методов: {len(methods)}")
        for line in extra_lines or []:
            print(line)

    def create_validation_table_data(
        self, validation_data: Dict[str, Dict[str, str]]
    ) -> Tuple[List[List], List[str], List[str], Dict[str, List[str]]]:
        """
        Создаёт данные для таблицы validation: ячейки в формате 0.99(0.88) = parsed(raw).
        Методы: сначала с BASELINE в названии. Пустое значение -> "-".
        Строка «Средняя разница»: средняя разница parsed rate с baseline (method - baseline), как в остальных таблицах.
        Ячейки с parsed rate == 1.0 выделяются зелёным.
        """
        def _parse_validation_rate(val: str) -> Optional[float]:
            if not val or (isinstance(val, str) and val.strip() in ("", "-")):
                return None
            match = re.match(r"^(\d+\.?\d*)", str(val).replace(",", "."))
            if not match:
                return None
            try:
                return float(match.group(1))
            except ValueError:
                return None

        models = sorted(set(validation_data.keys()))
        methods = _sort_methods_by_alias_number(set().union(*[set(validation_data[m].keys()) for m in validation_data]))
        baseline_method = next((m for m in methods if BASELINE_KEYWORD.upper() in m.upper()), methods[0] if methods else None)
        table_data = [_table_header(methods)]
        green_cells: List[str] = []
        for row_idx, model in enumerate(models):
            row = [model]
            for col_idx, method in enumerate(methods):
                val = validation_data.get(model, {}).get(method, "")
                cell_val = val if val else "-"
                row.append(cell_val)
                if cell_val and cell_val.strip() != "-":
                    parsed_rate = _parse_validation_rate(val)
                    if parsed_rate is not None and abs(parsed_rate - 1.0) < 1e-6:
                        green_cells.append(_data_cell_a1(row_idx, col_idx))
            table_data.append(row)
        avg_row = [AVG_ROW_LABEL]
        for method in methods:
            if method == baseline_method:
                avg_row.append("-")
                continue
            diffs = []
            for model in models:
                baseline_val = validation_data.get(model, {}).get(baseline_method, "") if baseline_method else ""
                method_val = validation_data.get(model, {}).get(method, "")
                baseline_rate = _parse_validation_rate(baseline_val)
                method_rate = _parse_validation_rate(method_val)
                if baseline_rate is not None and method_rate is not None:
                    diffs.append(method_rate - baseline_rate)
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                sign = "+" if avg_diff >= 0 else ""
                avg_row.append(f"{sign}{avg_diff:.2f}")
            else:
                avg_row.append("-")
        table_data.append(avg_row)
        format_info = {"green": green_cells, "red": [], "bold": []}
        return table_data, models, methods, format_info

    def create_inference_time_table_data(
        self, inference_time_data: Dict[str, Dict[str, float]]
    ) -> Tuple[List[List], List[str], List[str], Dict[str, List[str]]]:
        """
        Создаёт данные для таблицы среднего времени инференса: значение и разница с baseline.
        Меньше время = лучше: зелёный при отрицательной разнице, красный при положительной.
        Лучшее (мин) по столбцу — жирным.

        Returns:
            (данные таблицы, модели, методы, format_info)
        """
        models = sorted(set(inference_time_data.keys()))
        methods = _sort_methods_by_alias_number(set().union(*[set(inference_time_data[m].keys()) for m in inference_time_data]))
        baseline_method = next((m for m in methods if BASELINE_KEYWORD.upper() in m.upper()), methods[0] if methods else None)
        
        table_data = [_table_header(methods)]
        green_cells: List[str] = []
        red_cells: List[str] = []
        bold_cells: List[str] = []
        
        for row_idx, model in enumerate(models):
            row = [model]
            baseline_val = inference_time_data.get(model, {}).get(baseline_method) if baseline_method else None
            for col_idx, method in enumerate(methods):
                sec = inference_time_data.get(model, {}).get(method)
                if sec is not None and (not isinstance(sec, (int, float)) or abs(float(sec)) >= 1e-6):
                    if method == baseline_method or baseline_val is None:
                        row.append(f"{sec:.3f}")
                    else:
                        diff = sec - baseline_val
                        sign = "+" if diff >= 0 else ""
                        cell_val = f"{sec:.3f}\n({sign}{diff:.3f})"
                        row.append(cell_val)
                        if cell_val.strip() != "-":
                            cell_a1 = _data_cell_a1(row_idx, col_idx)
                            if diff < 0:
                                green_cells.append(cell_a1)
                            elif diff > 0:
                                red_cells.append(cell_a1)
                else:
                    row.append("-")
            table_data.append(row)
        
        for col_idx in range(len(methods)):
            col_values = [(row_idx, inference_time_data.get(models[row_idx], {}).get(methods[col_idx]))
                          for row_idx in range(len(models))]
            valid = [(ri, v) for ri, v in col_values if v is not None and (not isinstance(v, (int, float)) or abs(float(v)) >= 1e-6)]
            if valid:
                best_row_idx = min(valid, key=lambda x: x[1])[0]
                bold_cells.append(_data_cell_a1(best_row_idx, col_idx))

        avg_row = [AVG_ROW_LABEL]
        for col_idx, method in enumerate(methods):
            diffs = []
            for model in models:
                baseline_val = inference_time_data.get(model, {}).get(baseline_method) if baseline_method else None
                sec = inference_time_data.get(model, {}).get(method)
                if (baseline_val is not None and (isinstance(baseline_val, (int, float)) and abs(float(baseline_val)) >= 1e-6)
                    and sec is not None and (isinstance(sec, (int, float)) and abs(float(sec)) >= 1e-6)
                    and method != baseline_method):
                    diffs.append(float(sec) - float(baseline_val))
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                sign = "+" if avg_diff >= 0 else ""
                avg_row.append(f"{sign}{avg_diff:.3f}")
            else:
                avg_row.append("-")
        table_data.append(avg_row)

        format_info = {"green": green_cells, "red": red_cells, "bold": bold_cells}
        return table_data, models, methods, format_info

    def create_gpu_memory_table_data(
        self, gpu_memory_data: Dict[str, Dict[str, float]]
    ) -> Tuple[List[List], List[str], List[str], Dict[str, List[str]]]:
        """
        Создаёт данные для таблицы GPU memory during inference (GB).
        Разница = method - baseline (как в F1/Time): положительная = больше памяти, отрицательная = меньше.
        Нет значения -> "-". Меньше baseline -> зелёный, больше -> красный.
        Returns:
            (данные таблицы, модели, методы, format_info: green/red)
        """
        models = sorted(set(gpu_memory_data.keys()))
        methods = _sort_methods_by_alias_number(set().union(*[set(gpu_memory_data[m].keys()) for m in gpu_memory_data]))
        baseline_method = next((m for m in methods if BASELINE_KEYWORD.upper() in m.upper()), methods[0] if methods else None)
        table_data = [_table_header(methods)]
        green_cells: List[str] = []
        red_cells: List[str] = []

        def _is_empty_val(v) -> bool:
            if v is None:
                return True
            try:
                float(v)
                return False
            except (TypeError, ValueError):
                return True

        num_model_rows = len(models)
        for row_idx, model in enumerate(models):
            row = [model]
            baseline_val = None
            if baseline_method:
                bv = gpu_memory_data.get(model, {}).get(baseline_method)
                if not _is_empty_val(bv):
                    baseline_val = float(bv)
            for col_idx, method in enumerate(methods):
                gb_raw = gpu_memory_data.get(model, {}).get(method)
                if _is_empty_val(gb_raw):
                    row.append("-")
                else:
                    gb = float(gb_raw)
                    if method == baseline_method or baseline_val is None:
                        row.append(f"{gb:.2f}")
                    else:
                        diff = gb - baseline_val
                        sign = "+" if diff >= 0 else ""
                        cell_val = f"{gb:.2f}\n({sign}{diff:.2f})"
                        row.append(cell_val)
                        if cell_val.strip() != "-":
                            cell_a1 = _data_cell_a1(row_idx, col_idx)
                            if diff < 0:
                                green_cells.append(cell_a1)
                            elif diff > 0:
                                red_cells.append(cell_a1)
            table_data.append(row)

        avg_row = [AVG_ROW_LABEL]
        for col_idx, method in enumerate(methods):
            diffs = []
            for model in models:
                bv = gpu_memory_data.get(model, {}).get(baseline_method) if baseline_method else None
                gb_raw = gpu_memory_data.get(model, {}).get(method)
                if not _is_empty_val(bv) and not _is_empty_val(gb_raw) and method != baseline_method:
                    diffs.append(float(gb_raw) - float(bv))
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                sign = "+" if avg_diff >= 0 else ""
                cell_val = f"{sign}{avg_diff:.2f}"
                avg_row.append(cell_val)
                if cell_val.strip() != "-":
                    cell_avg = _data_cell_a1(num_model_rows, col_idx)
                    if avg_diff < 0:
                        green_cells.append(cell_avg)
                    elif avg_diff > 0:
                        red_cells.append(cell_avg)
            else:
                avg_row.append("-")
        table_data.append(avg_row)

        format_info = {"green": green_cells, "red": red_cells, "bold": []}
        return table_data, models, methods, format_info

    def upload_to_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        group: str = "массовая доля",
        clear_existing: bool = True,
        all_metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ):
        """
        Загружает данные F1 в Google Таблицу.

        Args:
            spreadsheet_id: ID Google Таблицы (из URL)
            worksheet_name: название листа
            group: группа метрик ("массовая доля" или "прочее")
            clear_existing: очищать ли существующие данные
            all_metrics: если передан, повторный сбор метрик не выполняется
        """
        self._ensure_client()
        table_data, models, methods, format_info = self.create_table_data(group=group, all_metrics=all_metrics)
        try:
            self._upload_table_to_sheet(
                spreadsheet_id, worksheet_name, table_data, models, methods, format_info,
                clear_existing,
                success_prefix=f"✅ Данные успешно загружены в лист '{worksheet_name}'",
                extra_lines=[f"   • Группа метрик: {group}"],
            )
        except Exception as e:
            raise Exception(f"Ошибка при загрузке данных в Google Таблицу: {e}")

    def upload_validation_to_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        validation_data: Dict[str, Dict[str, str]],
        clear_existing: bool = True,
    ):
        """
        Загружает таблицу validation (формат 0.99(0.88) = parsed(raw)) на лист.
        Ячейки с parsed rate == 1.0 выделяются зелёным.
        """
        self._ensure_client()
        table_data, models, methods, format_info = self.create_validation_table_data(validation_data)
        try:
            self._upload_table_to_sheet(
                spreadsheet_id, worksheet_name, table_data, models, methods, format_info,
                clear_existing,
                success_prefix=f"✅ Validation загружены в лист '{worksheet_name}'",
            )
        except Exception as e:
            raise Exception(f"Ошибка при загрузке validation в Google Таблицу: {e}")

    def upload_inference_time_to_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        inference_time_data: Dict[str, Dict[str, float]],
        clear_existing: bool = True,
    ):
        """Загружает таблицу среднего времени инференса (сек/ответ) на лист с разницей к baseline и форматированием."""
        self._ensure_client()
        table_data, models, methods, format_info = self.create_inference_time_table_data(inference_time_data)
        try:
            self._upload_table_to_sheet(
                spreadsheet_id, worksheet_name, table_data, models, methods, format_info,
                clear_existing,
                success_prefix=f"✅ Среднее время инференса загружено в лист '{worksheet_name}'",
            )
        except Exception as e:
            raise Exception(f"Ошибка при загрузке времени инференса в Google Таблицу: {e}")

    def upload_gpu_memory_to_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        gpu_memory_data: Dict[str, Dict[str, float]],
        clear_existing: bool = True,
    ):
        """Загружает таблицу GPU memory during inference (GB) на отдельный лист с окрашиванием относительно baseline."""
        self._ensure_client()
        table_data, models, methods, format_info = self.create_gpu_memory_table_data(gpu_memory_data)
        try:
            self._upload_table_to_sheet(
                spreadsheet_id, worksheet_name, table_data, models, methods, format_info,
                clear_existing,
                success_prefix=f"✅ GPU memory during inference загружено в лист '{worksheet_name}'",
            )
        except Exception as e:
            raise Exception(f"Ошибка при загрузке GPU memory в Google Таблицу: {e}")

    def export_to_csv(self, output_path: str, group: str = "массовая доля"):
        """
        Экспортирует данные в CSV файл
        
        Args:
            output_path: путь к выходному CSV файлу
            group: группа метрик ("массовая доля" или "прочее")
        """
        import csv
        
        table_data, models, methods, _ = self.create_table_data(group)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_data)
        
        print(f"✅ Данные экспортированы в CSV: {output_path}")
        print(f"   • Моделей: {len(models)}")
        print(f"   • Методов: {len(methods)}")
        print(f"   • Группа метрик: {group}")


# Имя файла по умолчанию для Google Service Account JSON (в корне проекта, в .gitignore)
DEFAULT_CREDENTIALS_FILENAME = "google_sheets_credentials.json"

# Листы по умолчанию при полном экспорте
DEFAULT_WORKSHEET_MASS = "F1 Scores"
DEFAULT_WORKSHEET_OTHER = "F1 Scores (прочее)"
DEFAULT_WORKSHEET_VALIDATION = "Validation (parsed(raw))"
DEFAULT_WORKSHEET_INFERENCE = "Avg inference time (s)"
DEFAULT_WORKSHEET_GPU_MEMORY = "GPU memory during inference (GB)"


def _default_results_dir() -> str:
    """Путь к результатам из config или текущая папка results."""
    try:
        from config import OUTPUT_DIR
        return OUTPUT_DIR
    except ImportError:
        return "results"


def _default_spreadsheet_id() -> Optional[str]:
    """ID таблицы из config_secrets или переменной окружения."""
    try:
        from config_secrets import GOOGLE_SHEETS_SPREADSHEET_ID
        if GOOGLE_SHEETS_SPREADSHEET_ID and str(GOOGLE_SHEETS_SPREADSHEET_ID).strip():
            return str(GOOGLE_SHEETS_SPREADSHEET_ID).strip()
    except (ImportError, AttributeError):
        pass
    return os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID")


def _default_credentials_path(script_dir: str) -> Optional[str]:
    """Путь к JSON credentials в корне проекта."""
    default_creds_path = os.path.join(script_dir, DEFAULT_CREDENTIALS_FILENAME)
    if os.path.exists(default_creds_path):
        return default_creds_path

    return None


def main():
    """
    Запуск без аргументов: обход results/, выбор последнего запуска по каждой паре
    (модель, метод), загрузка в Google Таблицу (если заданы credentials и spreadsheet id).
    """
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Экспорт метрик в Google Таблицу. Без аргументов: обход results, последние запуски по каждой паре (модель, метод), загрузка в таблицу."
    )
    parser.add_argument("--results-dir", type=str, default=None,
                        help=f"Папка с результатами (по умолчанию: из config.OUTPUT_DIR или results)")
    parser.add_argument("--spreadsheet-id", type=str, default=None,
                        help="ID Google Таблицы (из URL). Или GOOGLE_SHEETS_SPREADSHEET_ID в config/env.")
    parser.add_argument("--worksheet", type=str, default=None,
                        help=f"Название листа (по умолчанию: {DEFAULT_WORKSHEET_MASS}); при полном экспорте добавляется лист для «прочее»")
    parser.add_argument("--group", type=str, default=None,
                        choices=["массовая доля", "прочее"],
                        help="Экспортировать только эту группу; без флага — экспорт обеих групп на два листа")
    parser.add_argument("--export-csv", type=str, default=None,
                        help="Вместо таблицы экспортировать в CSV (укажите путь)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Только собрать и вывести метрики, не загружать в таблицу")

    args = parser.parse_args()

    results_dir = args.results_dir or _default_results_dir()
    credentials_path = _default_credentials_path(script_dir)
    spreadsheet_id = args.spreadsheet_id or _default_spreadsheet_id()

    integration = GoogleSheetsIntegration(
        results_dir=results_dir,
        credentials_path=credentials_path
    )

    if args.export_csv:
        group = args.group or "массовая доля"
        integration.export_to_csv(args.export_csv, group=group)
        return

    all_metrics, validation_data, inference_time_data, gpu_memory_data = integration.collect_all_metrics(latest_only=True)
    if not all_metrics and not validation_data and not inference_time_data and not gpu_memory_data:
        return

    def _print_metrics():
        print("\n📊 Собранные метрики (последний запуск по каждой паре модель+метод):")
        for model_key, methods in sorted(all_metrics.items()):
            print(f"\n  {model_key}:")
            for method, groups in sorted(methods.items()):
                print(f"    {method}: {groups}")
        if validation_data:
            print("\n📊 Validation (parsed(raw)):")
            for model_key, methods in sorted(validation_data.items()):
                print(f"\n  {model_key}:")
                for method, val in sorted(methods.items()):
                    print(f"    {method}: {val}")
        if inference_time_data:
            print("\n📊 Среднее время инференса (сек/ответ):")
            for model_key, methods in sorted(inference_time_data.items()):
                print(f"\n  {model_key}:")
                for method, sec in sorted(methods.items()):
                    print(f"    {method}: {sec:.3f}")
        if gpu_memory_data:
            print("\n📊 GPU memory during inference (GB):")
            for model_key, methods in sorted(gpu_memory_data.items()):
                print(f"\n  {model_key}:")
                for method, gb in sorted(methods.items()):
                    print(f"    {method}: {gb:.2f}")

    if args.no_upload:
        _print_metrics()
        return

    if not credentials_path:
        print(
            "⚠️ Не указан файл с credentials. "
            f"Положите {DEFAULT_CREDENTIALS_FILENAME} в корень проекта."
        )
        _print_metrics()
        return

    if not spreadsheet_id:
        print("⚠️ Не указан ID таблицы. Задайте GOOGLE_SHEETS_SPREADSHEET_ID в config_secrets.py или переменной окружения, либо передайте --spreadsheet-id.")
        _print_metrics()
        return

    worksheet_name = args.worksheet or DEFAULT_WORKSHEET_MASS
    if args.group:
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=worksheet_name,
            group=args.group,
            all_metrics=all_metrics,
        )
    else:
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=DEFAULT_WORKSHEET_MASS,
            group="массовая доля",
            all_metrics=all_metrics,
        )
        time.sleep(SHEETS_UPLOAD_DELAY_SEC)
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=DEFAULT_WORKSHEET_OTHER,
            group="прочее",
            all_metrics=all_metrics,
        )
        time.sleep(SHEETS_UPLOAD_DELAY_SEC)
        if validation_data:
            integration.upload_validation_to_sheet(
                spreadsheet_id=spreadsheet_id,
                worksheet_name=DEFAULT_WORKSHEET_VALIDATION,
                validation_data=validation_data,
            )
            time.sleep(SHEETS_UPLOAD_DELAY_SEC)
        if inference_time_data:
            integration.upload_inference_time_to_sheet(
                spreadsheet_id=spreadsheet_id,
                worksheet_name=DEFAULT_WORKSHEET_INFERENCE,
                inference_time_data=inference_time_data,
            )
            time.sleep(SHEETS_UPLOAD_DELAY_SEC)
        if gpu_memory_data:
            integration.upload_gpu_memory_to_sheet(
                spreadsheet_id=spreadsheet_id,
                worksheet_name=DEFAULT_WORKSHEET_GPU_MEMORY,
                gpu_memory_data=gpu_memory_data,
            )
        extra = []
        if validation_data:
            extra.append("«Validation (parsed(raw))»")
        if inference_time_data:
            extra.append("«Avg inference time (s)»")
        if gpu_memory_data:
            extra.append("«GPU memory during inference (GB)»")
        print("✅ Загружены листы: «массовая доля», «прочее»" + (", " + ", ".join(extra) + "." if extra else "."))


if __name__ == "__main__":
    main()
