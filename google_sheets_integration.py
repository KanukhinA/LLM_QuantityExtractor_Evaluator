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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

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
            
            return {
                "model_key": model_key,
                "method": method_folder,
                "f1_scores": f1_scores,
                "validation_str": validation_str,
                "average_response_time_seconds": avg_inference_sec,
                "file_path": file_path,
                "timestamp": data.get("timestamp")
            }
        except Exception as e:
            print(f"⚠️ Ошибка при парсинге {file_path}: {e}")
            return None
    
    def collect_all_metrics(
        self, latest_only: bool = True
    ) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, str]], Dict[str, Dict[str, float]]]:
        """
        Собирает метрики из metrics.json. Читает только последний файл по каждой
        паре (модель, метод) — по дате изменения файла (без разбора остальных).

        Args:
            latest_only: если True, для каждой пары (model_key, method) читается
                только один самый свежий metrics.json.

        Returns:
            (all_data, validation_data, inference_time_data):
            - all_data: {model_key: {method: {group: f1_score}}}
            - validation_data: {model_key: {method: "0.99(0.88)"}} (parsed(raw))
            - inference_time_data: {model_key: {method: avg_seconds}}
        """
        if latest_only:
            metrics_files = self.get_latest_metrics_files()
        else:
            metrics_files = self.find_metrics_files()
        if not metrics_files:
            print(f"📊 В {self.results_dir} не найдено файлов metrics_*.json")
            return {}, {}, {}

        all_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        validation_data: Dict[str, Dict[str, str]] = defaultdict(dict)
        inference_time_data: Dict[str, Dict[str, float]] = defaultdict(dict)
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

        total = len(metrics_files)
        print(f"📊 Прочитано {total} файлов metrics.json (последний запуск по каждой паре модель+метод)")
        return dict(all_data), dict(validation_data), dict(inference_time_data)
    
    def create_table_data(
        self, group: str = "массовая доля", all_metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    ) -> Tuple[List[List], List[str], List[str]]:
        """
        Создает данные для таблицы F1.

        Args:
            group: группа метрик ("массовая доля" или "прочее")
            all_metrics: если передан, используется вместо повторного сбора

        Returns:
            кортеж: (данные таблицы, список моделей, список методов)
        """
        if all_metrics is None:
            all_metrics, _, _ = self.collect_all_metrics()
        
        # Собираем уникальные модели и методы
        models = sorted(set(all_metrics.keys()))
        methods = set()
        for model_data in all_metrics.values():
            # Пропускаем служебное поле _timestamp
            methods.update(k for k in model_data.keys() if k != "_timestamp")
        methods = sorted(methods)
        
        # Создаем таблицу
        table_data = []
        
        # Заголовок
        header = ["Модель"] + methods
        table_data.append(header)
        
        # Данные
        for model in models:
            row = [model]
            for method in methods:
                method_data = all_metrics.get(model, {}).get(method, {})
                # Пропускаем служебное поле _timestamp
                f1_score = method_data.get(group) if group in method_data else None
                if f1_score is not None:
                    row.append(f"{f1_score:.4f}")
                else:
                    row.append("")
            table_data.append(row)
        
        return table_data, models, methods

    def create_validation_table_data(
        self, validation_data: Dict[str, Dict[str, str]]
    ) -> Tuple[List[List], List[str], List[str]]:
        """
        Создаёт данные для таблицы validation: ячейки в формате 0.99(0.88) = parsed(raw).

        Args:
            validation_data: {model_key: {method: "0.99(0.88)"}}

        Returns:
            (данные таблицы, список моделей, список методов)
        """
        models = sorted(set(validation_data.keys()))
        methods = set()
        for per_model in validation_data.values():
            methods.update(per_model.keys())
        methods = sorted(methods)
        table_data = [["Модель"] + methods]
        for model in models:
            row = [model]
            for method in methods:
                val = validation_data.get(model, {}).get(method, "")
                row.append(val if val else "")
            table_data.append(row)
        return table_data, models, methods

    def create_inference_time_table_data(
        self, inference_time_data: Dict[str, Dict[str, float]]
    ) -> Tuple[List[List], List[str], List[str]]:
        """
        Создаёт данные для таблицы среднего времени инференса (сек/ответ).

        Args:
            inference_time_data: {model_key: {method: seconds}}

        Returns:
            (данные таблицы, список моделей, список методов)
        """
        models = sorted(set(inference_time_data.keys()))
        methods = set()
        for per_model in inference_time_data.values():
            methods.update(per_model.keys())
        methods = sorted(methods)
        table_data = [["Модель"] + methods]
        for model in models:
            row = [model]
            for method in methods:
                sec = inference_time_data.get(model, {}).get(method)
                if sec is not None:
                    row.append(f"{sec:.3f}")
                else:
                    row.append("")
            table_data.append(row)
        return table_data, models, methods

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
        if not self.client:
            raise Exception("Google Sheets клиент не инициализирован. Укажите credentials_path.")
        table_data, models, methods = self.create_table_data(group=group, all_metrics=all_metrics)
        
        try:
            # Открываем таблицу
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            
            # Получаем или создаем лист
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=100, cols=20)
            
            # Очищаем существующие данные, если нужно
            if clear_existing:
                worksheet.clear()
            
            # Загружаем данные
            worksheet.update(values=table_data, range_name="A1")
            
            # Форматируем заголовок
            worksheet.format('A1:Z1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            })
            
            print(f"✅ Данные успешно загружены в лист '{worksheet_name}'")
            print(f"   • Моделей: {len(models)}")
            print(f"   • Методов: {len(methods)}")
            print(f"   • Группа метрик: {group}")
            
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
        """
        if not self.client:
            raise Exception("Google Sheets клиент не инициализирован. Укажите credentials_path.")
        table_data, models, methods = self.create_validation_table_data(validation_data)
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=100, cols=20)
            if clear_existing:
                worksheet.clear()
            worksheet.update(values=table_data, range_name="A1")
            worksheet.format("A1:Z1", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })
            print(f"✅ Validation загружены в лист '{worksheet_name}' (моделей: {len(models)}, методов: {len(methods)})")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке validation в Google Таблицу: {e}")

    def upload_inference_time_to_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        inference_time_data: Dict[str, Dict[str, float]],
        clear_existing: bool = True,
    ):
        """Загружает таблицу среднего времени инференса (сек/ответ) на лист."""
        if not self.client:
            raise Exception("Google Sheets клиент не инициализирован. Укажите credentials_path.")
        table_data, models, methods = self.create_inference_time_table_data(inference_time_data)
        try:
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=100, cols=20)
            if clear_existing:
                worksheet.clear()
            worksheet.update(values=table_data, range_name="A1")
            worksheet.format("A1:Z1", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })
            print(f"✅ Среднее время инференса загружено в лист '{worksheet_name}' (моделей: {len(models)}, методов: {len(methods)})")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке времени инференса в Google Таблицу: {e}")

    def export_to_csv(self, output_path: str, group: str = "массовая доля"):
        """
        Экспортирует данные в CSV файл
        
        Args:
            output_path: путь к выходному CSV файлу
            group: группа метрик ("массовая доля" или "прочее")
        """
        import csv
        
        table_data, models, methods = self.create_table_data(group)
        
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


def main():
    """
    Запуск без аргументов: обход results/, выбор последнего запуска по каждой паре
    (модель, метод), загрузка в Google Таблицу (если заданы credentials и spreadsheet id).
    """
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_creds_path = os.path.join(script_dir, DEFAULT_CREDENTIALS_FILENAME)

    parser = argparse.ArgumentParser(
        description="Экспорт метрик в Google Таблицу. Без аргументов: обход results, последние запуски по каждой паре (модель, метод), загрузка в таблицу."
    )
    parser.add_argument("--results-dir", type=str, default=None,
                        help=f"Папка с результатами (по умолчанию: из config.OUTPUT_DIR или results)")
    parser.add_argument("--credentials", type=str, default=None,
                        help=f"Путь к JSON credentials (по умолчанию: {DEFAULT_CREDENTIALS_FILENAME} в корне проекта)")
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
    credentials_path = args.credentials
    if credentials_path is None and os.path.exists(default_creds_path):
        credentials_path = default_creds_path
    spreadsheet_id = args.spreadsheet_id or _default_spreadsheet_id()

    integration = GoogleSheetsIntegration(
        results_dir=results_dir,
        credentials_path=credentials_path
    )

    if args.export_csv:
        group = args.group or "массовая доля"
        integration.export_to_csv(args.export_csv, group=group)
        return

    all_metrics, validation_data, inference_time_data = integration.collect_all_metrics(latest_only=True)
    if not all_metrics and not validation_data and not inference_time_data:
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

    if args.no_upload:
        _print_metrics()
        return

    if not credentials_path:
        print("⚠️ Не указан файл с credentials. Положите google_sheets_credentials.json в корень проекта или передайте --credentials.")
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
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=DEFAULT_WORKSHEET_OTHER,
            group="прочее",
            all_metrics=all_metrics,
        )
        if validation_data:
            integration.upload_validation_to_sheet(
                spreadsheet_id=spreadsheet_id,
                worksheet_name=DEFAULT_WORKSHEET_VALIDATION,
                validation_data=validation_data,
            )
        if inference_time_data:
            integration.upload_inference_time_to_sheet(
                spreadsheet_id=spreadsheet_id,
                worksheet_name=DEFAULT_WORKSHEET_INFERENCE,
                inference_time_data=inference_time_data,
            )
        extra = []
        if validation_data:
            extra.append("«Validation (parsed(raw))»")
        if inference_time_data:
            extra.append("«Avg inference time (s)»")
        print("✅ Загружены листы: «массовая доля», «прочее»" + (", " + ", ".join(extra) + "." if extra else "."))


if __name__ == "__main__":
    main()
