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
            for group in ["массовая доля", "прочее"]:
                if group in quality_metrics:
                    f1_scores[group] = quality_metrics[group].get("f1", None)
            
            return {
                "model_key": model_key,
                "method": method_folder,
                "f1_scores": f1_scores,
                "file_path": file_path,
                "timestamp": data.get("timestamp")
            }
        except Exception as e:
            print(f"⚠️ Ошибка при парсинге {file_path}: {e}")
            return None
    
    def collect_all_metrics(self, latest_only: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Собирает метрики из metrics.json. Для каждой пары (модель, метод) берётся
        один последний запуск (по timestamp в файле или по дате файла).

        Args:
            latest_only: если True, для каждой пары (model_key, method) берётся
                только самый свежий metrics.json; иначе собираются все и при конфликте — новее по timestamp.

        Returns:
            словарь: {model_key: {method: {group: f1_score}}}
        """
        metrics_files = self.find_metrics_files()
        if not metrics_files:
            print(f"📊 В {self.results_dir} не найдено файлов metrics_*.json")
            return {}

        # Группируем по (model_key, method), для каждой группы — список (timestamp_or_mtime, parsed)
        by_model_method: Dict[Tuple[str, str], List[Tuple[float, Dict]]] = defaultdict(list)

        for file_path in metrics_files:
            parsed = self.parse_metrics_file(file_path)
            if not parsed:
                continue
            model_key = parsed["model_key"]
            method = parsed["method"]
            ts_str = parsed.get("timestamp") or ""
            try:
                ts = float(os.path.getmtime(file_path))
            except OSError:
                ts = 0.0
            if ts_str:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts = dt.timestamp()
                except Exception:
                    pass
            by_model_method[(model_key, method)].append((ts, parsed))

        all_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        total_latest = 0
        for (model_key, method), items in by_model_method.items():
            items.sort(key=lambda x: x[0], reverse=True)
            _, latest_parsed = items[0]
            total_latest += 1
            for group, f1_score in latest_parsed["f1_scores"].items():
                if f1_score is not None:
                    all_data[model_key][method][group] = f1_score

        print(f"📊 Найдено {len(metrics_files)} файлов metrics.json, использовано {total_latest} последних запусков (модель+метод)")
        return dict(all_data)
    
    def create_table_data(self, group: str = "массовая доля") -> Tuple[List[List], List[str], List[str]]:
        """
        Создает данные для таблицы
        
        Args:
            group: группа метрик ("массовая доля" или "прочее")
            
        Returns:
            кортеж: (данные таблицы, список моделей, список методов)
        """
        all_metrics = self.collect_all_metrics()
        
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
    
    def upload_to_sheet(self, spreadsheet_id: str, worksheet_name: str, 
                       group: str = "массовая доля", clear_existing: bool = True):
        """
        Загружает данные в Google Таблицу
        
        Args:
            spreadsheet_id: ID Google Таблицы (из URL)
            worksheet_name: название листа
            group: группа метрик ("массовая доля" или "прочее")
            clear_existing: очищать ли существующие данные
        """
        if not self.client:
            raise Exception("Google Sheets клиент не инициализирован. Укажите credentials_path.")
        
        # Создаем данные таблицы
        table_data, models, methods = self.create_table_data(group)
        
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
            worksheet.update('A1', table_data)
            
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

# Листы по умолчанию для двух групп метрик при полном экспорте
DEFAULT_WORKSHEET_MASS = "F1 Scores"
DEFAULT_WORKSHEET_OTHER = "F1 Scores (прочее)"


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

    all_metrics = integration.collect_all_metrics(latest_only=True)
    if not all_metrics:
        return

    if args.no_upload:
        print("\n📊 Собранные метрики (последний запуск по каждой паре модель+метод):")
        for model_key, methods in sorted(all_metrics.items()):
            print(f"\n  {model_key}:")
            for method, groups in sorted(methods.items()):
                print(f"    {method}: {groups}")
        return

    if not credentials_path:
        print("⚠️ Не указан файл с credentials. Положите google_sheets_credentials.json в корень проекта или передайте --credentials.")
        print("\n📊 Собранные метрики (последний запуск по каждой паре модель+метод):")
        for model_key, methods in sorted(all_metrics.items()):
            print(f"\n  {model_key}:")
            for method, groups in sorted(methods.items()):
                print(f"    {method}: {groups}")
        return

    if not spreadsheet_id:
        print("⚠️ Не указан ID таблицы. Задайте GOOGLE_SHEETS_SPREADSHEET_ID в config_secrets.py или переменной окружения, либо передайте --spreadsheet-id.")
        print("\n📊 Собранные метрики (последний запуск по каждой паре модель+метод):")
        for model_key, methods in sorted(all_metrics.items()):
            print(f"\n  {model_key}:")
            for method, groups in sorted(methods.items()):
                print(f"    {method}: {groups}")
        return

    worksheet_name = args.worksheet or DEFAULT_WORKSHEET_MASS
    if args.group:
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=worksheet_name,
            group=args.group
        )
    else:
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=DEFAULT_WORKSHEET_MASS,
            group="массовая доля"
        )
        integration.upload_to_sheet(
            spreadsheet_id=spreadsheet_id,
            worksheet_name=DEFAULT_WORKSHEET_OTHER,
            group="прочее"
        )
        print("✅ Загружены оба листа: «массовая доля» и «прочее».")


if __name__ == "__main__":
    main()
