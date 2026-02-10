"""
Модуль для интеграции результатов оценки моделей с Google Таблицами.

Извлекает F1 метрики из metrics.json файлов и загружает их в Google Таблицу.
Структура таблицы:
- По вертикали: alias моделей (model_key)
- По горизонтали: методы (название папки = prompt_template или multi_agent_mode)
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
    
    def collect_all_metrics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Собирает все метрики из metrics.json файлов
        
        Returns:
            словарь: {model_key: {method: {group: f1_score}}}
        """
        metrics_files = self.find_metrics_files()
        all_data = defaultdict(lambda: defaultdict(dict))
        
        print(f"📊 Найдено {len(metrics_files)} файлов metrics.json")
        
        for file_path in metrics_files:
            parsed = self.parse_metrics_file(file_path)
            if parsed:
                model_key = parsed["model_key"]
                method = parsed["method"]
                f1_scores = parsed["f1_scores"]
                
                for group, f1_score in f1_scores.items():
                    if f1_score is not None:
                        # Если для этого метода уже есть значение, берем более свежее (по timestamp)
                        existing_timestamp = all_data[model_key][method].get("_timestamp", "")
                        current_timestamp = parsed.get("timestamp", "")
                        if group not in all_data[model_key][method] or current_timestamp > existing_timestamp:
                            all_data[model_key][method][group] = f1_score
                            all_data[model_key][method]["_timestamp"] = current_timestamp
        
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


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Интеграция результатов с Google Таблицами')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Путь к директории с результатами')
    parser.add_argument('--credentials', type=str, default=None,
                       help='Путь к JSON файлу с credentials для Google API')
    parser.add_argument('--spreadsheet-id', type=str, default=None,
                       help='ID Google Таблицы (из URL)')
    parser.add_argument('--worksheet', type=str, default='F1 Scores',
                       help='Название листа в таблице')
    parser.add_argument('--group', type=str, default='массовая доля',
                       choices=['массовая доля', 'прочее'],
                       help='Группа метрик для экспорта')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='Экспортировать в CSV файл (укажите путь)')
    
    args = parser.parse_args()
    
    integration = GoogleSheetsIntegration(
        results_dir=args.results_dir,
        credentials_path=args.credentials
    )
    
    if args.export_csv:
        # Экспорт в CSV
        integration.export_to_csv(args.export_csv, group=args.group)
    elif args.spreadsheet_id and args.credentials:
        # Загрузка в Google Таблицу
        integration.upload_to_sheet(
            spreadsheet_id=args.spreadsheet_id,
            worksheet_name=args.worksheet,
            group=args.group
        )
    else:
        # Просто выводим собранные данные
        all_metrics = integration.collect_all_metrics()
        print(f"\n📊 Собранные метрики:")
        for model_key, methods in sorted(all_metrics.items()):
            print(f"\n{model_key}:")
            for method, groups in sorted(methods.items()):
                print(f"  {method}:")
                for group, f1_score in groups.items():
                    print(f"    {group}: {f1_score:.4f}")


if __name__ == "__main__":
    main()
