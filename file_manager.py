"""
Класс для работы с файлами: поиск, сохранение, создание директорий.
Используется для единообразной работы с файлами в разных местах кода.
"""
import os
import json
import glob
import re
from typing import List, Optional, Dict, Any
import pandas as pd


class FileManager:
    """
    Класс для работы с файлами: поиск, сохранение, создание директорий.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Инициализирует FileManager.
        
        Args:
            base_dir: базовая директория для работы (опционально)
        """
        self.base_dir = base_dir
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Санитизирует имя для использования в имени файла.
        Заменяет все недопустимые символы на подчеркивания.
        
        Args:
            name: исходное имя
            
        Returns:
            безопасное имя для файла
        """
        if not name:
            return "unknown"
        
        # Недопустимые символы для имен файлов в Windows и Linux: < > : " / \ | ? *
        # Также заменяем пробелы и другие проблемные символы
        invalid_chars = r'[<>:"/\\|?*\s]'
        sanitized = re.sub(invalid_chars, '_', name)
        # Удаляем множественные подчеркивания
        sanitized = re.sub(r'_+', '_', sanitized)
        # Удаляем подчеркивания в начале и конце
        sanitized = sanitized.strip('_')
        
        # Если после санитизации имя пустое, возвращаем "unknown"
        if not sanitized:
            return "unknown"
        
        return sanitized
    
    def ensure_directory(self, directory_path: str) -> None:
        """
        Создает директорию, если она не существует.
        
        Args:
            directory_path: путь к директории
        """
        if self.base_dir and not os.path.isabs(directory_path):
            directory_path = os.path.join(self.base_dir, directory_path)
        os.makedirs(directory_path, exist_ok=True)
    
    def build_path(self, *parts: str) -> str:
        """
        Строит путь из частей.
        
        Args:
            *parts: части пути
            
        Returns:
            объединенный путь
        """
        path = os.path.join(*parts)
        if self.base_dir and not os.path.isabs(path):
            path = os.path.join(self.base_dir, path)
        return path
    
    def get_basename(self, file_path: str) -> str:
        """
        Возвращает базовое имя файла из пути.
        
        Args:
            file_path: путь к файлу
            
        Returns:
            базовое имя файла
        """
        return os.path.basename(file_path)
    
    def get_dirname(self, file_path: str) -> str:
        """
        Возвращает директорию файла из пути.
        
        Args:
            file_path: путь к файлу
            
        Returns:
            директория файла
        """
        return os.path.dirname(file_path)
    
    def get_name_without_ext(self, file_path: str) -> str:
        """
        Возвращает имя файла без расширения.
        
        Args:
            file_path: путь к файлу
            
        Returns:
            имя файла без расширения
        """
        basename = self.get_basename(file_path)
        return os.path.splitext(basename)[0]
    
    def file_exists(self, file_path: str) -> bool:
        """
        Проверяет существование файла.
        
        Args:
            file_path: путь к файлу
            
        Returns:
            True, если файл существует, иначе False
        """
        if self.base_dir and not os.path.isabs(file_path):
            file_path = os.path.join(self.base_dir, file_path)
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    def find_files(self, pattern: str, directory: Optional[str] = None, recursive: bool = False) -> List[str]:
        """
        Находит файлы по паттерну.
        
        Args:
            pattern: паттерн поиска (например, "*.json", "metrics_*.json")
            directory: директория для поиска (если None, используется текущая или base_dir)
            recursive: искать рекурсивно
            
        Returns:
            список найденных файлов
        """
        if directory:
            search_pattern = self.build_path(directory, pattern)
        else:
            search_pattern = pattern
        
        if recursive:
            # Для рекурсивного поиска используем **
            if not pattern.startswith("**"):
                search_pattern = self.build_path(directory or ".", "**", pattern)
        
        files = glob.glob(search_pattern, recursive=recursive)
        # Сортируем по времени модификации (новые первыми)
        files.sort(key=os.path.getmtime, reverse=True)
        return files
    
    def save_csv(self, dataframe: pd.DataFrame, file_path: str, encoding: str = 'utf-8-sig', index: bool = False) -> None:
        """
        Сохраняет DataFrame в CSV файл.
        
        Args:
            dataframe: DataFrame для сохранения
            file_path: путь к файлу
            encoding: кодировка (по умолчанию 'utf-8-sig')
            index: сохранять ли индекс (по умолчанию False)
        """
        # Создаем директорию, если она не существует
        directory = self.get_dirname(file_path)
        if directory:
            self.ensure_directory(directory)
        
        dataframe.to_csv(file_path, index=index, encoding=encoding)
    
    def save_json(self, data: Dict[str, Any], file_path: str, encoding: str = 'utf-8', indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        Сохраняет данные в JSON файл.
        
        Args:
            data: данные для сохранения (словарь)
            file_path: путь к файлу
            encoding: кодировка (по умолчанию 'utf-8')
            indent: отступ для форматирования (по умолчанию 2)
            ensure_ascii: экранировать ли не-ASCII символы (по умолчанию False)
        """
        # Создаем директорию, если она не существует
        directory = self.get_dirname(file_path)
        if directory:
            self.ensure_directory(directory)
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    
    def load_json(self, file_path: str, encoding: str = 'utf-8') -> Optional[Dict[str, Any]]:
        """
        Загружает данные из JSON файла.
        
        Args:
            file_path: путь к файлу
            encoding: кодировка (по умолчанию 'utf-8')
            
        Returns:
            загруженные данные или None, если файл не найден или произошла ошибка
        """
        if not self.file_exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_text(self, content: str, file_path: str, encoding: str = 'utf-8') -> None:
        """
        Сохраняет текстовый контент в файл.
        
        Args:
            content: текст для сохранения
            file_path: путь к файлу
            encoding: кодировка (по умолчанию 'utf-8')
        """
        # Создаем директорию, если она не существует
        directory = self.get_dirname(file_path)
        if directory:
            self.ensure_directory(directory)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    def append_text(self, content: str, file_path: str, encoding: str = 'utf-8') -> None:
        """
        Добавляет текстовый контент в конец файла.
        
        Args:
            content: текст для добавления
            file_path: путь к файлу
            encoding: кодировка (по умолчанию 'utf-8')
        """
        # Создаем директорию, если она не существует
        directory = self.get_dirname(file_path)
        if directory:
            self.ensure_directory(directory)
        
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
    
    def save_evaluation_results(
        self,
        evaluation_result: Dict[str, Any],
        results: List[Dict[str, Any]],
        output_dir: str,
        timestamp: str
    ) -> Dict[str, str]:
        """
        Сохраняет все результаты оценки в файлы.
        Инкапсулирует всю логику сохранения: CSV, JSON метрики, raw метрики, ошибки качества, summary.
        
        Args:
            evaluation_result: словарь с результатами оценки
            results: список результатов для каждого текста
            output_dir: базовая директория для сохранения
            timestamp: временная метка
            
        Returns:
            словарь с путями к сохраненным файлам
        """
        import re
        import json
        import pandas as pd
        
        saved_files = {}
        
        # Определяем model_key
        model_key = evaluation_result.get("model_key")
        if model_key:
            model_key = FileManager.sanitize_filename(model_key)
            model_key = re.sub(r'_\d{8}(_\d{4})?$', '', model_key)
        else:
            model_name = evaluation_result.get("model_name", "")
            model_key = re.sub(r'_\d{8}(_\d{4})?$', '', model_name)
            model_key = FileManager.sanitize_filename(model_key)
            if "_" in model_key:
                parts = model_key.split("_")
                non_date_parts = [p for p in parts if not re.match(r'^\d{8}(_\d{4})?$', p)]
                if non_date_parts:
                    if len(non_date_parts) > 1 and len(non_date_parts[-1]) <= 3:
                        model_key = non_date_parts[-2] if len(non_date_parts[-2]) > 3 else non_date_parts[-1]
                    else:
                        model_key = non_date_parts[-1]
        
        model_key = re.sub(r'_\d{8}(_\d{4})?$', '', model_key)
        if not model_key:
            model_key = "unknown_model"
        
        # Определяем название подпапки для промпта
        multi_agent_mode = evaluation_result.get("multi_agent_mode")
        prompt_template_name = evaluation_result.get("prompt_template", "unknown")
        
        # Проверяем использование structured_output, outlines и guidance
        hyperparameters = evaluation_result.get("hyperparameters", {})
        structured_output = hyperparameters.get("structured_output", False)
        use_outlines = hyperparameters.get("use_outlines", False)
        use_guidance = hyperparameters.get("use_guidance", False)
        
        if multi_agent_mode:
            prompt_folder_name = FileManager.sanitize_filename(multi_agent_mode)
        else:
            prompt_folder_name = FileManager.sanitize_filename(prompt_template_name)
        
        # Добавляем информацию о режимах structured_output, outlines и guidance в название папки
        mode_suffixes = []
        if use_guidance:
            mode_suffixes.append("GUIDANCE")
        elif use_outlines:
            mode_suffixes.append("OUTLINES")
        elif structured_output:
            mode_suffixes.append("STRUCTURED")
        
        if mode_suffixes:
            prompt_folder_name = f"{prompt_folder_name}_{'_'.join(mode_suffixes)}"
        
        # Создаем структуру папок
        model_dir = self.build_path(output_dir, model_key)
        prompt_dir = self.build_path(model_dir, prompt_folder_name)
        self.ensure_directory(prompt_dir)
        
        model_name_for_file = FileManager.sanitize_filename(evaluation_result.get("model_name", "unknown"))
        
        # 1. Сохраняем CSV с результатами
        df_results = pd.DataFrame(results)
        csv_path = self.build_path(prompt_dir, f"results_{model_name_for_file}_{timestamp}.csv")
        self.save_csv(df_results, csv_path)
        saved_files["csv"] = csv_path
        print(f"💾 Детальные результаты сохранены: {csv_path}")
        
        # 2. Подготавливаем и сохраняем метрики JSON
        evaluation_result_for_json = {}
        
        # Базовые поля
        for key in ["timestamp", "model_name", "model_key", "interrupted", "total_samples"]:
            if key in evaluation_result:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        # Метрики парсинга и валидации
        for key in ["valid_json_count", "invalid_json_count", "parsing_error_rate", "parsing_errors_count", "validation_stats"]:
            if key in evaluation_result:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        # Информация о GPU и производительности (после validation_stats)
        for key in ["gpu_info", "gpu_memory_after_load_gb", "gpu_memory_during_inference_gb", 
                    "average_response_time_seconds", "api_model"]:
            if key in evaluation_result:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        # Метрики качества
        if "quality_metrics" in evaluation_result:
            evaluation_result_for_json["quality_metrics"] = evaluation_result["quality_metrics"]
        
        # Подготавливаем единый список ошибок: парсинг + извлечение веществ (массовая доля, прочее)
        quality_metrics_for_json = evaluation_result_for_json.get("quality_metrics")
        all_quality_errors = []
        if quality_metrics_for_json:
            for group in ["массовая доля", "прочее"]:
                if group in quality_metrics_for_json:
                    group_errors = quality_metrics_for_json[group].get("все_ошибки", [])
                    for error in group_errors:
                        if isinstance(error, dict):
                            all_quality_errors.append(error)
                        else:
                            all_quality_errors.append({"error": str(error)})
                    quality_metrics_for_json[group].pop("все_ошибки", None)
                    quality_metrics_for_json[group].pop("ошибки", None)
            quality_metrics_for_json.pop("ошибки", None)
        
        # Получаем parsing_errors из evaluation_result напрямую (до добавления остальных полей)
        parsing_errors_list = evaluation_result.get("parsing_errors", [])
        all_errors = parsing_errors_list + all_quality_errors
        
        # Группируем ошибки по текстам
        errors_by_text = {}
        for error in all_errors:
            if isinstance(error, dict):
                text_idx = error.get("text_index", 0)
                if text_idx not in errors_by_text:
                    errors_by_text[text_idx] = {
                        "text_index": text_idx,
                        "text": error.get("text", ""),
                        "response": error.get("response", ""),
                        "prompt": error.get("prompt", ""),
                        "errors": []
                    }
                if error.get("error"):
                    errors_by_text[text_idx]["errors"].append(error.get("error"))
                if error.get("errors"):
                    errors_by_text[text_idx]["errors"].extend(error.get("errors"))
                if error.get("text") and not errors_by_text[text_idx]["text"]:
                    errors_by_text[text_idx]["text"] = error.get("text")
                if error.get("response") and not errors_by_text[text_idx]["response"]:
                    errors_by_text[text_idx]["response"] = error.get("response")
                if error.get("prompt") and not errors_by_text[text_idx]["prompt"]:
                    errors_by_text[text_idx]["prompt"] = error.get("prompt")
            elif isinstance(error, str):
                # Обрабатываем строковые ошибки (например, из parsing_errors)
                # Для строковых ошибок используем text_index = 0, если не указан
                text_idx = 0
                if text_idx not in errors_by_text:
                    errors_by_text[text_idx] = {
                        "text_index": text_idx,
                        "text": "",
                        "response": "",
                        "prompt": "",
                        "errors": []
                    }
                errors_by_text[text_idx]["errors"].append(error)
        
        errors_list = [v for v in errors_by_text.values() if v.get("errors")]
        
        # Добавляем остальные поля (ошибки — в самом конце JSON)
        excluded_keys = {
            "raw_output_metrics",  # Сохраняется отдельно в raw_metrics.json
            "parsing_errors",  # Входит в объединённое поле "ошибки"
            "ошибки",
        }
        for key in evaluation_result:
            if key not in evaluation_result_for_json and key not in excluded_keys:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        if "parsing_errors" in evaluation_result_for_json:
            evaluation_result_for_json.pop("parsing_errors")
        evaluation_result_for_json["ошибки"] = errors_list
        
        metrics_path = self.build_path(prompt_dir, f"metrics_{model_name_for_file}_{timestamp}.json")
        self.save_json(evaluation_result_for_json, metrics_path)
        saved_files["metrics"] = metrics_path
        print(f"💾 Метрики сохранены: {metrics_path}")
        print(f"   📋 Сохраненные гиперпараметры: {list(evaluation_result.get('hyperparameters', {}).keys())}")
        
        # 3. Сохраняем raw метрики
        raw_output_metrics = evaluation_result.get("raw_output_metrics")
        if raw_output_metrics:
            raw_metrics_path = self.build_path(prompt_dir, f"raw_metrics_{model_name_for_file}_{timestamp}.json")
            raw_metrics_for_json = {}
            
            if "validation" in raw_output_metrics:
                validation_data = raw_output_metrics["validation"]
                raw_metrics_for_json["validation"] = {
                    "valid_count": validation_data.get("valid_count", 0),
                    "invalid_count": validation_data.get("invalid_count", 0),
                    "validation_rate": validation_data.get("validation_rate", 0.0),
                    "validation_errors": validation_data.get("all_validation_errors", []) if "all_validation_errors" in validation_data else validation_data.get("validation_errors", [])
                }
            
            for group in ["массовая доля", "прочее"]:
                if group in raw_output_metrics:
                    group_data = raw_output_metrics[group]
                    raw_metrics_for_json[group] = {
                        "accuracy": group_data.get("accuracy", 0.0),
                        "precision": group_data.get("precision", 0.0),
                        "recall": group_data.get("recall", 0.0),
                        "f1": group_data.get("f1", 0.0),
                        "tp": group_data.get("tp", 0),
                        "fp": group_data.get("fp", 0),
                        "fn": group_data.get("fn", 0),
                        "количество_сравнений": group_data.get("количество_сравнений", group_data.get("tp", 0) + group_data.get("fp", 0) + group_data.get("fn", 0)),
                        "ошибки": group_data.get("все_ошибки", []) if "все_ошибки" in group_data else group_data.get("ошибки", [])
                    }
            
            try:
                self.save_json(raw_metrics_for_json, raw_metrics_path)
                saved_files["raw_metrics"] = raw_metrics_path
                print(f"💾 Raw метрики сохранены: {raw_metrics_path}")
            except Exception as e:
                print(f"⚠️ Ошибка при сохранении raw метрик: {e}")
        else:
            print(f"⚠️ Raw метрики не найдены в evaluation_result, файл raw_metrics не будет создан")
        
        # 4. Обновляем summary файл
        summary_path = self.build_path(output_dir, "evaluation_summary.jsonl")
        self.append_text(json.dumps(evaluation_result, ensure_ascii=False) + '\n', summary_path)
        saved_files["summary"] = summary_path
        print(f"💾 Результат добавлен в общий файл: {summary_path}")
        
        # 5. Сохраняем ошибки качества
        quality_metrics = evaluation_result.get("quality_metrics")
        if quality_metrics:
            errors_path = self.build_path(prompt_dir, f"quality_errors_{model_name_for_file}_{timestamp}.txt")
            error_content = f"Ошибки качества для модели: {evaluation_result['model_name']}\n"
            error_content += f"Дата: {timestamp}\n"
            error_content += f"{'='*80}\n\n"
            
            for group_name, group_key in [("МАССОВАЯ ДОЛЯ", "массовая доля"), ("ПРОЧЕЕ", "прочее")]:
                group = quality_metrics.get(group_key, {})
                errors = group.get('все_ошибки', group.get('ошибки', []))
                error_content += f"ОШИБКИ КАЧЕСТВА: {group_name}\n"
                if errors:
                    error_content += f"Всего ошибок: {len(errors)}\n"
                    error_content += f"{'─'*80}\n"
                    for i, error in enumerate(errors, 1):
                        error_content += f"{i}. {error}\n"
                    error_content += f"\n"
                else:
                    error_content += f"Ошибок не обнаружено.\n\n"
            
            self.save_text(error_content, errors_path)
            saved_files["quality_errors"] = errors_path
            print(f"💾 Ошибки качества сохранены: {errors_path}")
        
        return saved_files
    
    def save_reevaluation_results(
        self,
        evaluation_result: Dict[str, Any],
        results_csv_path: str,
        df_results: pd.DataFrame,
        predictions: List[Dict[str, Any]],
        quality_metrics: Optional[Dict[str, Any]],
        raw_output_metrics: Optional[Dict[str, Any]],
        timestamp: str,
        model_name: str
    ) -> Dict[str, str]:
        """
        Сохраняет все результаты переоценки в файлы.
        Инкапсулирует всю логику сохранения для reevaluation.
        
        Args:
            evaluation_result: словарь с результатами переоценки
            results_csv_path: путь к исходному CSV файлу
            df_results: DataFrame с исходными результатами
            predictions: список предсказаний
            quality_metrics: метрики качества (опционально)
            raw_output_metrics: raw метрики (опционально)
            timestamp: временная метка
            model_name: имя модели
            
        Returns:
            словарь с путями к сохраненным файлам
        """
        import json
        import pandas as pd
        
        saved_files = {}
        
        # Используем папку исходного CSV файла
        csv_dir = self.get_dirname(os.path.abspath(results_csv_path))
        self.ensure_directory(csv_dir)
        
        csv_name_without_ext = self.get_name_without_ext(results_csv_path)
        model_name_for_file = FileManager.sanitize_filename(model_name)
        
        # 1. Сохраняем CSV с результатами
        results_for_csv = []
        for idx, row in df_results.iterrows():
            pred = predictions[idx] if idx < len(predictions) else {}
            result_row = {
                "text": row.get("text", ""),
                "json": row.get("json", ""),
                "json_parsed": json.dumps(pred, ensure_ascii=False) if pred else "",
                "is_valid": bool(pred and isinstance(pred, dict) and len(pred) > 0)
            }
            for col in ["raw_output", "raw_validation", "parsed_validation"]:
                if col in df_results.columns:
                    result_row[col] = row.get(col, "")
            results_for_csv.append(result_row)
        
        df_results_reevaluated = pd.DataFrame(results_for_csv)
        csv_path = self.build_path(csv_dir, f"{csv_name_without_ext}_reevaluated_{timestamp}.csv")
        self.save_csv(df_results_reevaluated, csv_path)
        saved_files["csv"] = csv_path
        print(f"💾 Детальные результаты сохранены: {csv_path}")
        
        # 2. Пытаемся найти исходный файл метрик
        metrics_file_pattern = "metrics_*.json"
        metrics_files = self.find_files(metrics_file_pattern, csv_dir)
        original_metrics_files = [f for f in metrics_files if "_reevaluated" not in f]
        
        if not original_metrics_files:
            parent_dir = self.get_dirname(csv_dir)
            if parent_dir and parent_dir != csv_dir:
                metrics_files = self.find_files(metrics_file_pattern, parent_dir, recursive=True)
                original_metrics_files = [f for f in metrics_files if "_reevaluated" not in f]
        
        # Извлекаем информацию из исходного файла метрик
        if original_metrics_files:
            original_metrics = self.load_json(original_metrics_files[-1])
            if original_metrics:
                model_key = original_metrics.get("model_key")
                if model_key:
                    evaluation_result["model_key"] = model_key
        
        # 3. Сохраняем метрики JSON
        if csv_name_without_ext.startswith("results_"):
            metrics_base_name = csv_name_without_ext.replace("results_", "metrics_", 1)
        else:
            metrics_base_name = f"metrics_{csv_name_without_ext}"
        
        metrics_path = self.build_path(csv_dir, f"{metrics_base_name}_reevaluated_{timestamp}.json")
        
        evaluation_result_for_json = {}
        for key in ["timestamp", "model_name", "model_key", "interrupted", "total_samples"]:
            if key in evaluation_result:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        for key in ["valid_json_count", "invalid_json_count", "parsing_error_rate", "parsing_errors_count", "validation_stats"]:
            if key in evaluation_result:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        # Информация о GPU и производительности (после validation_stats)
        for key in ["gpu_info", "gpu_memory_after_load_gb", "gpu_memory_during_inference_gb", 
                    "average_response_time_seconds", "api_model"]:
            if key in evaluation_result:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        if "quality_metrics" in evaluation_result:
            evaluation_result_for_json["quality_metrics"] = evaluation_result["quality_metrics"]
        
        # Единый список ошибок: парсинг + извлечение веществ (как в save_evaluation_results)
        quality_metrics_for_json = evaluation_result_for_json.get("quality_metrics")
        all_quality_errors = []
        if quality_metrics_for_json:
            for group in ["массовая доля", "прочее"]:
                if group in quality_metrics_for_json:
                    group_errors = quality_metrics_for_json[group].get("все_ошибки", [])
                    for error in group_errors:
                        if isinstance(error, dict):
                            all_quality_errors.append(error)
                        else:
                            all_quality_errors.append({"error": str(error)})
                    quality_metrics_for_json[group].pop("все_ошибки", None)
                    quality_metrics_for_json[group].pop("ошибки", None)
            quality_metrics_for_json.pop("ошибки", None)
        
        parsing_errors_list = evaluation_result.get("parsing_errors", [])
        all_errors = parsing_errors_list + all_quality_errors
        
        errors_by_text = {}
        for error in all_errors:
            if isinstance(error, dict):
                text_idx = error.get("text_index", 0)
                if text_idx not in errors_by_text:
                    errors_by_text[text_idx] = {
                        "text_index": text_idx,
                        "text": error.get("text", ""),
                        "response": error.get("response", ""),
                        "prompt": error.get("prompt", ""),
                        "errors": []
                    }
                if error.get("error"):
                    errors_by_text[text_idx]["errors"].append(error.get("error"))
                if error.get("errors"):
                    errors_by_text[text_idx]["errors"].extend(error.get("errors"))
                if error.get("text") and not errors_by_text[text_idx]["text"]:
                    errors_by_text[text_idx]["text"] = error.get("text")
                if error.get("response") and not errors_by_text[text_idx]["response"]:
                    errors_by_text[text_idx]["response"] = error.get("response")
                if error.get("prompt") and not errors_by_text[text_idx]["prompt"]:
                    errors_by_text[text_idx]["prompt"] = error.get("prompt")
            elif isinstance(error, str):
                text_idx = 0
                if text_idx not in errors_by_text:
                    errors_by_text[text_idx] = {
                        "text_index": text_idx,
                        "text": "",
                        "response": "",
                        "prompt": "",
                        "errors": []
                    }
                errors_by_text[text_idx]["errors"].append(error)
        
        errors_list = [v for v in errors_by_text.values() if v.get("errors")]
        
        excluded_keys = {"raw_output_metrics", "parsing_errors", "ошибки"}
        for key in evaluation_result:
            if key not in evaluation_result_for_json and key not in excluded_keys:
                evaluation_result_for_json[key] = evaluation_result[key]
        
        if "parsing_errors" in evaluation_result_for_json:
            evaluation_result_for_json.pop("parsing_errors")
        evaluation_result_for_json["ошибки"] = errors_list
        
        self.save_json(evaluation_result_for_json, metrics_path)
        saved_files["metrics"] = metrics_path
        print(f"💾 Обновленные метрики сохранены: {metrics_path}")
        
        # 4. Сохраняем raw метрики
        if raw_output_metrics:
            if csv_name_without_ext.startswith("results_"):
                raw_metrics_base_name = csv_name_without_ext.replace("results_", "raw_metrics_", 1)
            else:
                raw_metrics_base_name = f"raw_metrics_{csv_name_without_ext}"
            
            raw_metrics_path = self.build_path(csv_dir, f"{raw_metrics_base_name}_reevaluated_{timestamp}.json")
            raw_metrics_for_json = {}
            
            if "validation" in raw_output_metrics:
                validation_data = raw_output_metrics["validation"]
                raw_metrics_for_json["validation"] = {
                    "valid_count": validation_data.get("valid_count", 0),
                    "invalid_count": validation_data.get("invalid_count", 0),
                    "validation_rate": validation_data.get("validation_rate", 0.0),
                    "validation_errors": validation_data.get("all_validation_errors", []) if "all_validation_errors" in validation_data else validation_data.get("validation_errors", [])
                }
            
            for group in ["массовая доля", "прочее"]:
                if group in raw_output_metrics:
                    group_data = raw_output_metrics[group]
                    raw_metrics_for_json[group] = {
                        "accuracy": group_data.get("accuracy", 0.0),
                        "precision": group_data.get("precision", 0.0),
                        "recall": group_data.get("recall", 0.0),
                        "f1": group_data.get("f1", 0.0),
                        "tp": group_data.get("tp", 0),
                        "fp": group_data.get("fp", 0),
                        "fn": group_data.get("fn", 0),
                        "количество_сравнений": group_data.get("количество_сравнений", group_data.get("tp", 0) + group_data.get("fp", 0) + group_data.get("fn", 0)),
                        "ошибки": group_data.get("все_ошибки", []) if "все_ошибки" in group_data else group_data.get("ошибки", [])
                    }
            
            self.save_json(raw_metrics_for_json, raw_metrics_path)
            saved_files["raw_metrics"] = raw_metrics_path
            print(f"💾 Raw метрики сохранены: {raw_metrics_path}")
        
        # 5. Сохраняем ошибки качества
        if quality_metrics:
            if csv_name_without_ext.startswith("results_"):
                errors_base_name = csv_name_without_ext.replace("results_", "quality_errors_", 1)
            else:
                errors_base_name = f"quality_errors_{csv_name_without_ext}"
            
            errors_path = self.build_path(csv_dir, f"{errors_base_name}_reevaluated_{timestamp}.txt")
            error_content = f"Ошибки качества для модели: {model_name}\n"
            error_content += f"Дата: {timestamp}\n"
            error_content += f"Переоценено из: {results_csv_path}\n"
            error_content += f"{'='*80}\n\n"
            
            for group_name, group_key in [("МАССОВАЯ ДОЛЯ", "массовая доля"), ("ПРОЧЕЕ", "прочее")]:
                group = quality_metrics.get(group_key, {})
                errors = group.get('все_ошибки', group.get('ошибки', []))
                error_content += f"ОШИБКИ КАЧЕСТВА: {group_name}\n"
                if errors:
                    error_content += f"Всего ошибок: {len(errors)}\n"
                    error_content += f"{'─'*80}\n"
                    for i, error in enumerate(errors, 1):
                        error_content += f"{i}. {error}\n"
                    error_content += f"\n"
                else:
                    error_content += f"Ошибок не обнаружено.\n\n"
            
            self.save_text(error_content, errors_path)
            saved_files["quality_errors"] = errors_path
            print(f"💾 Ошибки качества сохранены: {errors_path}")
        
        return saved_files