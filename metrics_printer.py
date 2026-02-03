"""
Класс для печати метрик качества в консоль.
Используется для единообразного вывода метрик в разных местах кода.
"""
from typing import Dict, Any


class MetricsPrinter:
    """
    Класс для печати метрик качества в консоль.
    Используется для единообразного вывода метрик в разных местах кода.
    """
    
    @staticmethod
    def print_quality_metrics(quality_metrics: Dict[str, Any], prefix: str = "   ") -> None:
        """
        Выводит метрики качества для групп "массовая доля" и "прочее".
        
        Args:
            quality_metrics: словарь с метриками качества
            prefix: префикс для отступов (по умолчанию "   ")
        """
        if not quality_metrics or not isinstance(quality_metrics, dict):
            return
        
        mass_dolya = quality_metrics.get('массовая доля', {})
        prochee = quality_metrics.get('прочее', {})
        
        print(f"{prefix}✅ Метрики качества вычислены:")
        print(f"{prefix}📊 Группа 'массовая доля':")
        print(f"{prefix}   • Accuracy: {mass_dolya.get('accuracy', 0):.2%}")
        print(f"{prefix}   • Precision: {mass_dolya.get('precision', 0):.2%}")
        print(f"{prefix}   • Recall: {mass_dolya.get('recall', 0):.2%}")
        print(f"{prefix}   • F1-score: {mass_dolya.get('f1', 0):.2%}")
        print(f"{prefix}   • TP: {mass_dolya.get('tp', 0)}, FP: {mass_dolya.get('fp', 0)}, FN: {mass_dolya.get('fn', 0)}")
        print(f"{prefix}   • Количество сравнений: {mass_dolya.get('количество_сравнений', 0)}")
        print(f"{prefix}📊 Группа 'прочее':")
        print(f"{prefix}   • Accuracy: {prochee.get('accuracy', 0):.2%}")
        print(f"{prefix}   • Precision: {prochee.get('precision', 0):.2%}")
        print(f"{prefix}   • Recall: {prochee.get('recall', 0):.2%}")
        print(f"{prefix}   • F1-score: {prochee.get('f1', 0):.2%}")
        print(f"{prefix}   • TP: {prochee.get('tp', 0)}, FP: {prochee.get('fp', 0)}, FN: {prochee.get('fn', 0)}")
        print(f"{prefix}   • Количество сравнений: {prochee.get('количество_сравнений', 0)}")
    
    @staticmethod
    def print_raw_output_metrics(raw_output_metrics: Dict[str, Any], prefix: str = "   ") -> None:
        """
        Выводит метрики для raw output (валидация и качество).
        
        Args:
            raw_output_metrics: словарь с raw метриками
            prefix: префикс для отступов (по умолчанию "   ")
        """
        if not raw_output_metrics or not isinstance(raw_output_metrics, dict):
            return
        
        # Выводим метрики валидации для raw output
        if "validation" in raw_output_metrics:
            raw_val = raw_output_metrics["validation"]
            print(f"{prefix}📊 МЕТРИКИ ВАЛИДАЦИИ RAW OUTPUT:")
            total_count = raw_val.get('valid_count', 0) + raw_val.get('invalid_count', 0)
            if total_count == 0:
                total_count = raw_val.get('total_count', 0)
            print(f"{prefix}   • Валидных: {raw_val.get('valid_count', 0)}/{total_count} ({raw_val.get('validation_rate', 0):.2%})")
            print(f"{prefix}   • Невалидных: {raw_val.get('invalid_count', 0)}")
        
        # Выводим детальные метрики качества для raw output
        if "массовая доля" in raw_output_metrics:
            mass_dolya_raw = raw_output_metrics["массовая доля"]
            print(f"{prefix}📊 RAW МЕТРИКИ - Группа 'массовая доля':")
            print(f"{prefix}   • Accuracy: {mass_dolya_raw.get('accuracy', 0):.2%}")
            print(f"{prefix}   • Precision: {mass_dolya_raw.get('precision', 0):.2%}")
            print(f"{prefix}   • Recall: {mass_dolya_raw.get('recall', 0):.2%}")
            print(f"{prefix}   • F1-score: {mass_dolya_raw.get('f1', 0):.2%}")
            print(f"{prefix}   • TP: {mass_dolya_raw.get('tp', 0)}, FP: {mass_dolya_raw.get('fp', 0)}, FN: {mass_dolya_raw.get('fn', 0)}")
            print(f"{prefix}   • Количество сравнений: {mass_dolya_raw.get('количество_сравнений', 0)}")
        
        if "прочее" in raw_output_metrics:
            prochee_raw = raw_output_metrics["прочее"]
            print(f"{prefix}📊 RAW МЕТРИКИ - Группа 'прочее':")
            print(f"{prefix}   • Accuracy: {prochee_raw.get('accuracy', 0):.2%}")
            print(f"{prefix}   • Precision: {prochee_raw.get('precision', 0):.2%}")
            print(f"{prefix}   • Recall: {prochee_raw.get('recall', 0):.2%}")
            print(f"{prefix}   • F1-score: {prochee_raw.get('f1', 0):.2%}")
            print(f"{prefix}   • TP: {prochee_raw.get('tp', 0)}, FP: {prochee_raw.get('fp', 0)}, FN: {prochee_raw.get('fn', 0)}")
            print(f"{prefix}   • Количество сравнений: {prochee_raw.get('количество_сравнений', 0)}")
    
    @staticmethod
    def print_validation_stats(validation_stats: Dict[str, Any], prefix: str = "   ") -> None:
        """
        Выводит статистику валидации для cleaned output.
        
        Args:
            validation_stats: словарь со статистикой валидации
            prefix: префикс для отступов (по умолчанию "   ")
        """
        if not validation_stats or not isinstance(validation_stats, dict):
            return
        
        print(f"\n{prefix}📊 МЕТРИКИ ВАЛИДАЦИИ CLEANED OUTPUT:")
        raw_total = validation_stats.get('raw_output', {}).get('total_count', 0)
        parsed_total = validation_stats.get('parsed', {}).get('total_count', 0)
        
        raw_output_stats = validation_stats.get('raw_output', {})
        parsed_stats = validation_stats.get('parsed', {})
        
        print(f"{prefix}   • Raw output: валидных {raw_output_stats.get('valid_count', 0)}/{raw_total} ({raw_output_stats.get('validation_rate', 0):.2%})")
        print(f"{prefix}   • Parsed (после парсинга safe json): валидных {parsed_stats.get('valid_count', 0)}/{parsed_total} ({parsed_stats.get('validation_rate', 0):.2%})")
