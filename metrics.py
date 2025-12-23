"""
Метрики для оценки качества ответов моделей
"""
from typing import Dict, Any, List, Tuple
import json


def normalize_none_lists(value: Any) -> Any:
    """
    Нормализует списки, состоящие только из None, в None.
    [None, None] -> None
    [None] -> None
    None -> None
    """
    if value is None:
        return None
    if isinstance(value, list):
        # Если список состоит только из None (или пустой), возвращаем None
        if len(value) == 0:
            return None
        if all(v is None for v in value):
            return None
        # Иначе нормализуем элементы списка
        return [normalize_none_lists(v) for v in value]
    return value


def normalize_value(value: Any) -> Any:
    """
    Нормализует значение для сравнения (конвертирует числа, списки и т.д.)
    """
    # Сначала нормализуем None-списки
    value = normalize_none_lists(value)
    
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, str):
        # Пытаемся преобразовать строку в число
        try:
            return float(value)
        except:
            return value.lower().strip()
    return value


def compare_mass_dolya(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, int, List[str], Dict[str, float]]:
    """
    Сравнивает группу "массовая доля" между предсказанием и истинным значением.
    
    Returns:
        (score, total_items, errors, metrics): точность, общее количество элементов, список ошибок, метрики (precision, recall, f1)
    """
    pred_mass = predicted.get("массовая доля", [])
    true_mass = ground_truth.get("массовая доля", [])
    
    if not isinstance(pred_mass, list):
        pred_mass = []
    if not isinstance(true_mass, list):
        true_mass = []
    
    # Создаем словари для быстрого поиска по веществу
    pred_dict = {}
    for item in pred_mass:
        if isinstance(item, dict) and "вещество" in item:
            substance = item.get("вещество", "")
            pred_dict[substance] = normalize_value(item.get("массовая доля"))
    
    true_dict = {}
    for item in true_mass:
        if isinstance(item, dict) and "вещество" in item:
            substance = item.get("вещество", "")
            true_dict[substance] = normalize_value(item.get("массовая доля"))
    
    # Вычисляем метрики
    all_substances = set(list(pred_dict.keys()) + list(true_dict.keys()))
    tp = 0  # True Positive: правильно извлечено
    fp = 0  # False Positive: извлечено, но неправильно
    fn = 0  # False Negative: не извлечено, но должно было быть
    errors = []
    
    for substance in all_substances:
        pred_val = pred_dict.get(substance)
        true_val = true_dict.get(substance)
        
        # Нормализуем значения перед сравнением (чтобы [None, None] == None == [None])
        pred_val_normalized = normalize_none_lists(pred_val)
        true_val_normalized = normalize_none_lists(true_val)
        
        # Определяем, есть ли значение в предсказании и в истине
        pred_exists = pred_val is not None
        true_exists = true_val is not None
        
        if pred_val_normalized == true_val_normalized:
            if pred_exists and true_exists:
                tp += 1  # Правильно извлечено
        else:
            if pred_exists and true_exists:
                fp += 1  # Извлечено, но неправильно
                errors.append(f"Вещество {substance}: предсказано {pred_val}, истина {true_val}")
            elif pred_exists and not true_exists:
                fp += 1  # Извлечено, но не должно было быть
                errors.append(f"Вещество {substance}: предсказано {pred_val}, истина отсутствует")
            elif not pred_exists and true_exists:
                fn += 1  # Не извлечено, но должно было быть
                errors.append(f"Вещество {substance}: предсказано отсутствует, истина {true_val}")
    
    # Вычисляем метрики
    total = len(all_substances)
    score = tp / total if total > 0 else 0.0
    
    # Precision = TP / (TP + FP) - доля правильных среди всех извлеченных
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN) - доля правильных среди всех, что должны были быть извлечены
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
    
    return score, total, errors, metrics


def compare_prochee(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, int, List[str], Dict[str, float]]:
    """
    Сравнивает группу "прочее" между предсказанием и истинным значением.
    
    Returns:
        (score, total_items, errors, metrics): точность, общее количество элементов, список ошибок, метрики (precision, recall, f1)
    """
    pred_prochee = predicted.get("прочее", [])
    true_prochee = ground_truth.get("прочее", [])
    
    if not isinstance(pred_prochee, list):
        pred_prochee = []
    if not isinstance(true_prochee, list):
        true_prochee = []
    
    # Создаем словари для быстрого поиска по параметру
    pred_dict = {}
    for item in pred_prochee:
        if isinstance(item, dict) and "параметр" in item:
            param = item.get("параметр", "")
            # Извлекаем значение (может быть "масса", "количество", "объем", "значение")
            value = None
            for key in ["масса", "количество", "объем", "значение"]:
                if key in item:
                    value = normalize_value(item[key])
                    break
            if value is not None:
                pred_dict[param] = value
    
    true_dict = {}
    for item in true_prochee:
        if isinstance(item, dict) and "параметр" in item:
            param = item.get("параметр", "")
            value = None
            for key in ["масса", "количество", "объем", "значение"]:
                if key in item:
                    value = normalize_value(item[key])
                    break
            if value is not None:
                true_dict[param] = value
    
    # Вычисляем метрики
    all_params = set(list(pred_dict.keys()) + list(true_dict.keys()))
    tp = 0  # True Positive: правильно извлечено
    fp = 0  # False Positive: извлечено, но неправильно
    fn = 0  # False Negative: не извлечено, но должно было быть
    errors = []
    
    for param in all_params:
        pred_val = pred_dict.get(param)
        true_val = true_dict.get(param)
        
        # Нормализуем значения перед сравнением (чтобы [None, None] == None == [None])
        pred_val_normalized = normalize_none_lists(pred_val)
        true_val_normalized = normalize_none_lists(true_val)
        
        # Определяем, есть ли значение в предсказании и в истине
        pred_exists = pred_val is not None
        true_exists = true_val is not None
        
        if pred_val_normalized == true_val_normalized:
            if pred_exists and true_exists:
                tp += 1  # Правильно извлечено
        else:
            if pred_exists and true_exists:
                fp += 1  # Извлечено, но неправильно
                errors.append(f"Параметр {param}: предсказано {pred_val}, истина {true_val}")
            elif pred_exists and not true_exists:
                fp += 1  # Извлечено, но не должно было быть
                errors.append(f"Параметр {param}: предсказано {pred_val}, истина отсутствует")
            elif not pred_exists and true_exists:
                fn += 1  # Не извлечено, но должно было быть
                errors.append(f"Параметр {param}: предсказано отсутствует, истина {true_val}")
    
    # Вычисляем метрики
    total = len(all_params)
    score = tp / total if total > 0 else 0.0
    
    # Precision = TP / (TP + FP) - доля правильных среди всех извлеченных
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN) - доля правильных среди всех, что должны были быть извлечены
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
    
    return score, total, errors, metrics


def calculate_quality_metrics(predictions: List[Dict[str, Any]], 
                              ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Вычисляет метрики качества для всех предсказаний.
    
    Args:
        predictions: список распарсенных JSON предсказаний
        ground_truths: список истинных JSON значений
    
    Returns:
        словарь с метриками
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Количество предсказаний ({len(predictions)}) и истинных значений ({len(ground_truths)}) должно совпадать")
    
    mass_dolya_scores = []
    prochee_scores = []
    all_mass_errors = []
    all_prochee_errors = []
    
    # Инициализация счетчиков для метрик
    mass_tp_total = 0
    mass_fp_total = 0
    mass_fn_total = 0
    prochee_tp_total = 0
    prochee_fp_total = 0
    prochee_fn_total = 0
    
    for pred, true_val in zip(predictions, ground_truths):
        # Массовая доля
        score_mass, total_mass, errors_mass, metrics_mass = compare_mass_dolya(pred, true_val)
        if total_mass > 0:
            mass_dolya_scores.append(score_mass)
            all_mass_errors.extend(errors_mass)
            mass_tp_total += metrics_mass["tp"]
            mass_fp_total += metrics_mass["fp"]
            mass_fn_total += metrics_mass["fn"]
        
        # Прочее
        score_prochee, total_prochee, errors_prochee, metrics_prochee = compare_prochee(pred, true_val)
        if total_prochee > 0:
            prochee_scores.append(score_prochee)
            all_prochee_errors.extend(errors_prochee)
            prochee_tp_total += metrics_prochee["tp"]
            prochee_fp_total += metrics_prochee["fp"]
            prochee_fn_total += metrics_prochee["fn"]
    
    avg_mass_dolya = sum(mass_dolya_scores) / len(mass_dolya_scores) if mass_dolya_scores else 0.0
    avg_prochee = sum(prochee_scores) / len(prochee_scores) if prochee_scores else 0.0
    
    # Вычисление Precision, Recall, F1 для массовой доли
    mass_precision = mass_tp_total / (mass_tp_total + mass_fp_total) if (mass_tp_total + mass_fp_total) > 0 else 0.0
    mass_recall = mass_tp_total / (mass_tp_total + mass_fn_total) if (mass_tp_total + mass_fn_total) > 0 else 0.0
    mass_f1 = 2 * (mass_precision * mass_recall) / (mass_precision + mass_recall) if (mass_precision + mass_recall) > 0 else 0.0
    
    # Вычисление Precision, Recall, F1 для прочее
    prochee_precision = prochee_tp_total / (prochee_tp_total + prochee_fp_total) if (prochee_tp_total + prochee_fp_total) > 0 else 0.0
    prochee_recall = prochee_tp_total / (prochee_tp_total + prochee_fn_total) if (prochee_tp_total + prochee_fn_total) > 0 else 0.0
    prochee_f1 = 2 * (prochee_precision * prochee_recall) / (prochee_precision + prochee_recall) if (prochee_precision + prochee_recall) > 0 else 0.0
    
    return {
        "массовая доля": {
            "средняя_точность": avg_mass_dolya,
            "precision": mass_precision,
            "recall": mass_recall,
            "f1": mass_f1,
            "tp": mass_tp_total,
            "fp": mass_fp_total,
            "fn": mass_fn_total,
            "количество_сравнений": len(mass_dolya_scores),
            "ошибки": all_mass_errors[:10],  # Первые 10 ошибок для примера (в JSON)
            "все_ошибки": all_mass_errors  # Все ошибки для сохранения в отдельный файл
        },
        "прочее": {
            "средняя_точность": avg_prochee,
            "precision": prochee_precision,
            "recall": prochee_recall,
            "f1": prochee_f1,
            "tp": prochee_tp_total,
            "fp": prochee_fp_total,
            "fn": prochee_fn_total,
            "количество_сравнений": len(prochee_scores),
            "ошибки": all_prochee_errors[:10],  # Первые 10 ошибок для примера (в JSON)
            "все_ошибки": all_prochee_errors  # Все ошибки для сохранения в отдельный файл
        }
    }

