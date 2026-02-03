"""
Метрики для оценки качества ответов моделей
"""
from typing import Dict, Any, List, Tuple, Optional
import json
from pydantic import ValidationError
from structured_schemas import FertilizerExtractionOutput


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


def normalize_value(value: Any, strict_mode: bool = False) -> Any:
    """
    Нормализует значение для сравнения (конвертирует числа, списки и т.д.)
    Обрабатывает проценты: "53%" -> 53.0, "53.5%" -> 53.5
    Конвертирует null/None в None, числа в float для корректного сравнения
    
    Args:
        strict_mode: если True, не делает допущений типа [None, None] = None или [x, x] = x
    """
    # Обрабатываем None/null (в Python null из JSON уже становится None)
    if value is None:
        return None
    
    # Нормализуем None-списки только если не в строгом режиме
    if not strict_mode:
        value = normalize_none_lists(value)
    
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        normalized_list = [normalize_value(v, strict_mode=strict_mode) for v in value]
        # Если список нормализовался в None (все элементы были None), возвращаем None
        # Только если не в строгом режиме
        if not strict_mode and all(v is None for v in normalized_list):
            return None
        return normalized_list
    if isinstance(value, str):
        # Убираем пробелы
        value = value.strip()
        
        # Обрабатываем проценты: "53%" -> 53.0, "53.5%" -> 53.5
        if value.endswith('%'):
            try:
                # Убираем символ % и преобразуем в число
                numeric_value = float(value[:-1].strip())
                return numeric_value
            except (ValueError, AttributeError):
                pass
        
        # Пытаемся преобразовать строку в число
        try:
            return float(value)
        except (ValueError, TypeError):
            return value.lower().strip()
    return value


def values_equal(val1: Any, val2: Any, strict_mode: bool = False) -> bool:
    """
    Сравнивает два нормализованных значения с учетом всех особенностей.
    Обрабатывает случаи с числами, списками, None и т.д.
    Если диапазон имеет равные начальное и конечное значения [x, x], он считается равным одиночному числу x.
    
    Args:
        strict_mode: если True, не делает допущений типа [x, x] = x
    """
    # Оба None
    if val1 is None and val2 is None:
        return True
    
    # Один None, другой нет
    if val1 is None or val2 is None:
        return False
    
    # Оба числа - сравниваем с небольшой погрешностью для float
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return abs(float(val1) - float(val2)) < 1e-9
    
    # Нормализуем диапазоны с равными значениями к одиночным числам только если не в строгом режиме
    # [3.5, 3.5] -> 3.5 для сравнения
    if not strict_mode:
        def normalize_range_to_single(value):
            """Если значение - список из двух одинаковых чисел, возвращает одиночное число"""
            if isinstance(value, list) and len(value) == 2:
                v1, v2 = value[0], value[1]
                # Если оба элемента - числа и они равны
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if abs(float(v1) - float(v2)) < 1e-9:
                        return float(v1)
            return value
        
        val1_normalized = normalize_range_to_single(val1)
        val2_normalized = normalize_range_to_single(val2)
    else:
        val1_normalized = val1
        val2_normalized = val2
    
    # После нормализации проверяем типы
    # Оба числа (после нормализации диапазонов)
    if isinstance(val1_normalized, (int, float)) and isinstance(val2_normalized, (int, float)):
        return abs(float(val1_normalized) - float(val2_normalized)) < 1e-9
    
    # Оба списка - сравниваем поэлементно
    if isinstance(val1_normalized, list) and isinstance(val2_normalized, list):
        if len(val1_normalized) != len(val2_normalized):
            return False
        return all(values_equal(v1, v2, strict_mode=strict_mode) for v1, v2 in zip(val1_normalized, val2_normalized))
    
    # Один список, другой число - в строгом режиме это всегда False
    # В нестрогом режиме это не должно происходить после нормализации
    if isinstance(val1_normalized, list) or isinstance(val2_normalized, list):
        return False
    
    # Остальные случаи - простое сравнение
    return val1_normalized == val2_normalized


def compare_mass_dolya(predicted: Dict[str, Any], ground_truth: Dict[str, Any], 
                       text_index: int = None, text: str = None, response: str = None,
                       strict_mode: bool = False) -> Tuple[float, int, List[Dict[str, Any]], Dict[str, float]]:
    """
    Сравнивает группу "массовая доля" между предсказанием и истинным значением.
    
    Args:
        predicted: предсказанный JSON
        ground_truth: истинный JSON
        text_index: индекс текста (для ошибок)
        text: исходный текст (для ошибок)
        response: ответ модели (для ошибок)
        strict_mode: если True, не делает допущений типа [None, None] = None или [x, x] = x
    
    Returns:
        (score, total_items, errors, metrics): точность, общее количество элементов, список ошибок (словари), метрики (precision, recall, f1)
    """
    # Обрабатываем случай, когда predicted - None или пустой словарь
    if not predicted:
        predicted = {}
    
    pred_mass = predicted.get("массовая доля", [])
    true_mass = ground_truth.get("массовая доля", [])
    
    if not isinstance(pred_mass, list):
        pred_mass = []
    if not isinstance(true_mass, list):
        true_mass = []
    
    # Нормализуем и фильтруем записи
    # Сохраняем оригинальные названия веществ для отображения в ошибках, но нормализуем для сравнения
    # ВАЖНО: Пропускаем записи с None значениями (None или [None, None]) - они не должны учитываться как FP/FN (только если не в строгом режиме
    pred_entries = []  # (substance_normalized, value_normalized, substance_original, value_original)
    for item in pred_mass:
        if isinstance(item, dict) and "вещество" in item:
            substance_original = item.get("вещество", "")
            # Нормализуем вещество к верхнему регистру для case-insensitive сравнения
            substance_normalized = substance_original.upper() if isinstance(substance_original, str) else substance_original
            value_original = item.get("массовая доля")
            value_normalized = normalize_value(value_original, strict_mode=strict_mode)
            # Пропускаем записи с None значениями (None или [None, None] после нормализации)
            # Только если не в строгом режиме
            if not strict_mode and value_normalized is not None:
                pred_entries.append((substance_normalized, value_normalized, substance_original, value_original))
            elif strict_mode:
                # В строгом режиме включаем все записи, даже с None
                pred_entries.append((substance_normalized, value_normalized, substance_original, value_original))
    
    true_entries = []  # (substance_normalized, value_normalized, substance_original, value_original)
    for item in true_mass:
        if isinstance(item, dict) and "вещество" in item:
            substance_original = item.get("вещество", "")
            # Нормализуем вещество к верхнему регистру для case-insensitive сравнения
            substance_normalized = substance_original.upper() if isinstance(substance_original, str) else substance_original
            value_original = item.get("массовая доля")
            value_normalized = normalize_value(value_original, strict_mode=strict_mode)
            # Пропускаем записи с None значениями (None или [None, None] после нормализации)
            # Только если не в строгом режиме
            if not strict_mode and value_normalized is not None:
                true_entries.append((substance_normalized, value_normalized, substance_original, value_original))
            elif strict_mode:
                # В строгом режиме включаем все записи, даже с None
                true_entries.append((substance_normalized, value_normalized, substance_original, value_original))
    
    # Сопоставляем записи (greedy matching)
    # Стратегия: для каждой записи из ground truth ищем лучшее совпадение среди всех предсказаний
    # Это позволяет правильно обрабатывать случаи, когда в предсказании несколько записей с одним веществом
    matched_pred_indices = set()
    matched_true_indices = set()
    tp = 0  # True Positive: правильно извлечено
    
    # Проходим по всем записям из ground truth и ищем совпадения в предсказаниях
    for true_idx, (true_subst_norm, true_val_norm, true_subst_orig, true_val_orig) in enumerate(true_entries):
        if true_idx in matched_true_indices:
            continue
        
        # Ищем совпадение в предсказаниях (только среди несопоставленных)
        best_match_idx = None
        for pred_idx, (pred_subst_norm, pred_val_norm, pred_subst_orig, pred_val_orig) in enumerate(pred_entries):
            if pred_idx in matched_pred_indices:
                continue
            
            # Проверяем совпадение вещества (используем нормализованные названия)
            if pred_subst_norm != true_subst_norm:
                continue
            
            # Проверяем совпадение значений (используем нормализованные значения с правильным сравнением)
            if values_equal(pred_val_norm, true_val_norm, strict_mode=strict_mode):
                # Нашли точное совпадение
                best_match_idx = pred_idx
                break
        
        if best_match_idx is not None:
            # Нашли совпадение
            matched_pred_indices.add(best_match_idx)
            matched_true_indices.add(true_idx)
            tp += 1
    
    # FP: записи из предсказания, которые не были сопоставлены
    fp = len(pred_entries) - len(matched_pred_indices)
    
    # FN: записи из ground truth, которые не были сопоставлены
    # НО: если для вещества есть хотя бы одно несопоставленное предсказание, это FP, а не FN
    # FN учитывается только для веществ, которые вообще не были предсказаны
    substances_with_predictions = set(pred_subst_norm for pred_idx, (pred_subst_norm, pred_val_norm, pred_subst_orig, pred_val_orig) in enumerate(pred_entries) 
                                     if pred_idx not in matched_pred_indices)
    
    fn = 0
    for true_idx, (true_subst_norm, true_val_norm, true_subst_orig, true_val_orig) in enumerate(true_entries):
        if true_idx not in matched_true_indices:
            # FN учитывается только если для этого вещества нет несопоставленных предсказаний
            # (т.е. модель вообще не пыталась извлечь это вещество)
            if true_subst_norm not in substances_with_predictions:
                fn += 1
    
    # Формируем список ошибок (в виде списка строк)
    error_messages = []
    
    # Создаем словарь для быстрого поиска ground truth по веществу (только несопоставленные)
    # Для каждого вещества храним список значений из ground truth (используем нормализованные названия для ключей)
    true_by_substance = {}  # {substance_normalized: [(true_val_orig, true_subst_orig), ...]}
    for true_idx, (true_subst_norm, true_val_norm, true_subst_orig, true_val_orig) in enumerate(true_entries):
        if true_idx not in matched_true_indices:
            if true_subst_norm not in true_by_substance:
                true_by_substance[true_subst_norm] = []
            true_by_substance[true_subst_norm].append((true_val_orig, true_subst_orig))
    
    # Создаем словарь несопоставленных предсказаний по веществу (используем нормализованные названия для ключей)
    fp_by_substance = {}  # {substance_normalized: [pred_val_orig, ...]}
    for pred_idx, (pred_subst_norm, pred_val_norm, pred_subst_orig, pred_val_orig) in enumerate(pred_entries):
        if pred_idx not in matched_pred_indices:
            if pred_subst_norm not in fp_by_substance:
                fp_by_substance[pred_subst_norm] = []
            fp_by_substance[pred_subst_norm].append(pred_val_orig)
    
    # FP ошибки: несопоставленные предсказания
    # Если для вещества есть ground truth, показываем его в ошибке
    for pred_idx, (pred_subst_norm, pred_val_norm, pred_subst_orig, pred_val_orig) in enumerate(pred_entries):
        if pred_idx not in matched_pred_indices:
            # Проверяем, есть ли для этого вещества несопоставленная запись в ground truth
            if pred_subst_norm in true_by_substance and true_by_substance[pred_subst_norm]:
                # Есть ground truth для этого вещества - показываем первое доступное значение
                # (одно и то же значение может быть показано для нескольких FP ошибок)
                true_val_orig, true_subst_orig = true_by_substance[pred_subst_norm][0]
                error_msg = f"Вещество {pred_subst_orig}: предсказано {pred_val_orig}, истина {true_val_orig}"
            else:
                # Нет ground truth для этого вещества
                error_msg = f"Вещество {pred_subst_orig}: предсказано {pred_val_orig}, истина отсутствует"
            
            error_messages.append(error_msg)
    
    # FN ошибки: оставшиеся несопоставленные ground truth
    # Показываем только те, для которых НЕТ несопоставленных предсказаний
    # (если есть несопоставленные предсказания, ground truth уже была показана в FP ошибках)
    for true_idx, (true_subst_norm, true_val_norm, true_subst_orig, true_val_orig) in enumerate(true_entries):
        if true_idx not in matched_true_indices:
            # Показываем FN только если для этого вещества нет несопоставленных предсказаний
            if true_subst_norm not in fp_by_substance:
                error_msg = f"Вещество {true_subst_orig}: предсказано отсутствует, истина {true_val_orig}"
                error_messages.append(error_msg)
    
    # Формируем результат в нужном формате: один словарь с полями text_index, text, response, errors
    errors = []
    if error_messages:
        error_dict = {
            "errors": error_messages
        }
        if text_index is not None:
            error_dict["text_index"] = text_index
        if text is not None:
            error_dict["text"] = text
        if response is not None:
            error_dict["response"] = response
        errors.append(error_dict)
    
    # Вычисляем метрики
    # total - это общее количество уникальных записей (TP + FP + FN)
    total = tp + fp + fn
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


def compare_prochee(predicted: Dict[str, Any], ground_truth: Dict[str, Any],
                   text_index: int = None, text: str = None, response: str = None,
                   strict_mode: bool = False) -> Tuple[float, int, List[Dict[str, Any]], Dict[str, float]]:
    """
    Сравнивает группу "прочее" между предсказанием и истинным значением.
    
    Args:
        predicted: предсказанный JSON
        ground_truth: истинный JSON
        text_index: индекс текста (для ошибок)
        text: исходный текст (для ошибок)
        response: ответ модели (для ошибок)
        strict_mode: если True, не делает допущений типа [None, None] = None или [x, x] = x
    
    Returns:
        (score, total_items, errors, metrics): точность, общее количество элементов, список ошибок (словари), метрики (precision, recall, f1)
    """
    # Обрабатываем случай, когда predicted - None или пустой словарь
    if not predicted:
        predicted = {}
    
    pred_prochee = predicted.get("прочее", [])
    true_prochee = ground_truth.get("прочее", [])
    
    if not isinstance(pred_prochee, list):
        pred_prochee = []
    if not isinstance(true_prochee, list):
        true_prochee = []
    
    # Нормализуем и фильтруем записи
    # Для "прочее" запись состоит из (параметр, значение, единица)
    # ВАЖНО: Пропускаем записи с None значениями (None или [None, None]) - они не должны учитываться как FP/FN
    # Но только если не в строгом режиме
    pred_entries = []
    for item in pred_prochee:
        if isinstance(item, dict) and "параметр" in item:
            param = item.get("параметр", "")
            # Извлекаем значение (может быть "масса", "количество", "объем", "значение")
            value = None
            for key in ["масса", "количество", "объем", "значение"]:
                if key in item:
                    value = normalize_value(item[key], strict_mode=strict_mode)
                    break
            # Извлекаем единицу, если есть
            unit = item.get("единица", None)
            # Пропускаем записи с None значениями (None или [None, None] после нормализации)
            # Только если не в строгом режиме
            if not strict_mode and value is not None:
                pred_entries.append((param, value, unit))
            elif strict_mode:
                # В строгом режиме включаем все записи, даже с None
                pred_entries.append((param, value, unit))
    
    true_entries = []
    for item in true_prochee:
        if isinstance(item, dict) and "параметр" in item:
            param = item.get("параметр", "")
            value = None
            for key in ["масса", "количество", "объем", "значение"]:
                if key in item:
                    value = normalize_value(item[key], strict_mode=strict_mode)
                    break
            # Извлекаем единицу, если есть
            unit = item.get("единица", None)
            # Пропускаем записи с None значениями (None или [None, None] после нормализации)
            # Только если не в строгом режиме
            if not strict_mode and value is not None:
                true_entries.append((param, value, unit))
            elif strict_mode:
                # В строгом режиме включаем все записи, даже с None
                true_entries.append((param, value, unit))
    
    # Сопоставляем записи (greedy matching)
    # Каждая запись - это (параметр, значение, единица)
    # Сопоставляем записи, где параметр, значение и единица совпадают
    matched_pred_indices = set()
    matched_true_indices = set()
    tp = 0  # True Positive: правильно извлечено
    
    # Проходим по всем предсказаниям и ищем совпадения в ground truth
    for pred_idx, (pred_param, pred_val, pred_unit) in enumerate(pred_entries):
        if pred_idx in matched_pred_indices:
            continue
        
        # Нормализуем значения для сравнения (только если не в строгом режиме)
        if not strict_mode:
            pred_val_norm = normalize_none_lists(pred_val)
        else:
            pred_val_norm = pred_val
        
        # Ищем совпадение в ground truth
        for true_idx, (true_param, true_val, true_unit) in enumerate(true_entries):
            if true_idx in matched_true_indices:
                continue
            
            # Нормализуем значения для сравнения (только если не в строгом режиме)
            if not strict_mode:
                true_val_norm = normalize_none_lists(true_val)
            else:
                true_val_norm = true_val
            
            # Проверяем совпадение: параметр, значение и единица должны совпадать
            # Используем values_equal для сравнения значений в строгом режиме
            if pred_param == true_param and values_equal(pred_val_norm, true_val_norm, strict_mode=strict_mode) and pred_unit == true_unit:
                # Нашли совпадение
                matched_pred_indices.add(pred_idx)
                matched_true_indices.add(true_idx)
                tp += 1
                break
    
    # FP: записи из предсказания, которые не были сопоставлены
    fp = len(pred_entries) - len(matched_pred_indices)
    
    # FN: записи из ground truth, которые не были сопоставлены
    # НО: если для параметра есть хотя бы одно несопоставленное предсказание, это FP, а не FN
    # FN учитывается только для параметров, которые вообще не были предсказаны
    params_with_predictions = set(pred_param for pred_idx, (pred_param, pred_val, pred_unit) in enumerate(pred_entries) 
                                  if pred_idx not in matched_pred_indices)
    
    fn = 0
    for true_idx, (true_param, true_val, true_unit) in enumerate(true_entries):
        if true_idx not in matched_true_indices:
            # FN учитывается только если для этого параметра нет несопоставленных предсказаний
            # (т.е. модель вообще не пыталась извлечь этот параметр)
            if true_param not in params_with_predictions:
                fn += 1
    
    # Формируем список ошибок (в виде списка строк)
    error_messages = []
    
    # Создаем словарь для быстрого поиска ground truth по параметру (только несопоставленные)
    true_by_param = {}  # {param: [(true_val, true_unit), ...]}
    for true_idx, (true_param, true_val, true_unit) in enumerate(true_entries):
        if true_idx not in matched_true_indices:
            if true_param not in true_by_param:
                true_by_param[true_param] = []
            true_by_param[true_param].append((true_val, true_unit))
    
    # Создаем словарь несопоставленных предсказаний по параметру
    fp_by_param = {}  # {param: [(pred_val, pred_unit), ...]}
    for pred_idx, (pred_param, pred_val, pred_unit) in enumerate(pred_entries):
        if pred_idx not in matched_pred_indices:
            if pred_param not in fp_by_param:
                fp_by_param[pred_param] = []
            fp_by_param[pred_param].append((pred_val, pred_unit))
    
    # FP ошибки: несопоставленные предсказания
    # Если для параметра есть ground truth, показываем его в ошибке
    for pred_idx, (pred_param, pred_val, pred_unit) in enumerate(pred_entries):
        if pred_idx not in matched_pred_indices:
            unit_str = f", единица: {pred_unit}" if pred_unit else ""
            # Проверяем, есть ли для этого параметра несопоставленная запись в ground truth
            if pred_param in true_by_param and true_by_param[pred_param]:
                # Есть ground truth для этого параметра - показываем первое доступное значение
                true_val, true_unit_gt = true_by_param[pred_param][0]
                true_unit_str = f", единица: {true_unit_gt}" if true_unit_gt else ""
                error_msg = f"Параметр {pred_param}: предсказано {pred_val}{unit_str}, истина {true_val}{true_unit_str}"
            else:
                # Нет ground truth для этого параметра
                error_msg = f"Параметр {pred_param}: предсказано {pred_val}{unit_str}, истина отсутствует"
            
            error_messages.append(error_msg)
    
    # FN ошибки: оставшиеся несопоставленные ground truth
    # Показываем только те, для которых НЕТ несопоставленных предсказаний
    for true_idx, (true_param, true_val, true_unit) in enumerate(true_entries):
        if true_idx not in matched_true_indices:
            # Показываем FN только если для этого параметра нет несопоставленных предсказаний
            if true_param not in fp_by_param:
                unit_str = f", единица: {true_unit}" if true_unit else ""
                error_msg = f"Параметр {true_param}: предсказано отсутствует, истина {true_val}{unit_str}"
                error_messages.append(error_msg)
    
    # Формируем результат в нужном формате: один словарь с полями text_index, text, response, errors
    errors = []
    if error_messages:
        error_dict = {
            "errors": error_messages
        }
        if text_index is not None:
            error_dict["text_index"] = text_index
        if text is not None:
            error_dict["text"] = text
        if response is not None:
            error_dict["response"] = response
        errors.append(error_dict)
    
    # Вычисляем метрики
    # total - это общее количество уникальных записей (TP + FP + FN)
    total = tp + fp + fn
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
                              ground_truths: List[Dict[str, Any]],
                              texts: List[str] = None,
                              responses: List[str] = None) -> Dict[str, Any]:
    """
    Вычисляет метрики качества для всех предсказаний.
    
    Args:
        predictions: список распарсенных JSON предсказаний
        ground_truths: список истинных JSON значений
        texts: список исходных текстов (для ошибок)
        responses: список ответов моделей (для ошибок)
    
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
    
    for idx, (pred, true_val) in enumerate(zip(predictions, ground_truths)):
        # Получаем текст и ответ для текущего индекса
        text = texts[idx] if texts and idx < len(texts) else None
        response = responses[idx] if responses and idx < len(responses) else None
        
        # Массовая доля
        score_mass, total_mass, errors_mass, metrics_mass = compare_mass_dolya(
            pred, true_val, text_index=idx, text=text, response=response
        )
        # Всегда добавляем метрики, даже если total = 0 (могут быть только FP или только FN)
        mass_tp_total += metrics_mass["tp"]
        mass_fp_total += metrics_mass["fp"]
        mass_fn_total += metrics_mass["fn"]
        if total_mass > 0:
            mass_dolya_scores.append(score_mass)
            # errors_mass - это список с одним словарем (или пустой), содержащим поле "errors"
            if errors_mass:
                all_mass_errors.extend(errors_mass)
        
        # Прочее
        score_prochee, total_prochee, errors_prochee, metrics_prochee = compare_prochee(
            pred, true_val, text_index=idx, text=text, response=response
        )
        # Всегда добавляем метрики, даже если total = 0 (могут быть только FP или только FN)
        prochee_tp_total += metrics_prochee["tp"]
        prochee_fp_total += metrics_prochee["fp"]
        prochee_fn_total += metrics_prochee["fn"]
        if total_prochee > 0:
            prochee_scores.append(score_prochee)
            # errors_prochee - это список с одним словарем (или пустой), содержащим поле "errors"
            if errors_prochee:
                all_prochee_errors.extend(errors_prochee)
    
    # Объединяем ошибки из mass и prochee для одного text_index
    # Создаем словарь для группировки ошибок по text_index
    errors_by_index = {}  # {text_index: {"text_index": ..., "text": ..., "response": ..., "errors": [...]}}
    
    # Обрабатываем ошибки массовой доли
    for error_dict in all_mass_errors:
        text_idx = error_dict.get("text_index")
        if text_idx is not None:
            if text_idx not in errors_by_index:
                errors_by_index[text_idx] = {
                    "text_index": text_idx,
                    "text": error_dict.get("text"),
                    "response": error_dict.get("response"),
                    "errors": []
                }
            # Добавляем ошибки из списка "errors"
            if "errors" in error_dict and error_dict["errors"]:
                errors_by_index[text_idx]["errors"].extend(error_dict["errors"])
    
    # Обрабатываем ошибки прочее
    for error_dict in all_prochee_errors:
        text_idx = error_dict.get("text_index")
        if text_idx is not None:
            if text_idx not in errors_by_index:
                errors_by_index[text_idx] = {
                    "text_index": text_idx,
                    "text": error_dict.get("text"),
                    "response": error_dict.get("response"),
                    "errors": []
                }
            # Добавляем ошибки из списка "errors"
            if "errors" in error_dict and error_dict["errors"]:
                errors_by_index[text_idx]["errors"].extend(error_dict["errors"])
    
    # Преобразуем словарь обратно в список, отсортированный по text_index
    # Включаем только те записи, где есть ошибки
    all_errors_combined = [errors_by_index[idx] for idx in sorted(errors_by_index.keys()) if errors_by_index[idx].get("errors")]
    
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
    
    # Вычисляем общее количество сравнений (total = tp + fp + fn)
    mass_total_comparisons = mass_tp_total + mass_fp_total + mass_fn_total
    prochee_total_comparisons = prochee_tp_total + prochee_fp_total + prochee_fn_total
    
    return {
        "массовая доля": {
            "accuracy": avg_mass_dolya,
            "precision": mass_precision,
            "recall": mass_recall,
            "f1": mass_f1,
            "tp": mass_tp_total,
            "fp": mass_fp_total,
            "fn": mass_fn_total,
            "количество_сравнений": mass_total_comparisons,  # Добавляем количество сравнений
            "все_ошибки": all_mass_errors  # Все ошибки для сохранения в структурированном виде в model_evaluator.py
        },
        "прочее": {
            "accuracy": avg_prochee,
            "precision": prochee_precision,
            "recall": prochee_recall,
            "f1": prochee_f1,
            "tp": prochee_tp_total,
            "fp": prochee_fp_total,
            "fn": prochee_fn_total,
            "количество_сравнений": prochee_total_comparisons,  # Добавляем количество сравнений
            "все_ошибки": all_prochee_errors  # Все ошибки для сохранения в структурированном виде в model_evaluator.py
        },
        "ошибки": all_errors_combined  # Объединенные ошибки из mass и prochee, сгруппированные по text_index
    }


def validate_with_pydantic(data: Any, stage: str = "parsed") -> Dict[str, Any]:
    """
    Валидирует данные через Pydantic схему.
    
    Args:
        data: данные для валидации (может быть dict, str или уже распарсенный объект)
        stage: этап валидации ("raw" для raw output, "parsed" для после парсинга)
    
    Returns:
        словарь с результатами валидации:
        {
            "is_valid": bool,
            "errors": List[str] - список ошибок валидации,
            "validated_data": Optional[Dict] - валидированные данные (если успешно)
        }
    """
    result = {
        "is_valid": False,
        "errors": [],
        "validated_data": None
    }
    
    # Если data - строка, пытаемся распарсить
    if isinstance(data, str):
        # Для raw output сначала пытаемся извлечь JSON
        if stage == "raw":
            try:
                # Пытаемся распарсить как JSON напрямую
                data = json.loads(data)
            except json.JSONDecodeError:
                # Если не удалось, пытаемся найти JSON в тексте
                try:
                    from utils import extract_json_from_response
                    json_str = extract_json_from_response(data)
                    if json_str:
                        data = json.loads(json_str)
                    else:
                        result["errors"].append("Не удалось извлечь JSON из raw output")
                        return result
                except Exception as e:
                    result["errors"].append(f"Ошибка извлечения JSON из raw output: {str(e)}")
                    return result
        else:
            # Для parsed stage просто парсим строку
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                result["errors"].append(f"JSON decode error: {str(e)}")
                return result
            except Exception as e:
                result["errors"].append(f"Parse error: {str(e)}")
                return result
    
    # Если data не dict, не можем валидировать
    if not isinstance(data, dict):
        result["errors"].append(f"Expected dict, got {type(data).__name__}")
        return result
    
    # Пытаемся валидировать через Pydantic
    try:
        validated = FertilizerExtractionOutput.model_validate(data)
        result["is_valid"] = True
        result["validated_data"] = validated.model_dump(by_alias=True)
    except ValidationError as e:
        result["errors"] = [str(err) for err in e.errors()]
    except Exception as e:
        result["errors"].append(f"Validation error: {str(e)}")
    
    return result


def calculate_raw_output_metrics(raw_outputs: List[str], 
                                 ground_truths: List[Dict[str, Any]],
                                 texts: List[str] = None,
                                 responses: List[str] = None) -> Dict[str, Any]:
    """
    Вычисляет метрики качества для raw output (без допущений, кроме регистра).
    Использует строгий парсинг JSON без умных исправлений.
    
    Args:
        raw_outputs: список сырых ответов моделей (response_text)
        ground_truths: список истинных JSON значений
        texts: список исходных текстов (для ошибок)
        responses: список ответов моделей (для ошибок) - может быть None, тогда используется raw_outputs
    
    Returns:
        словарь с метриками для raw output
    """
    if len(raw_outputs) != len(ground_truths):
        raise ValueError(f"Количество raw outputs ({len(raw_outputs)}) и истинных значений ({len(ground_truths)}) должно совпадать")
    
    # Парсим raw output строго (без умных исправлений)
    parsed_raw = []
    raw_validation_results = []
    
    for idx, raw_output in enumerate(raw_outputs):
        # Валидация raw output через Pydantic
        raw_validation = validate_with_pydantic(raw_output, stage="raw")
        raw_validation_results.append(raw_validation)
        
        # Строгий парсинг JSON (только json.loads, без умных исправлений)
        parsed = {}
        try:
            # Пытаемся распарсить raw output напрямую как JSON
            parsed = json.loads(raw_output)
            if not isinstance(parsed, dict):
                parsed = {}
        except (json.JSONDecodeError, Exception):
            # Если не удалось, пытаемся найти JSON объект в тексте (простой поиск)
            try:
                import re
                # Ищем JSON объект в тексте (простой паттерн)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    if not isinstance(parsed, dict):
                        parsed = {}
            except (json.JSONDecodeError, Exception):
                parsed = {}
        
        parsed_raw.append(parsed)
    
    # Вычисляем метрики на строго распарсенных данных
    # Используем те же функции сравнения, но без нормализации (кроме регистра)
    mass_dolya_scores = []
    prochee_scores = []
    all_mass_errors = []
    all_prochee_errors = []
    
    mass_tp_total = 0
    mass_fp_total = 0
    mass_fn_total = 0
    prochee_tp_total = 0
    prochee_fp_total = 0
    prochee_fn_total = 0
    
    for idx, (pred, true_val) in enumerate(zip(parsed_raw, ground_truths)):
        text = texts[idx] if texts and idx < len(texts) else None
        response = responses[idx] if responses and idx < len(responses) else raw_outputs[idx] if idx < len(raw_outputs) else None
        
        # Массовая доля (используем строгий режим для raw метрик)
        score_mass, total_mass, errors_mass, metrics_mass = compare_mass_dolya(
            pred, true_val, text_index=idx, text=text, response=response, strict_mode=True
        )
        # Всегда добавляем метрики, даже если total = 0 (могут быть только FP или только FN)
        mass_tp_total += metrics_mass["tp"]
        mass_fp_total += metrics_mass["fp"]
        mass_fn_total += metrics_mass["fn"]
        if total_mass > 0:
            mass_dolya_scores.append(score_mass)
            if errors_mass:
                all_mass_errors.extend(errors_mass)
        
        # Прочее (используем строгий режим для raw метрик)
        score_prochee, total_prochee, errors_prochee, metrics_prochee = compare_prochee(
            pred, true_val, text_index=idx, text=text, response=response, strict_mode=True
        )
        # Всегда добавляем метрики, даже если total = 0 (могут быть только FP или только FN)
        prochee_tp_total += metrics_prochee["tp"]
        prochee_fp_total += metrics_prochee["fp"]
        prochee_fn_total += metrics_prochee["fn"]
        if total_prochee > 0:
            prochee_scores.append(score_prochee)
            if errors_prochee:
                all_prochee_errors.extend(errors_prochee)
    
    # Вычисляем итоговые метрики
    mass_accuracy = sum(mass_dolya_scores) / len(mass_dolya_scores) if mass_dolya_scores else 0.0
    prochee_accuracy = sum(prochee_scores) / len(prochee_scores) if prochee_scores else 0.0
    
    mass_precision = mass_tp_total / (mass_tp_total + mass_fp_total) if (mass_tp_total + mass_fp_total) > 0 else 0.0
    mass_recall = mass_tp_total / (mass_tp_total + mass_fn_total) if (mass_tp_total + mass_fn_total) > 0 else 0.0
    mass_f1 = 2 * (mass_precision * mass_recall) / (mass_precision + mass_recall) if (mass_precision + mass_recall) > 0 else 0.0
    
    prochee_precision = prochee_tp_total / (prochee_tp_total + prochee_fp_total) if (prochee_tp_total + prochee_fp_total) > 0 else 0.0
    prochee_recall = prochee_tp_total / (prochee_tp_total + prochee_fn_total) if (prochee_tp_total + prochee_fn_total) > 0 else 0.0
    prochee_f1 = 2 * (prochee_precision * prochee_recall) / (prochee_precision + prochee_recall) if (prochee_precision + prochee_recall) > 0 else 0.0
    
    # Статистика валидации
    raw_valid_count = sum(1 for v in raw_validation_results if v["is_valid"])
    raw_invalid_count = len(raw_validation_results) - raw_valid_count
    raw_validation_errors = [v["errors"] for v in raw_validation_results if not v["is_valid"]]
    
    # Вычисляем общее количество сравнений (total = tp + fp + fn) для соответствия структуре quality_metrics
    mass_total_comparisons = mass_tp_total + mass_fp_total + mass_fn_total
    prochee_total_comparisons = prochee_tp_total + prochee_fp_total + prochee_fn_total
    
    # Возвращаем в порядке, аналогичном quality_metrics в metrics.json:
    # 1. validation (аналог validation_stats)
    # 2. "массовая доля" (аналог quality_metrics["массовая доля"])
    # 3. "прочее" (аналог quality_metrics["прочее"])
    # Порядок полей внутри каждой группы: accuracy, precision, recall, f1, tp, fp, fn, количество_сравнений, ошибки, все_ошибки
    return {
        "validation": {
            "valid_count": raw_valid_count,
            "invalid_count": raw_invalid_count,
            "validation_rate": raw_valid_count / len(raw_validation_results) if raw_validation_results else 0.0,
            "validation_errors": raw_validation_errors[:10],  # Первые 10 ошибок
            "all_validation_errors": raw_validation_errors  # Все ошибки
        },
        "массовая доля": {
            "accuracy": mass_accuracy,
            "precision": mass_precision,
            "recall": mass_recall,
            "f1": mass_f1,
            "tp": mass_tp_total,
            "fp": mass_fp_total,
            "fn": mass_fn_total,
            "количество_сравнений": mass_total_comparisons,
            "ошибки": all_mass_errors[:10],
            "все_ошибки": all_mass_errors
        },
        "прочее": {
            "accuracy": prochee_accuracy,
            "precision": prochee_precision,
            "recall": prochee_recall,
            "f1": prochee_f1,
            "tp": prochee_tp_total,
            "fp": prochee_fp_total,
            "fn": prochee_fn_total,
            "количество_сравнений": prochee_total_comparisons,
            "ошибки": all_prochee_errors[:10],
            "все_ошибки": all_prochee_errors
        }
    }

