"""
Модуль для извлечения few-shot примеров из неразмеченного корпуса
на основе алгоритма Dual-Level Introspective Uncertainty
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import os
from collections import Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import warnings

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Вычисляет расстояние Левенштейна между двумя строками.
    
    Args:
        s1: первая строка
        s2: вторая строка
    
    Returns:
        Расстояние Левенштейна (минимальное количество операций вставки,
        удаления или замены символов для преобразования s1 в s2)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

from utils import build_prompt3, parse_json_safe, is_valid_json, extract_json_from_response
from config import PROMPT_TEMPLATE_NAME


def cluster_russian_texts(
    df: pd.DataFrame,
    text_column: str,
    n_clusters: int = 100,
    model_name: str = "intfloat/multilingual-e5-large"
) -> pd.DataFrame:
    """
    Кластеризует строки из столбца `text_column` в DataFrame `df`.
    Возвращает исходный DataFrame + добавленная колонка 'Кластер'.
    
    Args:
        df: DataFrame с текстами
        text_column: название колонки с текстами
        n_clusters: количество кластеров
        model_name: название модели для эмбеддингов
    
    Returns:
        DataFrame с добавленной колонкой 'Кластер'
    """
    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' отсутствует в DataFrame")
    
    print(f"   Загрузка модели эмбеддингов: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"   Преобразование текстов в векторы...")
    texts = df[text_column].astype(str).tolist()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    print(f"   Кластеризация на {n_clusters} кластеров...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    df_result = df.copy()
    df_result["Кластер"] = clusters
    df_result = df_result.sort_values("Кластер").reset_index(drop=True)
    
    print(f"   ✓ Кластеризация завершена")
    return df_result


def filter_unlabeled_texts(
    unlabeled_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    text_column: str = "text"
) -> pd.DataFrame:
    """
    Фильтрует неразмеченные тексты, убирая те, что уже есть в размеченном датасете.
    
    Args:
        unlabeled_df: DataFrame с неразмеченными текстами
        labeled_df: DataFrame с размеченными текстами
        text_column: название колонки с текстами
    
    Returns:
        Отфильтрованный DataFrame с неразмеченными текстами
    """
    if text_column not in unlabeled_df.columns:
        raise ValueError(f"Колонка '{text_column}' отсутствует в unlabeled_df")
    if text_column not in labeled_df.columns:
        raise ValueError(f"Колонка '{text_column}' отсутствует в labeled_df")
    
    labeled_texts = set(labeled_df[text_column].astype(str).str.strip())
    unlabeled_texts = unlabeled_df[text_column].astype(str).str.strip()
    
    mask = ~unlabeled_texts.isin(labeled_texts)
    filtered_df = unlabeled_df[mask].reset_index(drop=True)
    
    print(f"   Исходно неразмеченных текстов: {len(unlabeled_df)}")
    print(f"   Размеченных текстов: {len(labeled_df)}")
    print(f"   Отфильтровано неразмеченных текстов: {len(filtered_df)}")
    
    return filtered_df


def extract_structured_output(json_obj: Any) -> set:
    """
    Извлекает структурированные данные из JSON в виде множества (type, text) кортежей.
    
    Args:
        json_obj: распарсенный JSON объект
    
    Returns:
        Множество кортежей (type, text)
    """
    result = set()
    
    if not isinstance(json_obj, dict):
        return result
    
    # Извлекаем массовые доли
    if "массовая доля" in json_obj:
        mass_dolya = json_obj["массовая доля"]
        if isinstance(mass_dolya, list):
            for item in mass_dolya:
                if isinstance(item, dict):
                    substance = item.get("вещество", "")
                    value = item.get("массовая доля", "")
                    if substance:
                        result.add(("массовая_доля", f"{substance}:{value}"))
    
    # Извлекаем прочее
    if "прочее" in json_obj:
        prochee = json_obj["прочее"]
        if isinstance(prochee, list):
            for item in prochee:
                if isinstance(item, dict):
                    param = item.get("параметр", "")
                    value = item.get("значение") or item.get("масса") or item.get("количество") or item.get("объем")
                    if param:
                        result.add(("прочее", f"{param}:{value}"))
    
    return result


def calculate_generation_disagreement(
    responses: List[str]
) -> float:
    """
    Вычисляет Generation Disagreement (𝒰_d) на основе среднего попарного 
    расстояния Левенштейна между k сгенерированными ответами.
    
    Формула: 𝒰_d(s_i) = 2/(k(k-1)) * Σ Levenshtein(ℳ_θ^j(s_i), ℳ_θ^l(s_i))
    для всех пар 1 ≤ j < l ≤ k
    
    Args:
        responses: список ответов модели (k вариантов)
    
    Returns:
        Generation Disagreement (среднее попарное расстояние Левенштейна)
    """
    k = len(responses)
    if k < 2:
        return 0.0
    
    # Вычисляем попарные расстояния Левенштейна
    distances = []
    for i in range(k):
        for j in range(i + 1, k):
            dist = levenshtein_distance(responses[i], responses[j])
            # Нормализуем расстояние на максимальную длину строк для получения значения в [0, 1]
            max_len = max(len(responses[i]), len(responses[j]), 1)
            normalized_dist = dist / max_len
            distances.append(normalized_dist)
    
    # Среднее попарное расстояние согласно формуле (2)
    # Формула: 2/(k(k-1)) * Σ distances, но так как мы уже усредняем, просто берем mean
    avg_distance = np.mean(distances) if distances else 0.0
    
    return avg_distance


def calculate_format_uncertainty(
    responses: List[str],
    parser_func: callable = parse_json_safe
) -> Tuple[float, float]:
    """
    Вычисляет format-level uncertainty (R_fail + Structural Disagreement).
    
    Args:
        responses: список ответов модели (k вариантов)
        parser_func: функция для парсинга JSON
    
    Returns:
        Кортеж (R_fail, structural_disagreement)
    """
    k = len(responses)
    if k == 0:
        return 1.0, 1.0
    
    # Вычисляем R_fail (parsing failure rate)
    failed_count = 0
    parsed_outputs = []
    
    for response in responses:
        json_part = extract_json_from_response(response)
        parsed = parser_func(json_part)
        is_valid = is_valid_json(json_part)
        
        if not is_valid or parsed is None:
            failed_count += 1
        else:
            parsed_outputs.append(parsed)
    
    R_fail = failed_count / k
    
    # Вычисляем Structural Disagreement (для успешно распарсенных)
    if len(parsed_outputs) < 2:
        structural_disagreement = 1.0 if R_fail > 0 else 0.0
    else:
        # Измеряем вариативность структуры (разные ключи, разная длина списков)
        structures = []
        for parsed in parsed_outputs:
            if isinstance(parsed, dict):
                # Извлекаем структуру: набор ключей и длины списков
                keys = set(parsed.keys())
                list_lengths = {}
                for key, value in parsed.items():
                    if isinstance(value, list):
                        list_lengths[key] = len(value)
                structures.append((keys, list_lengths))
        
        # Вычисляем вариативность структур
        if len(structures) == 0:
            structural_disagreement = 1.0
        else:
            # Сравниваем структуры попарно
            disagreements = []
            for i in range(len(structures)):
                for j in range(i + 1, len(structures)):
                    keys_i, lengths_i = structures[i]
                    keys_j, lengths_j = structures[j]
                    
                    # Различие в ключах
                    key_diff = len(keys_i.symmetric_difference(keys_j)) / max(len(keys_i.union(keys_j)), 1)
                    
                    # Различие в длинах списков
                    all_keys = set(lengths_i.keys()) | set(lengths_j.keys())
                    length_diff = 0.0
                    if all_keys:
                        for key in all_keys:
                            len_i = lengths_i.get(key, 0)
                            len_j = lengths_j.get(key, 0)
                            if len_i != len_j:
                                length_diff += 1.0
                        length_diff /= len(all_keys)
                    
                    disagreement = (key_diff + length_diff) / 2.0
                    disagreements.append(disagreement)
            
            structural_disagreement = np.mean(disagreements) if disagreements else 0.0
    
    return R_fail, structural_disagreement


def calculate_content_uncertainty(
    responses: List[str],
    parser_func: callable = parse_json_safe
) -> float:
    """
    Вычисляет content-level uncertainty на основе Jaccard similarity.
    
    Args:
        responses: список ответов модели (k вариантов)
        parser_func: функция для парсинга JSON
    
    Returns:
        Content-level uncertainty (1 - average Jaccard similarity)
    """
    # Парсим все ответы
    parsed_outputs = []
    for response in responses:
        json_part = extract_json_from_response(response)
        parsed = parser_func(json_part)
        if parsed is not None and isinstance(parsed, dict):
            parsed_outputs.append(parsed)
    
    k_prime = len(parsed_outputs)
    if k_prime < 2:
        return 1.0  # Высокая неопределенность, если нет успешно распарсенных ответов
    
    # Извлекаем структурированные данные для каждого ответа
    extracted_sets = []
    for parsed in parsed_outputs:
        extracted = extract_structured_output(parsed)
        extracted_sets.append(extracted)
    
    # Вычисляем попарную Jaccard similarity
    jaccard_similarities = []
    for i in range(len(extracted_sets)):
        for j in range(i + 1, len(extracted_sets)):
            set_i = extracted_sets[i]
            set_j = extracted_sets[j]
            
            if len(set_i) == 0 and len(set_j) == 0:
                similarity = 1.0
            elif len(set_i) == 0 or len(set_j) == 0:
                similarity = 0.0
            else:
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0.0
            
            jaccard_similarities.append(similarity)
    
    avg_jaccard = np.mean(jaccard_similarities) if jaccard_similarities else 0.0
    content_uncertainty = 1.0 - avg_jaccard
    
    return content_uncertainty


def generate_multiple_responses(
    text: str,
    generate_func: callable,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 1024,
    k: int = 3,
    hyperparameters: Dict[str, Any] = None,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> List[str]:
    """
    Генерирует k вариантов ответа для одного текста с использованием sampling.
    
    Args:
        text: входной текст
        generate_func: функция генерации
        model: модель
        tokenizer: токенизатор
        max_new_tokens: максимальное количество новых токенов
        k: количество вариантов ответа (по умолчанию 3)
        hyperparameters: гиперпараметры модели
        temperature: температура для sampling (по умолчанию 0.7)
        top_p: top_p для sampling (по умолчанию 0.95)
    
    Returns:
        Список из k ответов
    """
    import torch
    prompt = build_prompt3(text)
    responses = []
    
    # Проверяем, является ли это API моделью
    is_api_model = hyperparameters and hyperparameters.get("api_model", False)
    
    for i in range(k):
        try:
            if is_api_model:
                # Для API моделей передаем параметры sampling напрямую
                if "model_name" in hyperparameters:
                    # Передаем temperature и top_p для API моделей
                    try:
                        response = generate_func(
                            model, tokenizer, prompt, max_new_tokens, 
                            model_name=hyperparameters["model_name"],
                            temperature=temperature,
                            top_p=top_p if temperature > 0 else None
                        )
                    except TypeError:
                        # Если функция не поддерживает эти параметры, используем без них
                        response = generate_func(model, tokenizer, prompt, max_new_tokens, 
                                                model_name=hyperparameters["model_name"])
                else:
                    try:
                        response = generate_func(
                            model, tokenizer, prompt, max_new_tokens,
                            temperature=temperature,
                            top_p=top_p if temperature > 0 else None
                        )
                    except TypeError:
                        # Если функция не поддерживает эти параметры, используем без них
                        response = generate_func(model, tokenizer, prompt, max_new_tokens)
            else:
                # Для локальных моделей используем прямое обращение к model.generate()
                # с параметрами sampling
                
                # Проверяем, нужен ли chat template (для Gemma 3 и подобных)
                from transformers import Gemma3ForCausalLM
                is_gemma3 = isinstance(model, Gemma3ForCausalLM) or model.__class__.__name__ == 'Gemma3ForCausalLM'
                
                if is_gemma3 and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                    # Используем формат сообщений для Gemma 3
                    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
                    inputs_dict = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    device = next(model.parameters()).device
                    inputs = {}
                    for key, value in inputs_dict.items():
                        if isinstance(value, torch.Tensor):
                            if key == "input_ids":
                                inputs[key] = value.to(device)
                            else:
                                inputs[key] = value.to(device).to(torch.bfloat16)
                        else:
                            inputs[key] = value
                    
                    generate_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    
                    if tokenizer.eos_token_id is not None:
                        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
                    
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, **generate_kwargs)
                    
                    outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    full_text = outputs_decoded[0] if outputs_decoded else ""
                    
                    inputs_decoded = tokenizer.batch_decode(
                        inputs['input_ids'] if isinstance(inputs, dict) else inputs, 
                        skip_special_tokens=True
                    )
                    input_text = inputs_decoded[0] if inputs_decoded else ""
                    
                    if full_text.startswith(input_text):
                        response = full_text[len(input_text):].strip()
                    else:
                        response = full_text
                else:
                    # Стандартная обработка для других моделей
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    # Декодируем только новые токены
                    input_length = input_ids.shape[1]
                    generated_ids = output_ids[0][input_length:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Если декодирование новых токенов дало пустой результат, пробуем декодировать весь ответ
                    if not response.strip():
                        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        if generated_text.startswith(prompt):
                            response = generated_text[len(prompt):].strip()
                        else:
                            response = generated_text.strip()
            
            responses.append(response)
        except Exception as e:
            warnings.warn(f"Ошибка при генерации варианта {i+1} для текста: {e}")
            responses.append("")  # Пустой ответ в случае ошибки
    
    return responses


def extract_few_shot_examples(
    unlabeled_corpus_path: str,
    labeled_dataset_path: str,
    n_examples: int,
    generate_func: callable,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 1024,
    k: int = 3,
    hyperparameters: Dict[str, Any] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    text_column: str = "text",
    n_clusters: int = 100,
    alpha: float = 0.33,  # вес для Generation Disagreement (𝒰_d)
    beta: float = 0.33,   # вес для Format Uncertainty (𝒰_f)
    gamma: float = 0.34,  # вес для Content Uncertainty (𝒰_c)
    verbose: bool = False
) -> pd.DataFrame:
    """
    Извлекает few-shot примеры из неразмеченного корпуса на основе алгоритма
    Dual-Level Introspective Uncertainty.
    
    Алгоритм вычисляет три типа неопределенности:
    1. Generation Disagreement (𝒰_d): среднее попарное расстояние Левенштейна между k ответами
    2. Format-Level Uncertainty (𝒰_f): R_fail (parsing failure rate) + Structural Disagreement
    3. Content-Level Uncertainty (𝒰_c): 1 - средняя Jaccard similarity между извлеченными данными
    
    Общая неопределенность: 𝒰_total = α·𝒰_d + β·𝒰_f + γ·𝒰_c
    
    Args:
        unlabeled_corpus_path: путь к файлу с неразмеченным корпусом (Excel)
        labeled_dataset_path: путь к размеченному датасету (Excel)
        n_examples: количество примеров для возврата в итоговом результате (топ-N по неопределенности).
                   Обрабатываются все кластеры (по одному тексту из каждого), но возвращается только топ-n_examples
        generate_func: функция генерации модели
        model: модель
        tokenizer: токенизатор
        max_new_tokens: максимальное количество новых токенов
        k: количество вариантов ответа для каждого текста
        hyperparameters: гиперпараметры модели
        text_column: название колонки с текстами
        n_clusters: количество кластеров для кластеризации (определяет, сколько текстов будет обработано)
        alpha: вес для Generation Disagreement (𝒰_d)
        beta: вес для Format Uncertainty (𝒰_f)
        gamma: вес для Content Uncertainty (𝒰_c)
        verbose: подробный вывод
    
    Returns:
        DataFrame с топ-n_examples примерами, отсортированными по общей неопределенности
    """
    print(f"\n{'='*80}")
    print(f"ИЗВЛЕЧЕНИЕ FEW-SHOT ПРИМЕРОВ")
    print(f"{'='*80}\n")
    
    # 1. Загружаем данные
    print(f"📂 Загрузка данных...")
    print(f"   Неразмеченный корпус: {unlabeled_corpus_path}")
    unlabeled_df = pd.read_excel(unlabeled_corpus_path)
    print(f"   ✓ Загружено {len(unlabeled_df)} неразмеченных текстов")
    
    print(f"   Размеченный датасет: {labeled_dataset_path}")
    labeled_df = pd.read_excel(labeled_dataset_path)
    print(f"   ✓ Загружено {len(labeled_df)} размеченных текстов")
    
    # 2. Фильтруем неразмеченные тексты
    print(f"\n🔍 Фильтрация неразмеченных текстов...")
    filtered_df = filter_unlabeled_texts(unlabeled_df, labeled_df, text_column)
    
    if len(filtered_df) == 0:
        print("   ⚠️ Нет неразмеченных текстов после фильтрации!")
        return pd.DataFrame()
    
    # 3. Кластеризация
    print(f"\n📊 Кластеризация текстов...")
    clustered_df = cluster_russian_texts(filtered_df, text_column, n_clusters=n_clusters)
    
    # Выбираем по одному примеру из каждого кластера (обрабатываем все кластеры)
    # n_examples определяет, сколько из обработанных текстов вернуть в итоге (топ-N по неопределенности)
    unique_clusters = clustered_df["Кластер"].unique()
    selected_indices = []
    
    # Выбираем по одному примеру из каждого кластера
    for cluster_id in unique_clusters:
        cluster_texts = clustered_df[clustered_df["Кластер"] == cluster_id]
        if len(cluster_texts) > 0:
            selected_indices.append(cluster_texts.index[0])
    
    candidate_df = clustered_df.loc[selected_indices].reset_index(drop=True)
    print(f"   ✓ Выбрано {len(candidate_df)} кандидатов из {len(unique_clusters)} кластеров для оценки")
    print(f"   ℹ️  В итоговый результат будет возвращено топ-{n_examples} примеров по неопределенности")
    
    # 4. Оценка неопределенности для каждого кандидата
    print(f"\n🤖 Генерация {k} вариантов ответа для каждого кандидата...")
    results = []
    
    for idx, row in candidate_df.iterrows():
        text = row[text_column]
        print(f"   [{idx+1}/{len(candidate_df)}] Обработка текста из кластера {row.get('Кластер', -1)}...")
        if verbose:
            print(f"      Текст: {text[:100]}...")
        
        # Генерируем k вариантов ответа с sampling
        responses = generate_multiple_responses(
            text, generate_func, model, tokenizer, max_new_tokens, k, hyperparameters,
            temperature=temperature, top_p=top_p
        )
        
        # Вычисляем метрики неопределенности
        generation_disagreement = calculate_generation_disagreement(responses)
        
        R_fail, structural_disagreement = calculate_format_uncertainty(responses)
        format_uncertainty = (R_fail + structural_disagreement) / 2.0
        
        content_uncertainty = calculate_content_uncertainty(responses)
        
        # Нормализуем все метрики к [0, 1] (они уже в этом диапазоне)
        # Общая неопределенность согласно формуле (5)
        total_uncertainty = alpha * generation_disagreement + beta * format_uncertainty + gamma * content_uncertainty
        
        results.append({
            "text": text,
            "cluster": row.get("Кластер", -1),
            "generation_disagreement": generation_disagreement,
            "R_fail": R_fail,
            "structural_disagreement": structural_disagreement,
            "format_uncertainty": format_uncertainty,
            "content_uncertainty": content_uncertainty,
            "total_uncertainty": total_uncertainty,
            "responses": responses
        })
        
        # Выводим результаты анализа для каждого примера (по умолчанию)
        print(f"      ✓ Total Uncertainty: {total_uncertainty:.3f} "
              f"(Disagreement: {generation_disagreement:.3f}, "
              f"Format: {format_uncertainty:.3f}, "
              f"Content: {content_uncertainty:.3f})")
        
        if verbose:
            print(f"      R_fail: {R_fail:.3f}, Structural disagreement: {structural_disagreement:.3f}")
    
    # 5. Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    
    # 6. Сортируем по общей неопределенности (по убыванию)
    results_df = results_df.sort_values("total_uncertainty", ascending=False).reset_index(drop=True)
    
    print(f"\n✅ Извлечение завершено!")
    print(f"   Топ-{min(n_examples, len(results_df))} примеров с наивысшей неопределенностью:\n")
    for idx, row in results_df.head(n_examples).iterrows():
        print(f"{'='*80}")
        print(f"Пример {idx+1} (Total Uncertainty: {row['total_uncertainty']:.3f})")
        print(f"  Disagreement: {row['generation_disagreement']:.3f}, "
              f"Format: {row['format_uncertainty']:.3f}, "
              f"Content: {row['content_uncertainty']:.3f}")
        print(f"{'-'*80}")
        print(f"Текст:")
        print(f"{row['text']}")
        print(f"\nОТВЕТ:")
        
        # Пытаемся найти первый успешно распарсенный JSON из responses
        best_response = None
        responses = row.get('responses', [])
        
        for response in responses:
            if response and response.strip():
                json_part = extract_json_from_response(response)
                parsed = parse_json_safe(json_part)
                if parsed is not None and isinstance(parsed, dict):
                    best_response = json_part
                    break
        
        # Если не нашли распарсенный JSON, используем первый непустой ответ
        if best_response is None:
            for response in responses:
                if response and response.strip():
                    best_response = extract_json_from_response(response)
                    if best_response:
                        break
        
        # Если нашли JSON, выводим его в формате markdown
        if best_response:
            print(f"```json")
            print(f"{best_response}")
            print(f"```")
        else:
            # Если JSON не найден, выводим первый ответ как есть
            if responses and responses[0]:
                print(responses[0])
            else:
                print("(Ответ не получен)")
        print()
    
    return results_df


if __name__ == "__main__":
    import argparse
    import sys
    from config import UNLABELED_CORPUS_PATH
    from utils import find_dataset_path, find_file_path
    import model_loaders as ml
    
    parser = argparse.ArgumentParser(
        description="Извлечение few-shot примеров из неразмеченного корпуса на основе Dual-Level Introspective Uncertainty"
    )
    parser.add_argument(
        "model_key",
        type=str,
        help="Ключ модели из MODEL_CONFIGS (например, gemma-3-4b, qwen-2.5-3b)"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Количество текстов из неразмеченного корпуса для извлечения (по умолчанию: 10)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
        help="Количество кластеров для кластеризации (по умолчанию: 100)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Количество вариантов ответа для каждого текста (по умолчанию: 3)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Температура для sampling (по умолчанию: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p для sampling (по умолчанию: 0.95)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.33,
        help="Вес для Generation Disagreement (по умолчанию: 0.33)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.33,
        help="Вес для Format Uncertainty (по умолчанию: 0.33)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.34,
        help="Вес для Content Uncertainty (по умолчанию: 0.34)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Название колонки с текстами (по умолчанию: text)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Максимальное количество новых токенов (по умолчанию: 1024)"
    )
    parser.add_argument(
        "--unlabeled-corpus",
        type=str,
        default=None,
        help="Путь к неразмеченному корпусу (переопределяет значение из config.py)"
    )
    parser.add_argument(
        "--labeled-dataset",
        type=str,
        default=None,
        help="Путь к размеченному датасету (по умолчанию используется стандартный датасет через find_dataset_path())"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения результатов в CSV (опционально)"
    )
    
    args = parser.parse_args()
    
    # Определяем пути к корпусам
    # Размеченный датасет использует тот же файл, что и для оценки моделей
    labeled_dataset_path = args.labeled_dataset or find_dataset_path()
    
    # Для неразмеченного корпуса используем поиск в родительских директориях
    if args.unlabeled_corpus:
        unlabeled_corpus_path = args.unlabeled_corpus
        if not os.path.exists(unlabeled_corpus_path):
            print(f"❌ Ошибка: файл неразмеченного корпуса не найден: {unlabeled_corpus_path}")
            sys.exit(1)
    elif UNLABELED_CORPUS_PATH:
        try:
            unlabeled_corpus_path = find_file_path(UNLABELED_CORPUS_PATH)
        except FileNotFoundError as e:
            print(f"❌ Ошибка: {e}")
            print("   Укажите путь одним из способов:")
            print("   1. В config.py: UNLABELED_CORPUS_PATH = 'data/udobrenia.xlsx'")
            print("   2. Через аргумент: --unlabeled-corpus path/to/unlabeled_corpus.xlsx")
            sys.exit(1)
    else:
        print("❌ Ошибка: путь к неразмеченному корпусу не указан!")
        print("   Укажите путь одним из способов:")
        print("   1. В config.py: UNLABELED_CORPUS_PATH = 'data/udobrenia.xlsx'")
        print("   2. Через аргумент: --unlabeled-corpus path/to/unlabeled_corpus.xlsx")
        sys.exit(1)
    
    if not os.path.exists(labeled_dataset_path):
        print(f"❌ Ошибка: файл размеченного датасета не найден: {labeled_dataset_path}")
        sys.exit(1)
    
    # Загружаем конфигурацию модели
    from main import MODEL_CONFIGS
    
    if args.model_key not in MODEL_CONFIGS:
        print(f"❌ Ошибка: модель '{args.model_key}' не найдена в MODEL_CONFIGS")
        print(f"   Доступные модели: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    model_config = MODEL_CONFIGS[args.model_key]
    
    # Проверяем, является ли это API моделью
    is_api_model = model_config["hyperparameters"].get("api_model", False)
    
    if is_api_model:
        print("⚠️  Внимание: API модели могут не поддерживать sampling параметры напрямую")
        print("   Генерация будет использовать стандартную функцию генерации")
    
    print(f"\n{'='*80}")
    print(f"ЗАГРУЗКА МОДЕЛИ: {model_config['name']}")
    print(f"{'='*80}\n")
    
    # Загружаем модель
    try:
        model, tokenizer = model_config["load_func"]()
        print(f"✅ Модель успешно загружена\n")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Запускаем извлечение
    try:
        results_df = extract_few_shot_examples(
            unlabeled_corpus_path=unlabeled_corpus_path,
            labeled_dataset_path=labeled_dataset_path,
            n_examples=args.n_examples,
            generate_func=model_config["generate_func"],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            k=args.k,
            hyperparameters=model_config["hyperparameters"],
            temperature=args.temperature,
            top_p=args.top_p,
            text_column=args.text_column,
            n_clusters=args.n_clusters,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            verbose=args.verbose
        )
        
        # Сохраняем результаты, если указан путь
        if args.output:
            results_df.to_csv(args.output, index=False, encoding='utf-8')
            print(f"\n✅ Результаты сохранены в: {args.output}")
        else:
            # Сохраняем в results/ по умолчанию
            from config import OUTPUT_DIR
            from datetime import datetime
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"few_shot_examples_{args.model_key}_{timestamp}.csv")
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\n✅ Результаты сохранены в: {output_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка при извлечении примеров: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

