"""
Проверка запаса до max_position_embeddings для локальных моделей.

Берёт самый длинный пример из тестового множества (входной текст + эталонный ответ),
строит полную последовательность (промпт + ответ), токенизирует её для каждой модели
и выводит: сколько токенов занято и сколько остаётся до лимита контекста.
"""
import argparse
import json
import os
import sys

import pandas as pd


def _load_dataset(path):
    """Загрузка датасета: CSV или XLSX (для XLSX нужен openpyxl)."""
    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        try:
            return pd.read_excel(path)
        except ImportError:
            csv_path = os.path.splitext(path)[0] + ".csv"
            if os.path.exists(csv_path):
                print(f"openpyxl не установлен. Используется CSV: {csv_path}")
                return pd.read_csv(csv_path, encoding="utf-8")
            raise ImportError(
                "Для Excel нужен openpyxl: pip install openpyxl"
            ) from None
    raise ValueError(f"Неизвестный формат: {path}")


def _tokenizer_name_from_model_name(name):
    """Имя модели для Hugging Face (убираем суффикс :free и т.п.)."""
    if not name or not isinstance(name, str):
        return None
    s = name.strip()
    if ":" in s:
        s = s.split(":")[0].strip()
    if "/" in s:
        return s
    if s.startswith("gemma") or "gemma" in s.lower():
        return f"google/{s}"
    if s:
        return s
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Запас до max_position_embeddings по самому длинному примеру (промпт + ответ)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Путь к датасету. По умолчанию: find_dataset_path().",
    )
    args = parser.parse_args()

    from utils import find_dataset_path, parse_json_safe, build_prompt3
    from config import PROMPT_TEMPLATE_NAME

    if args.dataset:
        dataset_path = os.path.abspath(args.dataset)
    else:
        dataset_path = find_dataset_path()
    if not os.path.exists(dataset_path):
        print(f"Ошибка: датасет не найден: {dataset_path}")
        sys.exit(1)

    df = _load_dataset(dataset_path)
    if "text" not in df.columns:
        print("Ошибка: в датасете нет колонки 'text'")
        sys.exit(1)
    col = "json_parsed" if "json_parsed" in df.columns else "json"
    if col not in df.columns:
        print("Ошибка: в датасете нет колонки 'json_parsed' или 'json'")
        sys.exit(1)

    # Строим для каждой строки: промпт + эталонный ответ (строка)
    rows_data = []
    for i in range(len(df)):
        text = df["text"].iloc[i]
        if pd.isna(text):
            text = ""
        text = str(text).strip()
        raw = df[col].iloc[i]
        if isinstance(raw, dict):
            answer_str = json.dumps(raw, ensure_ascii=False, indent=2)
        elif isinstance(raw, str):
            obj = parse_json_safe(raw)
            answer_str = json.dumps(obj, ensure_ascii=False, indent=2) if obj else raw.strip()
        else:
            answer_str = ""
        prompt = build_prompt3(text, prompt_template_name=PROMPT_TEMPLATE_NAME)
        full = prompt + "\n" + answer_str
        rows_data.append((prompt, answer_str, full, len(prompt) + len(answer_str)))

    if not rows_data:
        print("Нет данных в датасете.")
        sys.exit(1)

    # Самый длинный пример по символам
    idx_max = max(range(len(rows_data)), key=lambda i: rows_data[i][3])
    prompt_longest, answer_longest, full_sequence, _ = rows_data[idx_max]

    print(f"Датасет: {dataset_path}")
    print(f"Строк: {len(df)}")
    print(f"Самый длинный пример: строка {idx_max + 1}")
    print(f"  Длина промпта: {len(prompt_longest)} символов")
    print(f"  Длина ответа: {len(answer_longest)} символов")
    print(f"  Всего символов: {len(full_sequence)}")
    print()

    try:
        from config import HF_TOKEN
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN") or ""

    from model_config_loader import MODEL_CONFIGS
    local_configs = {
        k: v for k, v in MODEL_CONFIGS.items()
        if "-api" not in k and "-ollama" not in k
    }

    try:
        from transformers import AutoTokenizer, AutoConfig
    except ImportError as e:
        print(f"Ошибка: нужен transformers. {e}")
        sys.exit(1)

    results = []
    for model_key, config in local_configs.items():
        name = config.get("name")
        hf_name = _tokenizer_name_from_model_name(name)
        if not hf_name:
            results.append((model_key, hf_name, None, None, None))
            continue
        try:
            from utils import from_pretrained_local_first
            tokenizer = from_pretrained_local_first(AutoTokenizer.from_pretrained, hf_name, token=HF_TOKEN)
            ids = tokenizer.encode(full_sequence, add_special_tokens=True)
            n_tokens = len(ids)
            max_pos = None
            try:
                cfg = from_pretrained_local_first(AutoConfig.from_pretrained, hf_name, token=HF_TOKEN)
                max_pos = getattr(cfg, "max_position_embeddings", None)
            except Exception:
                pass
            if max_pos is None:
                max_pos = getattr(tokenizer, "model_max_length", None)
            if max_pos is None:
                max_pos = 131072
            remaining = max_pos - n_tokens
            results.append((model_key, hf_name, max_pos, n_tokens, remaining))
        except Exception as e:
            results.append((model_key, hf_name, None, None, str(e)))

    print(f"{'Модель':<32} {'max_pos':>10} {'Токенов':>10} {'Остаток':>10}")
    print("-" * 68)
    for model_key, hf_name, max_pos, n_tok, remaining in results:
        max_str = str(max_pos) if max_pos is not None else "—"
        tok_str = str(n_tok) if n_tok is not None else "—"
        if isinstance(remaining, int):
            rem_str = str(remaining)
            if remaining < 0:
                rem_str += " (превышен)"
        else:
            rem_str = remaining if remaining else "—"
        print(f"{model_key:<32} {max_str:>10} {tok_str:>10} {rem_str:>10}")

    print()
    over = [r for r in results if isinstance(r[4], int) and r[4] < 0]
    if over:
        print(f"Внимание: превышение лимита у {len(over)} моделей: {[r[0] for r in over]}")


if __name__ == "__main__":
    main()
