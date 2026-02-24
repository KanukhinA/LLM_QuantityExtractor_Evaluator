"""
Оценка max_new_tokens по тестовому датасету.

Режимы:
- --all-models: взять самый длинный ответ (по символам), прогнать через токенизаторы
  только локальных моделей из models.yaml (API и Ollama пропускаются); вывести таблицу.
- Без --all-models: один токенизатор (--tokenizer), подсчёт по всем ответам, статистика и одна рекомендация.
"""
import argparse
import json
import os
import sys

import pandas as pd


def _load_dataset(path):
    """Загрузка датасета: CSV без openpyxl, XLSX требует openpyxl. При отсутствии openpyxl пробует .csv рядом."""
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
                "Для чтения Excel (.xlsx) нужен openpyxl: pip install openpyxl\n"
                "Либо сохраните датасет в CSV (тот же путь с расширением .csv) и запустите снова."
            ) from None
    raise ValueError(
        f"Неизвестный формат: {path}. Поддерживаются .csv (без openpyxl) и .xlsx/.xls (нужен openpyxl)."
    )


def _longest_string_from_dataset(df, col, parse_json_safe):
    strings = []
    for j in df[col]:
        if pd.isna(j):
            strings.append("")
            continue
        if isinstance(j, dict):
            obj = j
        elif isinstance(j, list):
            obj = j
        elif isinstance(j, str):
            obj = parse_json_safe(j)
        else:
            obj = {}
        if obj:
            s = json.dumps(obj, ensure_ascii=False, indent=2)
            strings.append(s)
        elif col == "json" and isinstance(j, str) and j.strip():
            strings.append(j.strip())
        else:
            strings.append("")
    return strings


def _tokenizer_name_from_model_name(name):
    """Имя для Hugging Face: убираем суффикс типа :free (OpenRouter), при необходимости добавляем org."""
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


def _load_tokenizer_and_count(hf_name, text, hf_token, _fallback_printed=None):
    """Загрузка токенизатора: сначала transformers, при OSError (DLL/torch) — tokenizers (без PyTorch)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_name, token=hf_token or None)
        return len(tok.encode(text, add_special_tokens=False)), _fallback_printed
    except OSError:
        if _fallback_printed is not None and not _fallback_printed[0]:
            _fallback_printed[0] = True
            print("PyTorch не загружается (ошибка DLL на Windows). Используется библиотека tokenizers (без torch).\n")
        try:
            from tokenizers import Tokenizer
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            tok = Tokenizer.from_pretrained(hf_name)
            return len(tok.encode(text).ids), _fallback_printed
        except Exception:
            raise


def _make_tokenizer_encoder(hf_name, hf_token):
    """Возвращает функцию (text) -> число токенов. При OSError (DLL) или отсутствии transformers использует tokenizers."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_name, token=hf_token or None)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    except (OSError, ImportError):
        print("Используется библиотека tokenizers (без PyTorch/transformers).\n")
        from tokenizers import Tokenizer
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        tok = Tokenizer.from_pretrained(hf_name)
        return lambda s: len(tok.encode(s).ids)


def run_all_models(args, dataset_path, df, col, longest_str, HF_TOKEN):
    from model_config_loader import MODEL_CONFIGS

    # Нелокальные модели (API, Ollama) не имеют локального токенизатора — оцениваем только локальные
    local_configs = {
        k: v for k, v in MODEL_CONFIGS.items()
        if "-api" not in k and "-ollama" not in k
    }

    fallback_printed = [False]

    results = []
    for model_key, config in local_configs.items():
        name = config.get("name")
        hf_name = _tokenizer_name_from_model_name(name)
        if not hf_name:
            results.append((model_key, name, None, None))
            continue
        try:
            n_tokens, _ = _load_tokenizer_and_count(hf_name, longest_str, HF_TOKEN, fallback_printed)
            rec = n_tokens + int(n_tokens * args.margin)
            rec = max(rec, args.min)
            if args.round > 0:
                rec = ((rec + args.round - 1) // args.round) * args.round
            results.append((model_key, hf_name, n_tokens, rec))
        except Exception as e:
            results.append((model_key, hf_name, None, str(e)))

    print(f"\nТекст: самый длинный ответ из датасета ({len(longest_str)} символов)")
    print(f"Запас: {int(args.margin*100)}%, min={args.min}, округление до {args.round}\n")
    print(f"{'Модель':<35} {'Токенизатор':<45} {'Токенов':>8} {'max_new_tokens':>16}")
    print("-" * 110)
    for model_key, hf_name, n_tok, rec in results:
        tok_str = str(n_tok) if n_tok is not None else "—"
        rec_str = str(rec) if isinstance(rec, int) else (rec if rec else "—")
        hf_short = (hf_name[:42] + "...") if hf_name and len(hf_name) > 45 else (hf_name or "—")
        print(f"{model_key:<35} {hf_short:<45} {tok_str:>8} {rec_str:>16}")


def main():
    parser = argparse.ArgumentParser(
        description="Оценка max_new_tokens по длине ground truth ответов в токенах."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Путь к датасету (Excel). По умолчанию: find_dataset_path().",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Прогнать самый длинный ответ через токенизаторы всех моделей из models.yaml.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Имя токенизатора (если не указан --all-models).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Запас к max в долях (по умолчанию 0.15 = 15%%).",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=256,
        help="Минимальное рекомендуемое max_new_tokens (по умолчанию 256).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=64,
        help="Округлять рекомендацию до кратного (по умолчанию 64).",
    )
    args = parser.parse_args()

    from utils import find_dataset_path, parse_json_safe
    if args.dataset:
        dataset_path = os.path.abspath(args.dataset)
    else:
        dataset_path = find_dataset_path()
    if not os.path.exists(dataset_path):
        print(f"Ошибка: датасет не найден: {dataset_path}")
        sys.exit(1)

    print(f"Датасет: {dataset_path}")
    df = _load_dataset(dataset_path)
    if "json_parsed" not in df.columns and "json" not in df.columns:
        print("Ошибка: в датасете нет колонки 'json_parsed' или 'json'")
        sys.exit(1)

    col = "json_parsed" if "json_parsed" in df.columns else "json"
    strings = _longest_string_from_dataset(df, col, parse_json_safe)
    non_empty = sum(1 for s in strings if s)
    if non_empty == 0:
        print("Нет непустых ground truth для подсчёта.")
        print(f"Колонка: {col}, всего строк: {len(strings)}. Примеры значений (первые 3):")
        for i, j in enumerate(df[col][:3]):
            t = type(j).__name__
            v = repr(j)[:80] + "..." if len(repr(j)) > 80 else repr(j)
            print(f"  [{i}] {t}: {v}")
        sys.exit(1)
    print(f"Колонка: {col}, непустых ответов: {non_empty} из {len(strings)}")

    longest_str = max(strings, key=len)
    if args.all_models:
        try:
            from config import HF_TOKEN
        except Exception:
            HF_TOKEN = os.environ.get("HF_TOKEN") or ""
        run_all_models(args, dataset_path, df, col, longest_str, HF_TOKEN)
        return

    try:
        from config import HF_TOKEN
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN") or ""

    print(f"Токенизатор: {args.tokenizer}")
    try:
        encode_fn = _make_tokenizer_encoder(args.tokenizer, HF_TOKEN)
    except ImportError as e:
        print(f"Ошибка: нужен transformers или tokenizers. {e}")
        sys.exit(1)
    lengths = []
    for s in strings:
        if not s:
            lengths.append(0)
            continue
        lengths.append(encode_fn(s))

    lengths = [l for l in lengths if l > 0]
    if not lengths:
        print("Нет ненулевых длин.")
        sys.exit(1)

    n = len(lengths)
    max_len = max(lengths)
    mean_len = sum(lengths) / n
    lengths_sorted = sorted(lengths)
    p90 = lengths_sorted[int(0.9 * n) - 1] if n >= 1 else 0
    p95 = lengths_sorted[int(0.95 * n) - 1] if n >= 1 else 0
    p99 = lengths_sorted[int(0.99 * n) - 1] if n >= 1 else 0

    print(f"\nОтветов (непустых): {n}")
    print(f"Длина в токенах: min={min(lengths)}, max={max_len}, mean={mean_len:.1f}")
    print(f"Перцентили: p90={p90}, p95={p95}, p99={p99}")

    recommended = max_len + int(max_len * args.margin)
    recommended = max(recommended, args.min)
    if args.round > 0:
        recommended = ((recommended + args.round - 1) // args.round) * args.round
    print(f"\nРекомендуемое max_new_tokens (max + {int(args.margin*100)}%%, не менее {args.min}, кратно {args.round}): {recommended}")


if __name__ == "__main__":
    main()
