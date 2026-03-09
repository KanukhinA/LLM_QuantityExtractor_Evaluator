"""
Автоматическая разметка few-shot примеров через Gemini API.

Читает CSV от few_shot_extractor (колонка text), для каждого текста запрашивает
у Gemini извлечение JSON по той же задаче, сохраняет результат в XLSX для ручной проверки.
Промпт MINIMAL_INSTR_FIVESHOT_APIE при оценке модели читает примеры из XLSX (приоритет) или CSV.
"""
import os
import sys
import time
import argparse
from datetime import datetime

try:
    from google import genai
except ImportError:
    genai = None

import pandas as pd

from config import OUTPUT_DIR
from utils import extract_json_from_response
import prompt_config


def extract_json_with_gemini(
    text: str,
    gemini_api_key: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Отправляет текст в Gemini с промптом извлечения, возвращает извлечённый JSON-фрагмент.
    """
    if genai is None:
        raise RuntimeError("google-genai не установлен. Установите: pip install google-genai")
    prompt = prompt_config.DETAILED_INSTR_ZEROSHOT_BASELINE.format(text=text)
    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    if hasattr(response, "text"):
        raw = response.text
    elif hasattr(response, "candidates") and response.candidates:
        raw = response.candidates[0].content.parts[0].text
    else:
        raw = str(response)
    return extract_json_from_response(raw) or ""


def label_few_shot_to_xlsx(
    input_path: str,
    output_path: str = None,
    model_key: str = None,
    max_examples: int = None,
    gemini_api_key: str = None,
    gemini_model: str = "gemini-2.5-flash",
) -> str:
    """
    Читает CSV (или XLSX) с колонкой text, для каждой строки получает JSON от Gemini, сохраняет в XLSX.

    Args:
        input_path: путь к CSV/XLSX от few_shot_extractor
        output_path: путь к выходному XLSX (если None — генерируется в OUTPUT_DIR)
        model_key: ключ модели для имени файла (если output_path не задан)
        max_examples: максимум строк для разметки (None — все)
        gemini_api_key: API ключ Gemini (None — из config/переменных окружения)
        gemini_model: модель Gemini

    Returns:
        Путь к сохранённому XLSX.
    """
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
    try:
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY and not gemini_api_key:
            gemini_api_key = GEMINI_API_KEY
    except Exception:
        pass
    if not gemini_api_key or not gemini_api_key.strip():
        raise ValueError("GEMINI_API_KEY не задан. Укажите в config_secrets.py или переменной окружения.")

    if input_path.lower().endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path, encoding="utf-8")

    if "text" not in df.columns:
        raise ValueError(f"В файле {input_path} нет колонки 'text'.")

    if max_examples is not None:
        df = df.head(max_examples)

    texts = df["text"].astype(str).tolist()
    n = len(texts)
    json_list = []
    for i, text in enumerate(texts):
        print(f"   [{i+1}/{n}] Gemini разметка...")
        try:
            js = extract_json_with_gemini(text, gemini_api_key, model=gemini_model)
            json_list.append(js)
        except Exception as e:
            print(f"      Ошибка: {e}")
            json_list.append("")
        time.sleep(0.3)
    df = df.copy()
    df["json"] = json_list

    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"few_shot_examples_{model_key}_" if model_key else "few_shot_examples_"
        output_path = os.path.join(OUTPUT_DIR, f"{prefix}gemini_{timestamp}.xlsx")
    df.to_excel(output_path, index=False, engine="openpyxl")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Разметка few-shot примеров через Gemini API (результат в XLSX для ручной проверки)"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Путь к CSV или XLSX от few_shot_extractor (обязательна колонка text)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь к выходному XLSX (по умолчанию: results/few_shot_examples_<model_key>_gemini_<timestamp>.xlsx)"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Ключ модели для имени файла (если --output не указан)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Максимум примеров для разметки (по умолчанию — все)"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Модель Gemini (по умолчанию: gemini-2.5-flash)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Ошибка: файл не найден: {args.input}")
        sys.exit(1)

    if genai is None:
        print("Ошибка: установите google-genai: pip install google-genai")
        sys.exit(1)

    print("Разметка few-shot примеров через Gemini API...")
    try:
        out = label_few_shot_to_xlsx(
            args.input,
            output_path=args.output,
            model_key=args.model_key,
            max_examples=args.max_examples,
            gemini_model=args.gemini_model,
        )
        print(f"Готово. Результат сохранён: {out}")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
