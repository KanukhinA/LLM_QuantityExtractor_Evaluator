"""
Проверка, что для каждого ключа *-ollama из MODEL_CONFIGS в локальном Ollama есть модель
(команда `ollama show <тег>` успешна). Для путей .gguf проверяется стабильный тег, как в load_ollama.

Запуск из корня проекта:
  python check_ollama_models.py

При необходимости укажите теги Ollama в models.yaml у базовой модели:
  ollama_name: "qwen2.5:3b-instruct"
(поле опционально; иначе в Ollama передаётся `name`, для HF-ид это обычно не тег библиотеки.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from model_config_loader import load_model_configs
from model_loaders_ollama import ollama_inference_tag


def main() -> int:
    if not shutil.which("ollama"):
        print("ollama не найден в PATH — установите Ollama и повторите.", file=sys.stderr)
        return 2

    configs = load_model_configs()
    bad: list[tuple[str, str]] = []

    for model_key in sorted(configs.keys()):
        if "-ollama" not in model_key:
            continue
        name_for_loader = configs[model_key]["name"]
        if (
            isinstance(name_for_loader, str)
            and name_for_loader.lower().endswith(".gguf")
            and not os.path.exists(os.path.abspath(os.path.expanduser(name_for_loader)))
        ):
            bad.append((model_key, f"файл GGUF не найден: {name_for_loader!r}"))
            continue

        tag, gguf_abs = ollama_inference_tag(name_for_loader)
        kind = "gguf" if gguf_abs else "tag"

        r = subprocess.run(
            ["ollama", "show", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            hint = ""
            if "/" in tag and kind == "tag":
                hint = " (похоже на HF id — задайте ollama_name в models.yaml или путь к .gguf в name)"
            bad.append((model_key, f"нет в Ollama: {tag!r}{hint}"))
        else:
            print(f"OK  {model_key} -> {tag}")

    if bad:
        print("", file=sys.stderr)
        print("Отсутствуют или недоступны:", file=sys.stderr)
        for k, msg in bad:
            print(f"  {k}: {msg}", file=sys.stderr)
        return 1

    print("Все *-ollama модели из конфига найдены в Ollama.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
