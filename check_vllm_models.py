"""
Проверка, что для каждого ключа *-vllm из MODEL_CONFIGS на сервере vLLM зарегистрирована модель
с тем же id, что в конфиге (поле name или vllm_name).

Сервер должен быть запущен отдельно, например:
  vllm serve Qwen/Qwen2.5-3B-Instruct --quantization awq

Базовый URL: переменная окружения VLLM_BASE_URL (по умолчанию http://127.0.0.1:8000).

Запуск из корня проекта:
  python check_vllm_models.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from model_config_loader import load_model_configs


def _vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _list_model_ids(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/v1/models"
    req = urllib.request.Request(url, method="GET")
    key = os.environ.get("VLLM_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for item in data.get("data") or []:
        mid = item.get("id")
        if mid:
            out.append(str(mid))
    return out


def main() -> int:
    base = _vllm_base_url()
    try:
        available = set(_list_model_ids(base))
    except urllib.error.URLError as e:
        print(f"vLLM недоступен ({base}): {e.reason}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Ошибка запроса к vLLM ({base}/v1/models): {e}", file=sys.stderr)
        return 2

    configs = load_model_configs()
    bad: list[tuple[str, str]] = []

    for model_key in sorted(configs.keys()):
        if "-vllm" not in model_key:
            continue
        expected = (configs[model_key].get("name") or "").strip()
        if not expected:
            bad.append((model_key, "пустое name в конфиге"))
            continue
        if expected in available:
            print(f"OK  {model_key} -> {expected!r}")
        else:
            bad.append(
                (
                    model_key,
                    f"на сервере нет модели {expected!r} (доступны: {len(available)} id); "
                    f"запустите vllm serve с этим id или задайте vllm_name в models.yaml",
                )
            )

    if bad:
        print("", file=sys.stderr)
        print("Отсутствуют на vLLM:", file=sys.stderr)
        for k, msg in bad:
            print(f"  {k}: {msg}", file=sys.stderr)
        return 1

    print("Все *-vllm модели из конфига найдены на сервере vLLM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
