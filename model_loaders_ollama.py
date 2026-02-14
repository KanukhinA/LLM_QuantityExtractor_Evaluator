"""
Загрузка и генерация через Ollama (локальный API http://localhost:11434).
Модель должна быть заранее загружена в Ollama (ollama pull <model>).
"""
import os
import json
import urllib.request
import urllib.error
from typing import Tuple, Any, Optional, Dict

# Метрики последнего ответа Ollama (eval_duration, eval_count и т.д.) для замера потребления
_last_ollama_metrics: Optional[Dict[str, Any]] = None


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def get_last_ollama_metrics() -> Optional[Dict[str, Any]]:
    """Возвращает метрики последнего вызова generate (eval_duration, eval_count, ...) и сбрасывает их."""
    global _last_ollama_metrics
    out = _last_ollama_metrics
    _last_ollama_metrics = None
    return out


def load_ollama(model_name: str) -> Tuple[str, None]:
    """
    «Загрузка» модели Ollama: весов нет, возвращаем имя модели для вызовов API.
    Перед запуском: ollama pull <model_name> (например ollama pull gemma3:4b).
    """
    print(f"   Ollama: использование модели '{model_name}' (должна быть загружена: ollama pull {model_name})")
    return (model_name, None)


def generate_ollama(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 1024,
    **kwargs
) -> str:
    """
    Генерация через POST /api/generate. model — имя модели Ollama (строка).
    В _last_ollama_metrics сохраняются метрики ответа (eval_duration, eval_count и т.д.).
    """
    global _last_ollama_metrics
    model_name = model if isinstance(model, str) else str(model)
    url = f"{_ollama_base_url()}/api/generate"
    body = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_new_tokens},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _last_ollama_metrics = {
            "eval_duration_ns": data.get("eval_duration"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
        }
        return (data.get("response") or "").strip()
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8") if e.fp else ""
        try:
            err_obj = json.loads(body_err) if body_err else {}
            msg = err_obj.get("error", body_err or str(e))
        except Exception:
            msg = body_err or str(e)
        raise RuntimeError(f"Ollama API error: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama недоступен ({_ollama_base_url()}): {e.reason}") from e
