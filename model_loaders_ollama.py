"""
Загрузка и генерация через Ollama (локальный API http://localhost:11434).
Модель должна быть заранее загружена в Ollama (ollama pull <model>).
"""
import os
import json
import urllib.request
import urllib.error
import subprocess
import tempfile
import hashlib
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


def ollama_inference_tag(model_user_name: str) -> Tuple[str, Optional[str]]:
    """
    Тег модели для API Ollama и абсолютный путь GGUF (если применимо), без создания модели.

    Возвращает (ollama_tag, abs_gguf_path_or_none).
    Если путь GGUF существует — tag стабильный от пути; иначе tag == model_user_name (тег из ollama pull и т.п.).
    """
    if not (isinstance(model_user_name, str) and model_user_name.strip()):
        return str(model_user_name), None
    expanded = os.path.expanduser(model_user_name)
    abs_path = os.path.abspath(expanded) if os.path.exists(expanded) else expanded
    if model_user_name.lower().endswith(".gguf") and os.path.exists(abs_path):
        digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:12]
        base = os.path.splitext(os.path.basename(abs_path))[0]
        safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in base)
        return f"gguf-{safe_base}-{digest}", abs_path
    return model_user_name, None


def load_ollama(model_name: str) -> Tuple[str, None]:
    """
    «Загрузка» модели Ollama: весов нет, возвращаем имя модели для вызовов API.

    Поддерживаем 2 сценария:
    1) model_name — это тег модели в Ollama (например: llama3:8b-instruct). В этом случае модель предполагается заранее загруженной.
    2) model_name — это путь к локальному GGUF-файлу (например: /path/to/model.Q4_K_M.gguf). В этом случае создаём Ollama-модель через Modelfile автоматически.
    """
    ollama_tag, abs_path = ollama_inference_tag(model_name)

    # 1) GGUF путь -> auto-create model in ollama
    if abs_path is not None:
        print(f"   Ollama: GGUF '{abs_path}' -> создаём/используем '{ollama_tag}'")

        try:
            subprocess.run(["ollama", "show", ollama_tag],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True)
            print(f"   Ollama: модель '{ollama_tag}' уже существует, пропускаем create")
            return (ollama_tag, None)
        except subprocess.CalledProcessError:
            pass

        with tempfile.TemporaryDirectory() as td:
            modelfile_path = os.path.join(td, "Modelfile")
            modelfile = f'FROM "{abs_path}"\n'
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile)

            # Создаём модель в Ollama
            proc = subprocess.run(
                ["ollama", "create", ollama_tag, "-f", modelfile_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Ollama create failed for '{ollama_tag}'. Output:\n{proc.stdout}")

        return (ollama_tag, None)

    # 2) model_name — это тег модели в Ollama
    print(f"   Ollama: использование модели '{model_name}' (должна быть загружена: ollama pull {model_name})")
    return (model_name, None)


def generate_ollama(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 1792,
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
