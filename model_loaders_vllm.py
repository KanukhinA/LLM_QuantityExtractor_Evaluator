"""
Загрузка и генерация через vLLM (OpenAI-совместимый HTTP API).

Ожидается отдельно запущенный сервер, например:
  vllm serve <HF_MODEL_ID> --quantization awq   # или gptq / bitsandbytes — см. документацию vLLM

Базовый URL: переменная окружения VLLM_BASE_URL (по умолчанию http://127.0.0.1:8000).
Имя модели в запросах — то же, что передано в ``vllm serve`` (поле ``name`` или ``vllm_name`` в models.yaml).

Квантизация (Q4 и т.д.) задаётся при запуске vLLM, не этим модулем.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional, Tuple


def _vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _vllm_api_key() -> Optional[str]:
    return os.environ.get("VLLM_API_KEY") or None


def load_vllm(served_model_id: str) -> Tuple[Dict[str, Any], None]:
    """
    «Загрузка»: весов нет, возвращаем словарь для generate_vllm и tokenizer=None.

    Args:
        served_model_id: идентификатор модели на сервере vLLM (как в ``vllm serve``).
    """
    base = _vllm_base_url()
    sid = (served_model_id or "").strip()
    if not sid:
        raise ValueError("vLLM: пустой идентификатор модели (vllm_name / name в models.yaml)")
    print(f"   vLLM: инференс через {base} (модель «{sid}»). Запустите сервер: vllm serve <id> ...")
    return (
        {
            "vllm": True,
            "base_url": base,
            "served_model_id": sid,
        },
        None,
    )


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return default


def generate_vllm(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 1792,
    hyperparameters: Optional[Mapping[str, Any]] = None,
    repetition_penalty: Any = None,
    max_length: Any = None,
    **kwargs,
) -> str:
    """
    POST /v1/chat/completions (OpenAI-совместимый API vLLM).

    Параметры выборки и шаблона чата согласованы с локальной генерацией
    (``_make_generation_config`` / ``generate_qwen_3``) и с ``build_ollama_options``:

    - ``do_sample``, ``temperature``, ``top_p``, ``top_k`` — как при локальном запуске;
    - ``repetition_penalty`` — в тело запроса (SamplingParams в vLLM);
    - ``enable_thinking`` — если ключ есть в ``models.yaml``, в ``chat_template_kwargs``
      (для Qwen3 и др., см. документацию vLLM по ``chat_template_kwargs``).

    Поля только для загрузки весов (``torch_dtype``, ``dtype``, ``vllm_quant_tag``) не передаются в API.
    """
    if not isinstance(model, dict) or not model.get("vllm"):
        raise TypeError("generate_vllm: ожидается model от load_vllm (dict с ключами vllm, base_url, served_model_id)")

    base_url = str(model.get("base_url") or _vllm_base_url()).rstrip("/")
    served_id = str(model.get("served_model_id") or "").strip()
    if not served_id:
        raise ValueError("generate_vllm: пустой served_model_id")

    hp: Dict[str, Any] = dict(hyperparameters or {})
    if repetition_penalty is not None:
        hp["repetition_penalty"] = repetition_penalty
    if max_length is not None:
        hp["max_length"] = max_length
    for k in (
        "vllm",
        "ollama",
        "api_model",
        "structured_output",
        "use_outlines",
        "use_guidance",
        "pydantic_outlines",
        "multi_agent_mode",
        "prompt_template_name",
        "vllm_quant_tag",
        "torch_dtype",
        "dtype",
    ):
        hp.pop(k, None)

    do_sample = _coerce_bool(hp.get("do_sample"), False)
    body: Dict[str, Any] = {
        "model": served_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_new_tokens),
    }
    if do_sample:
        body["temperature"] = float(hp.get("temperature", 0.8))
        if hp.get("top_p") is not None:
            body["top_p"] = float(hp["top_p"])
        if hp.get("top_k") is not None:
            body["top_k"] = int(hp["top_k"])
    else:
        body["temperature"] = 0.0

    rp = hp.get("repetition_penalty")
    if rp is not None:
        try:
            body["repetition_penalty"] = float(rp)
        except (TypeError, ValueError):
            pass

    if "enable_thinking" in hp:
        body["chat_template_kwargs"] = {
            "enable_thinking": _coerce_bool(hp.get("enable_thinking"), False),
        }

    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = _vllm_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            err_obj = json.loads(raw) if raw else {}
            msg = err_obj.get("error", {}).get("message") if isinstance(err_obj.get("error"), dict) else err_obj.get("message", raw)
            if not msg:
                msg = raw or str(e)
        except Exception:
            msg = raw or str(e)
        raise RuntimeError(f"vLLM API error: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"vLLM недоступен ({base_url}): {e.reason}") from e

    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0].get("message") or {})
    content = msg.get("content")
    if content is None:
        return ""
    return str(content).strip()
