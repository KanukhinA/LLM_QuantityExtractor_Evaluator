"""
Автозапуск и остановка ``vllm serve`` для ``main.py --vllm`` и ``run_all_models.py --vllm``.

По умолчанию, если модель уже есть в кэше Hugging Face Hub, подставляется путь к снапшоту и
``--served-model-name <repo id>``, в окружение процесса добавляются ``HF_HUB_OFFLINE=1`` и
``TRANSFORMERS_OFFLINE=1``, чтобы не вызывать Hub (list_repo_files / DNS). Отключить:
``VLLM_SERVE_LOCAL_SNAPSHOT=0``.

Для ``mistralai/Ministral-*`` по умолчанию добавляется ``--limit-mm-per-prompt`` с нулевыми лимитами
(эквивалент «только текст» по документации vLLM, без бага ``--language-model-only`` в 0.19.x, когда
``text_config.architectures`` остаётся None и падает загрузка весов). JSON можно задать в
``VLLM_MINISTRAL_LIMIT_MM_PER_PROMPT_JSON``. Отключить автодобавление:
``VLLM_MINISTRAL_TEXT_ONLY_MM=0`` (или устар. ``VLLM_MINISTRAL_LANGUAGE_MODEL_ONLY=0``).

Если в кэше у Mistral3 в ``text_config`` стоит ``"architectures": null`` (типично для Ministral в HF),
перед запуском в снапшоте правится ``config.json`` (как во внутренней логике vLLM для Pixtral).
Отключить: ``VLLM_SKIP_PATCH_MISTRAL3_CONFIG=1``.

Для ``mistralai/Ministral-*`` дополнительно: ``--config-format`` (по умолчанию ``hf``, см. доку vLLM),
``--load-format`` (по умолчанию ``mistral`` для consolidated safetensors). Переопределение или отмена:
``VLLM_MINISTRAL_CONFIG_FORMAT``, ``VLLM_MINISTRAL_LOAD_FORMAT`` (пусто / none / skip — не добавлять флаг).
Tool-calling (``--enable-auto-tool-choice``, ``--tool-call-parser mistral``): только при
``VLLM_MINISTRAL_TOOL_CALLING=1``.

Для ``mistralai/Ministral-*`` процессу ``vllm serve`` всегда задаётся ``VLLM_USE_V1=0`` (пока
``VLLM_MINISTRAL_PREFER_V0`` не в ``0/false``), независимо от переменных в родительском shell —
иначе наследуется ``VLLM_USE_V1=1`` и возможен SIGSEGV при инспекции ``Mistral3ForConditionalGeneration``.

PID и лог: ``OUTPUT_DIR/.vllm_autoserver.pid``, ``OUTPUT_DIR/vllm_autoserver.log``.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import shutil
import socket
import subprocess
import time
import urllib.request
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from config import OUTPUT_DIR

_VLLM_PID_FILENAME = ".vllm_autoserver.pid"
_VLLM_LOG_FILENAME = "vllm_autoserver.log"


def _local_hf_snapshot_dir(repo_id: str) -> Optional[str]:
    """
    Путь к снапшоту в HF hub cache (как utils.local_cache_path_for_model), без сетевых вызовов.
    """
    if not repo_id or not isinstance(repo_id, str) or "/" not in repo_id or os.path.isabs(repo_id):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        for filename in ("config.json", "tokenizer.json"):
            path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
            if path and os.path.isfile(path):
                d = os.path.dirname(path)
                if os.path.isfile(os.path.join(d, "config.json")):
                    return d
    except Exception:
        pass
    return None


def _extras_has_served_model_name(extras: list[str]) -> bool:
    for a in extras:
        if a == "--served-model-name" or str(a).startswith("--served-model-name="):
            return True
    return False


def _extras_has_tokenizer_mode(extras: list[str]) -> bool:
    for a in extras:
        if a == "--tokenizer-mode" or str(a).startswith("--tokenizer-mode="):
            return True
    return False


def _extras_has_config_format(extras: list[str]) -> bool:
    for a in extras:
        if a == "--config-format" or str(a).startswith("--config-format="):
            return True
    return False


def _extras_has_load_format(extras: list[str]) -> bool:
    for a in extras:
        if a == "--load-format" or str(a).startswith("--load-format="):
            return True
    return False


def _extras_has_enable_auto_tool_choice(extras: list[str]) -> bool:
    for a in extras:
        al = str(a).lower()
        if al == "--enable-auto-tool-choice" or al.startswith("--enable-auto-tool-choice="):
            return True
    return False


def _extras_has_tool_call_parser(extras: list[str]) -> bool:
    for a in extras:
        if a == "--tool-call-parser" or str(a).startswith("--tool-call-parser="):
            return True
    return False


def _env_optional_cli_value(var: str, default: str) -> Optional[str]:
    """
    Значение для опционального флага vLLM. Если переменная не задана — default.
    Пустая строка / none / skip — не добавлять флаг.
    """
    if var not in os.environ:
        return default
    v = os.environ[var].strip().lower()
    if v in ("", "none", "skip", "-", "off"):
        return None
    return os.environ[var].strip()


def _extras_has_mm_or_lm_only_flags(extras: list[str]) -> bool:
    """Пользователь задал лимиты MM, LM-only или явный отказ — не подмешиваем свои флаги."""
    for a in extras:
        al = str(a).lower()
        if al == "--limit-mm-per-prompt" or al.startswith("--limit-mm-per-prompt="):
            return True
        if al == "--language-model-only" or al.startswith("--language-model-only="):
            return True
        if al == "--no-language-model-only" or al.startswith("--no-language-model-only="):
            return True
    return False


def _ministral_auto_mm_limits_enabled() -> bool:
    if "VLLM_MINISTRAL_TEXT_ONLY_MM" in os.environ:
        return os.environ["VLLM_MINISTRAL_TEXT_ONLY_MM"].strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    if "VLLM_MINISTRAL_LANGUAGE_MODEL_ONLY" in os.environ:
        return os.environ["VLLM_MINISTRAL_LANGUAGE_MODEL_ONLY"].strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    return True


def _ministral_limit_mm_per_prompt_json() -> str:
    default = '{"image":0,"video":0}'
    raw = (os.environ.get("VLLM_MINISTRAL_LIMIT_MM_PER_PROMPT_JSON") or "").strip()
    if not raw:
        return default
    try:
        json.loads(raw)
        return raw
    except Exception:
        return default


def _patch_mistral3_text_config_architectures(snapshot_dir: str) -> bool:
    """
    vLLM 0.19: init_vllm_registered_model(hf_config=text_config) вызывает get_model_architecture;
    при ``architectures: null`` в JSON получается tuple(None) → TypeError.
    В mistral3.py для Hub-конфига это иногда чинится, для локального снапшота — дописываем в файл.
    """
    if os.environ.get("VLLM_SKIP_PATCH_MISTRAL3_CONFIG", "").strip().lower() in ("1", "true", "yes"):
        return False
    cfg_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    top_arch = data.get("architectures")
    if not isinstance(top_arch, list) or not any("Mistral3" in str(a) for a in top_arch):
        return False
    tc = data.get("text_config")
    if not isinstance(tc, dict):
        return False
    mt = (tc.get("model_type") or "").lower()
    changed = False
    # Для части снапшотов Ministral в text_config приходит model_type=ministral3,
    # а transformers ожидает базовый model_type внутренней LM (обычно "mistral").
    if mt == "ministral3":
        tc["model_type"] = "mistral"
        mt = "mistral"
        changed = True
    if tc.get("architectures") is not None and not changed:
        return False
    if mt == "mistral":
        inner = ["MistralForCausalLM"]
    elif mt == "llama":
        inner = ["LlamaForCausalLM"]
    else:
        inner = ["MistralForCausalLM"]
    if tc.get("architectures") is None:
        tc["architectures"] = inner
        changed = True
    if not changed:
        return False
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        return False
    print(
        f"   ⚙️ Патч {cfg_path}: text_config.model_type={tc.get('model_type')!r}, "
        f"text_config.architectures={tc.get('architectures')!r} "
        "(нормализация под transformers/vLLM). "
        "VLLM_SKIP_PATCH_MISTRAL3_CONFIG=1 — не менять файл."
    )
    return True


def _needs_mistral_tokenizer_mode(model_id: str) -> bool:
    s = (model_id or "").strip().lower()
    return s.startswith("mistralai/ministral-") or s.startswith("mistralai/mistral-")


def _is_mistralai_ministral_line(model_id: str) -> bool:
    """Линейка Ministral-3 (Mistral3): нужны нулевые MM-лимиты или отдельные костыли vLLM для текстового режима."""
    s = (model_id or "").strip().lower()
    return s.startswith("mistralai/ministral-")


def _resolve_vllm_serve_args(model_id: str) -> tuple[str, list[str], dict[str, str]]:
    """
    Если в кэше HF уже есть снапшот — передаём в vLLM локальный путь и выставляем offline для Hub,
    чтобы не было list_repo_files / DNS к huggingface.co. Имя в API — исходный repo id
    (через --served-model-name), чтобы совпадало с models.yaml и check_vllm_models.

    Отключить: VLLM_SERVE_LOCAL_SNAPSHOT=0
    """
    logical = (model_id or "").strip()
    if not logical:
        return logical, [], {}

    if os.path.isdir(logical) and os.path.isfile(os.path.join(logical, "config.json")):
        return logical, [], {}

    use_local = os.environ.get("VLLM_SERVE_LOCAL_SNAPSHOT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if not use_local:
        return logical, [], {}

    snap = _local_hf_snapshot_dir(logical)
    if not snap:
        return logical, [], {}

    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return snap, ["--served-model-name", logical], env


def vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _vllm_host_port() -> tuple[str, int]:
    parsed = urlparse(vllm_base_url())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    return host, int(port)


def _tcp_port_has_listener(host: str, port: int) -> bool:
    """True, если на host:port кто-то принимает TCP (старый vLLM ещё держит порт)."""
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def wait_until_vllm_port_free(host: str, port: int, timeout_sec: Optional[float] = None) -> None:
    """
    После SIGTERM дочерние процессы vLLM могут отпустить порт с задержкой.
    Ждём, пока TCP-порт перестанет принимать соединения, затем запускаем новый сервер.
    """
    if timeout_sec is None:
        try:
            timeout_sec = float(os.environ.get("VLLM_PORT_FREE_WAIT_SEC", "90"))
        except ValueError:
            timeout_sec = 90.0
    timeout_sec = max(5.0, timeout_sec)
    deadline = time.time() + timeout_sec
    if not _tcp_port_has_listener(host, port):
        return
    print("   ⏳ Жду освобождения порта после остановки предыдущего vLLM...")
    while time.time() < deadline:
        if not _tcp_port_has_listener(host, port):
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"Порт {host}:{port} занят дольше {timeout_sec:.0f} с после остановки сервера. "
        f"Завершите процесс вручную или увеличьте VLLM_PORT_FREE_WAIT_SEC / смените VLLM_BASE_URL."
    )


def _vllm_pid_path() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, _VLLM_PID_FILENAME)


def _write_vllm_pid(pid: int, model_id: str) -> None:
    try:
        with open(_vllm_pid_path(), "w", encoding="utf-8") as f:
            json.dump({"pid": int(pid), "model": model_id}, f, ensure_ascii=False)
    except Exception:
        pass


def clear_vllm_pid_file() -> None:
    try:
        if os.path.exists(_vllm_pid_path()):
            os.remove(_vllm_pid_path())
    except Exception:
        pass


def _vllm_log_path() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, _VLLM_LOG_FILENAME)


def _read_log_tail(path: str, max_bytes: int = 12000, max_lines: int = 40) -> str:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return "(лог пуст или отсутствует)"
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            raw = f.read().decode("utf-8", errors="replace")
        lines = raw.strip().splitlines()
        tail = "\n".join(lines[-max_lines:])
        return tail if tail else raw.strip()[:max_bytes]
    except Exception as exc:
        return f"(не удалось прочитать лог: {exc})"


def normalize_vllm_serve_extra_args(raw: Any) -> list[str]:
    """Аргументы командной строки для ``vllm serve`` из models.yaml (список строк или одна строка для shlex)."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        import shlex

        return [x for x in shlex.split(raw, posix=os.name != "nt") if x]
    s = str(raw).strip()
    return [s] if s else []


def _extras_has_quantization_flag(extras: list[str]) -> bool:
    for a in extras:
        al = str(a).lower()
        if al == "--quantization" or al.startswith("--quantization="):
            return True
    return False


def build_vllm_serve_extra_args(hyperparameters: Optional[Mapping[str, Any]]) -> list[str]:
    """
    Нормализует и дополняет аргументы для `vllm serve`:
    1) `vllm_serve_extra_args` (как есть),
    2) если не задан `--quantization`, добавляет из `vllm_quantization` (только явно).

    ``vllm_quant_tag`` используется в проекте для имён папок результатов и не подставляет
    ``--quantization`` автоматически: для обычного HF-чекпойнта (bf16/fp16) флаг ``awq``/``gptq``
    ломает загрузку; квантизация задаётся только если веса реально в этом формате.
    """
    hp = dict(hyperparameters or {})
    extras = normalize_vllm_serve_extra_args(hp.get("vllm_serve_extra_args"))
    if _extras_has_quantization_flag(extras):
        return extras
    explicit_quant = (hp.get("vllm_quantization") or "").strip()
    if explicit_quant and explicit_quant.lower() in ("none", "null", "off", "-", "skip"):
        return extras
    if explicit_quant:
        return ["--quantization", explicit_quant] + extras
    return extras


def vllm_autoserver_fingerprint(model_id: str, hyperparameters: Optional[Mapping[str, Any]]) -> tuple[str, tuple[str, ...]]:
    """Одинаковый отпечаток — тот же процесс serve, можно не перезапускать."""
    extra = build_vllm_serve_extra_args(hyperparameters)
    return ((model_id or "").strip(), tuple(extra))


_VLLM_FAIL_HINT = (
    "Полная причина часто выше показанного фрагмента — откройте лог и найдите первый Traceback/CUDA/OOM. "
    "Типично: нехватка VRAM, несовместимость движка v1 (попробуйте окружение VLLM_USE_V1=0), "
    "или доп. флаги в hyperparameters.vllm_serve_extra_args (см. README, разд. 2.5)."
)


def default_vllm_ready_timeout_sec() -> int:
    try:
        return max(30, int(os.environ.get("VLLM_READY_TIMEOUT_SEC", "600")))
    except ValueError:
        return 600


def _wait_vllm_ready(proc: subprocess.Popen, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    url = f"{vllm_base_url()}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def terminate_vllm_process(proc: Optional[subprocess.Popen], reason: str = "") -> None:
    if proc is None:
        return
    log_fp = getattr(proc, "_vllm_log_fp", None)
    try:
        if proc.poll() is None:
            if reason:
                print(f"   ⚠️ Останавливаю vLLM автосервер ({reason})...")
            # Вся группа процессов (Linux/macOS): дочерние worker/engine vLLM обычно в той же группе при start_new_session.
            if os.name != "nt":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                else:
                    proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass
            try:
                del proc._vllm_log_fp
            except Exception:
                pass


def kill_stale_autovllm_if_any() -> None:
    path = _vllm_pid_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data.get("pid", 0))
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass
    clear_vllm_pid_file()


def start_vllm_autoserver(
    model_id: str,
    ready_timeout_sec: int,
    extra_args: Optional[list[str]] = None,
) -> subprocess.Popen:
    if not shutil.which("vllm"):
        raise RuntimeError("Команда 'vllm' не найдена в PATH. Установите vllm и повторите запуск.")
    host, port = _vllm_host_port()
    wait_until_vllm_port_free(host, port)
    extras = normalize_vllm_serve_extra_args(extra_args)
    serve_arg, local_extra, env_updates = _resolve_vllm_serve_args(model_id)
    if os.path.isdir(serve_arg) and os.path.isfile(os.path.join(serve_arg, "config.json")):
        _patch_mistral3_text_config_architectures(serve_arg)
    if _extras_has_served_model_name(extras):
        local_extra = []
    if _needs_mistral_tokenizer_mode(model_id) and not _extras_has_tokenizer_mode(extras):
        # Для Ministral/Mistral в vLLM 0.19.x это снижает риск падения на tokenizer-конвертации.
        extras = ["--tokenizer-mode", "mistral"] + extras
    if _is_mistralai_ministral_line(model_id):
        cf = _env_optional_cli_value("VLLM_MINISTRAL_CONFIG_FORMAT", "hf")
        if cf and not _extras_has_config_format(extras):
            extras = ["--config-format", cf] + extras
        lf = _env_optional_cli_value("VLLM_MINISTRAL_LOAD_FORMAT", "mistral")
        if lf and not _extras_has_load_format(extras):
            extras = ["--load-format", lf] + extras
        if os.environ.get("VLLM_MINISTRAL_TOOL_CALLING", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            if not _extras_has_enable_auto_tool_choice(extras):
                extras = ["--enable-auto-tool-choice"] + extras
            if not _extras_has_tool_call_parser(extras):
                extras = ["--tool-call-parser", "mistral"] + extras
    if (
        _is_mistralai_ministral_line(model_id)
        and _ministral_auto_mm_limits_enabled()
        and not _extras_has_mm_or_lm_only_flags(extras)
    ):
        # Эквивалент «только текст» по доке vLLM; --language-model-only в 0.19.x ломает Mistral3 (text_config.architectures=None).
        mm_json = _ministral_limit_mm_per_prompt_json()
        extras = ["--limit-mm-per-prompt", mm_json] + extras
        print(
            "   ⚙️ Ministral-3: --limit-mm-per-prompt "
            f"{mm_json!r} (текст без vision; при сбое задайте VLLM_MINISTRAL_TEXT_ONLY_MM=0 "
            "или свой JSON в VLLM_MINISTRAL_LIMIT_MM_PER_PROMPT_JSON)"
        )
    # --host/--port в конце: у части версий vLLM иначе парсер рвёт цепочку опций после пути к модели.
    cmd = (
        ["vllm", "serve", serve_arg]
        + local_extra
        + extras
        + ["--host", host, "--port", str(port)]
    )
    log_path = _vllm_log_path()
    if env_updates:
        print(f"   📁 vLLM: локальный снапшот HF → {serve_arg}")
        print(
            f"      Имя в API: «{model_id.strip()}» (--served-model-name); "
            "HF_HUB_OFFLINE=1 — без запросов к Hugging Face Hub"
        )
    print(f"   🚀 Запуск vLLM автосервера: {' '.join(cmd)}")
    print(f"   📋 Та же команда для shell (с кавычками): {shlex.join(cmd)}")
    print(f"   📝 Лог stdout/stderr vLLM: {log_path}")
    print(f"   ⏱ Ожидание готовности до {ready_timeout_sec} с (переменная окружения VLLM_READY_TIMEOUT_SEC)")
    log_fp = open(log_path, "a", encoding="utf-8")
    try:
        log_fp.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')} vllm serve {model_id}\n")
        if serve_arg != (model_id or "").strip():
            log_fp.write(f"(локальный путь: {serve_arg})\n")
        log_fp.write(" ".join(cmd) + "\n")
        log_fp.write(f"# shell (с кавычками): {shlex.join(cmd)}\n")
        log_fp.flush()
        child_env = os.environ.copy()
        child_env.update(env_updates)
        if _is_mistralai_ministral_line(model_id):
            if os.environ.get("VLLM_MINISTRAL_PREFER_V0", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            ):
                # Всегда для дочернего vllm: родитель мог экспортировать VLLM_USE_V1=1.
                child_env["VLLM_USE_V1"] = "0"
                print(
                    "   ⚙️ Ministral-3: процессу vLLM задано VLLM_USE_V1=0 "
                    "(снижает риск SIGSEGV; отключить: VLLM_MINISTRAL_PREFER_V0=0)"
                )
        popen_kw: dict = {
            "stdout": log_fp,
            "stderr": subprocess.STDOUT,
            "close_fds": False,
            "env": child_env,
        }
        if os.name != "nt":
            popen_kw["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kw)
        proc._vllm_log_fp = log_fp
    except Exception:
        try:
            log_fp.close()
        except Exception:
            pass
        raise
    _write_vllm_pid(proc.pid, model_id)
    if proc.poll() is not None:
        clear_vllm_pid_file()
        tail = _read_log_tail(log_path, max_bytes=48000, max_lines=120)
        raise RuntimeError(
            f"vLLM процесс сразу завершился (код {proc.returncode}). См. {log_path}\n"
            f"{_VLLM_FAIL_HINT}\n{tail}"
        )
    if not _wait_vllm_ready(proc, timeout_sec=ready_timeout_sec):
        code = proc.poll()
        terminate_vllm_process(proc, reason="сервер не поднялся вовремя")
        clear_vllm_pid_file()
        tail = _read_log_tail(log_path, max_bytes=48000, max_lines=120)
        if code is not None:
            raise RuntimeError(
                f"vLLM процесс завершился (код {code}) до готовности. См. {log_path}\n"
                f"{_VLLM_FAIL_HINT}\n{tail}"
            )
        raise RuntimeError(
            f"vLLM сервер не поднялся за {ready_timeout_sec} с: {vllm_base_url()}. "
            f"Увеличьте VLLM_READY_TIMEOUT_SEC. {_VLLM_FAIL_HINT}\n{log_path}\n{tail}"
        )
    print(f"   ✅ vLLM автосервер готов: {vllm_base_url()}")
    return proc
