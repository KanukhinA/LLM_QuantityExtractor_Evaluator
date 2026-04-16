"""
Автозапуск и остановка ``vllm serve`` для ``main.py --vllm`` и ``run_all_models.py --vllm``.

По умолчанию, если модель уже есть в кэше Hugging Face Hub, подставляется путь к снапшоту и
``--served-model-name <repo id>``, в окружение процесса добавляются ``HF_HUB_OFFLINE=1`` и
``TRANSFORMERS_OFFLINE=1``, чтобы не вызывать Hub (list_repo_files / DNS). Отключить:
``VLLM_SERVE_LOCAL_SNAPSHOT=0``.

Для ``mistralai/Ministral-*`` (Mistral3) при отсутствии ``VLLM_USE_V1`` в окружении выставляется
``VLLM_USE_V1=0``, чтобы обойти падения v1 на PixtralProcessor при текстовом инференсе.

PID и лог: ``OUTPUT_DIR/.vllm_autoserver.pid``, ``OUTPUT_DIR/vllm_autoserver.log``.
"""
from __future__ import annotations

import json
import os
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


def _needs_mistral_tokenizer_mode(model_id: str) -> bool:
    s = (model_id or "").strip().lower()
    return s.startswith("mistralai/ministral-") or s.startswith("mistralai/mistral-")


def _is_mistralai_ministral_line(model_id: str) -> bool:
    """Ministral-3 (Mistral3ForConditionalGeneration): в vLLM 0.19 v1 может падать на PixtralProcessor."""
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


def vllm_autoserver_fingerprint(model_id: str, hyperparameters: Optional[Mapping[str, Any]]) -> tuple[str, tuple[str, ...]]:
    """Одинаковый отпечаток — тот же процесс serve, можно не перезапускать."""
    hp = dict(hyperparameters or {})
    extra = normalize_vllm_serve_extra_args(hp.get("vllm_serve_extra_args"))
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
    if _extras_has_served_model_name(extras):
        local_extra = []
    if _needs_mistral_tokenizer_mode(model_id) and not _extras_has_tokenizer_mode(extras):
        # Для Ministral/Mistral в vLLM 0.19.x это снижает риск падения на tokenizer-конвертации.
        extras = ["--tokenizer-mode", "mistral"] + extras
    cmd = ["vllm", "serve", serve_arg] + local_extra + ["--host", host, "--port", str(port)]
    cmd.extend(extras)
    log_path = _vllm_log_path()
    if env_updates:
        print(f"   📁 vLLM: локальный снапшот HF → {serve_arg}")
        print(
            f"      Имя в API: «{model_id.strip()}» (--served-model-name); "
            "HF_HUB_OFFLINE=1 — без запросов к Hugging Face Hub"
        )
    print(f"   🚀 Запуск vLLM автосервера: {' '.join(cmd)}")
    print(f"   📝 Лог stdout/stderr vLLM: {log_path}")
    print(f"   ⏱ Ожидание готовности до {ready_timeout_sec} с (переменная окружения VLLM_READY_TIMEOUT_SEC)")
    log_fp = open(log_path, "a", encoding="utf-8")
    try:
        log_fp.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')} vllm serve {model_id}\n")
        if serve_arg != (model_id or "").strip():
            log_fp.write(f"(локальный путь: {serve_arg})\n")
        log_fp.write(" ".join(cmd) + "\n")
        log_fp.flush()
        child_env = os.environ.copy()
        child_env.update(env_updates)
        if _is_mistralai_ministral_line(model_id) and "VLLM_USE_V1" not in os.environ:
            child_env["VLLM_USE_V1"] = "0"
            print(
                "   ⚙️ Ministral-3: VLLM_USE_V1=0 (обход падения v1 на PixtralProcessor; "
                "задайте VLLM_USE_V1 в окружении, если нужен явно v1)"
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
