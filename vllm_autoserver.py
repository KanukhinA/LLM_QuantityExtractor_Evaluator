"""
Автозапуск и остановка ``vllm serve`` для ``main.py --vllm`` и ``run_all_models.py --vllm``.

PID и лог: ``OUTPUT_DIR/.vllm_autoserver.pid``, ``OUTPUT_DIR/vllm_autoserver.log``.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from config import OUTPUT_DIR

_VLLM_PID_FILENAME = ".vllm_autoserver.pid"
_VLLM_LOG_FILENAME = "vllm_autoserver.log"


def vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _vllm_host_port() -> tuple[str, int]:
    parsed = urlparse(vllm_base_url())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    return host, int(port)


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


def _read_log_tail(path: str, max_bytes: int = 12000) -> str:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return "(лог пуст или отсутствует)"
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            raw = f.read().decode("utf-8", errors="replace")
        lines = raw.strip().splitlines()
        tail = "\n".join(lines[-40:])
        return tail if tail else raw.strip()[:max_bytes]
    except Exception as exc:
        return f"(не удалось прочитать лог: {exc})"


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
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
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


def start_vllm_autoserver(model_id: str, ready_timeout_sec: int) -> subprocess.Popen:
    if not shutil.which("vllm"):
        raise RuntimeError("Команда 'vllm' не найдена в PATH. Установите vllm и повторите запуск.")
    host, port = _vllm_host_port()
    cmd = ["vllm", "serve", model_id, "--host", host, "--port", str(port)]
    log_path = _vllm_log_path()
    print(f"   🚀 Запуск vLLM автосервера: {' '.join(cmd)}")
    print(f"   📝 Лог stdout/stderr vLLM: {log_path}")
    print(f"   ⏱ Ожидание готовности до {ready_timeout_sec} с (переменная окружения VLLM_READY_TIMEOUT_SEC)")
    log_fp = open(log_path, "a", encoding="utf-8")
    try:
        log_fp.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')} vllm serve {model_id}\n")
        log_fp.write(" ".join(cmd) + "\n")
        log_fp.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            close_fds=False,
        )
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
        tail = _read_log_tail(log_path)
        raise RuntimeError(
            f"vLLM процесс сразу завершился (код {proc.returncode}). См. {log_path}\n{tail}"
        )
    if not _wait_vllm_ready(proc, timeout_sec=ready_timeout_sec):
        code = proc.poll()
        terminate_vllm_process(proc, reason="сервер не поднялся вовремя")
        clear_vllm_pid_file()
        tail = _read_log_tail(log_path)
        if code is not None:
            raise RuntimeError(
                f"vLLM процесс завершился (код {code}) до готовности. См. {log_path}\n{tail}"
            )
        raise RuntimeError(
            f"vLLM сервер не поднялся за {ready_timeout_sec} с: {vllm_base_url()}. "
            f"Увеличьте VLLM_READY_TIMEOUT_SEC. См. лог:\n{log_path}\n{tail}"
        )
    print(f"   ✅ vLLM автосервер готов: {vllm_base_url()}")
    return proc
