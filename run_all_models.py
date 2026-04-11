"""
Скрипт для запуска оценки всех моделей подряд
"""
import os
import sys
import argparse
import logging
import json
import time
import signal
import shutil
import subprocess
import urllib.request
import urllib.error
from typing import Optional
from urllib.parse import urlparse
from main import run_evaluation
from gemini_analyzer import check_gemini_api
from config import GEMINI_API_KEY, MODEL_CONFIGS, OUTPUT_DIR
from utils import ConsoleLogCapture
from model_evaluator import StopAllModelsInterrupt, _append_to_model_errors_log


_VLLM_PID_FILENAME = ".vllm_autoserver.pid"
_VLLM_LOG_FILENAME = "vllm_autoserver.log"


def _vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _vllm_host_port() -> tuple[str, int]:
    parsed = urlparse(_vllm_base_url())
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


def _clear_vllm_pid() -> None:
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


def _default_vllm_ready_timeout_sec() -> int:
    try:
        return max(30, int(os.environ.get("VLLM_READY_TIMEOUT_SEC", "600")))
    except ValueError:
        return 600


def _resolve_vllm_ready_timeout(override: Optional[int]) -> int:
    if override is not None:
        return max(30, int(override))
    return _default_vllm_ready_timeout_sec()


def _wait_vllm_ready(proc: subprocess.Popen, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    url = f"{_vllm_base_url()}/v1/models"
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


def _terminate_process(proc: Optional[subprocess.Popen], reason: str = "") -> None:
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


def _kill_stale_autovllm_if_any() -> None:
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
    _clear_vllm_pid()


def _start_vllm_autoserver(model_id: str, ready_timeout_sec: int) -> subprocess.Popen:
    if not shutil.which("vllm"):
        raise RuntimeError("Команда 'vllm' не найдена в PATH. Установите vllm и повторите запуск.")
    host, port = _vllm_host_port()
    cmd = ["vllm", "serve", model_id, "--host", host, "--port", str(port)]
    log_path = _vllm_log_path()
    print(f"   🚀 Запуск vLLM автосервера: {' '.join(cmd)}")
    print(f"   📝 Лог stdout/stderr vLLM: {log_path}")
    print(f"   ⏱ Ожидание готовности до {ready_timeout_sec} с (переменная VLLM_READY_TIMEOUT_SEC или --vllm-ready-timeout)")
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
        _clear_vllm_pid()
        tail = _read_log_tail(log_path)
        raise RuntimeError(
            f"vLLM процесс сразу завершился (код {proc.returncode}). См. {log_path}\n{tail}"
        )
    if not _wait_vllm_ready(proc, timeout_sec=ready_timeout_sec):
        code = proc.poll()
        _terminate_process(proc, reason="сервер не поднялся вовремя")
        _clear_vllm_pid()
        tail = _read_log_tail(log_path)
        if code is not None:
            raise RuntimeError(
                f"vLLM процесс завершился (код {code}) до готовности. См. {log_path}\n{tail}"
            )
        raise RuntimeError(
            f"vLLM сервер не поднялся за {ready_timeout_sec} с: {_vllm_base_url()}. "
            f"Увеличьте VLLM_READY_TIMEOUT_SEC или --vllm-ready-timeout. См. лог:\n{log_path}\n{tail}"
        )
    print(f"   ✅ vLLM автосервер готов: {_vllm_base_url()}")
    return proc

def run_all_models(local_only: bool = False, multi_agent_mode: str = None,
                   structured_output: bool = False, use_outlines: bool = False,
                   prompt_template_name: str = None, pydantic_outlines: bool = False,
                   use_guidance: bool = False, ollama_only: bool = False, vllm_only: bool = False,
                   vllm_ready_timeout_sec: Optional[int] = None):
    """Запускает оценку всех моделей из конфигурации"""
    # Проверяем работоспособность Gemini API в самом начале
    print(f"\n{'='*80}")
    print(f"ПРОВЕРКА СИСТЕМЫ")
    print(f"{'='*80}")
    # GEMINI_API_KEY загружается из config.py (который берет его из config_secrets.py или переменных окружения)
    
    if GEMINI_API_KEY:
        print(f"Проверка работоспособности Gemini API...")
        gemini_working, gemini_message = check_gemini_api(GEMINI_API_KEY)
        print(f"   {gemini_message}\n")
    else:
        print(f"GEMINI_API_KEY не установлен, пропускаем проверку API")
        gemini_working = False
        print()
    
    use_gemini = True
    if not gemini_working:
        print(f"{'='*80}")
        print(f"ВНИМАНИЕ: Gemini API недоступен")
        print(f"{'='*80}")
        print(f"Оценка будет выполнена без анализа ошибок через Gemini.\n")
        use_gemini = False
    
    # Выводим информацию о режимах
    if multi_agent_mode:
        print(f"📌 Режим: Мультиагентный ({multi_agent_mode})")
    else:
        print(f"📌 Режим: Одноагентный")
    if structured_output:
        print(f"📌 Structured Output: Включен (Pydantic валидация)")
    if use_outlines or pydantic_outlines:
        print(f"📌 Outlines: Включен" + (" (схема из Pydantic)" if pydantic_outlines else " (outlines_schema.py)"))
    if use_guidance:
        effective_prompt = prompt_template_name or "DETAILED_INSTR_ZEROSHOT_CD_RUS"
        print(f"📌 Guidance (llguidance): Включен, промпт: {effective_prompt}")
    if prompt_template_name:
        print(f"📌 Промпт: {prompt_template_name}")
    print()
    
    # Фильтруем модели, если указан флаг local_only / ollama_only / vllm_only
    all_models = list(MODEL_CONFIGS.keys())
    if vllm_only:
        models = [model_key for model_key in all_models if "-vllm" in model_key]
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ VLLM-МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"Всего моделей в конфигурации: {len(all_models)}")
        print(f"vLLM-моделей: {len(models)}")
    elif ollama_only:
        # Ollama-версии автоматически добавляются в model_config_loader как <base_key>-ollama
        models = [model_key for model_key in all_models if "-ollama" in model_key]
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ OLLAMA-МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"Всего моделей в конфигурации: {len(all_models)}")
        print(f"Ollama-моделей: {len(models)}")
    elif local_only:
        # Локальные модели - это те, у которых нет "-api" в ключе
        models = [model_key for model_key in all_models if "-api" not in model_key]
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ ЛОКАЛЬНЫХ МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"Всего моделей в конфигурации: {len(all_models)}")
        print(f"Локальных моделей: {len(models)}")
        print(f"API моделей (пропущено): {len(all_models) - len(models)}")
    else:
        models = all_models
        print(f"\n{'='*80}")
        print(f"ЗАПУСК ОЦЕНКИ ВСЕХ МОДЕЛЕЙ")
        print(f"{'='*80}")
        print(f"Количество моделей: {len(models)}")
    
    if not models:
        print(f"⚠️  Не найдено моделей для оценки.")
        if local_only:
            print(f"   Попробуйте запустить без флага --local-only для оценки всех моделей.")
        if vllm_only:
            print(f"   Убедитесь, что в model_config_loader сгенерированы ключи *-vllm (перезапустите после правки models.yaml).")
        return
    
    print(f"Модели: {', '.join(models)}\n")
    
    results_summary = []
    vllm_proc = None
    active_vllm_model = None
    vllm_ready_timeout = _resolve_vllm_ready_timeout(vllm_ready_timeout_sec)
    _kill_stale_autovllm_if_any()

    try:
        for i, model_key in enumerate(models, 1):
            print(f"\n{'='*80}")
            print(f"Модель {i}/{len(models)}: {model_key}")
            print(f"{'='*80}\n")
            try:
                import copy
                config = copy.deepcopy(MODEL_CONFIGS[model_key])
                if multi_agent_mode:
                    config["hyperparameters"]["multi_agent_mode"] = multi_agent_mode
                if structured_output:
                    config["hyperparameters"]["structured_output"] = True
                if use_outlines or pydantic_outlines:
                    config["hyperparameters"]["use_outlines"] = True
                    config["hyperparameters"]["pydantic_outlines"] = pydantic_outlines
                if use_guidance:
                    config["hyperparameters"]["use_guidance"] = True
                    config["hyperparameters"]["prompt_template_name"] = (
                        prompt_template_name or "DETAILED_INSTR_ZEROSHOT_CD_RUS"
                    )
                elif prompt_template_name is not None:
                    config["hyperparameters"]["prompt_template_name"] = prompt_template_name

                is_vllm_model = bool((config.get("hyperparameters") or {}).get("vllm", False))
                if is_vllm_model:
                    served_model_id = (config.get("name") or "").strip()
                    if not served_model_id:
                        raise RuntimeError("Пустой name/vllm_name для vLLM модели")
                    if (vllm_proc is None) or (vllm_proc.poll() is not None) or (active_vllm_model != served_model_id):
                        _terminate_process(vllm_proc, reason="переключение на следующую vLLM модель")
                        _clear_vllm_pid()
                        vllm_proc = _start_vllm_autoserver(served_model_id, ready_timeout_sec=vllm_ready_timeout)
                        active_vllm_model = served_model_id
                else:
                    _terminate_process(vllm_proc, reason="запуск не-vLLM модели")
                    _clear_vllm_pid()
                    vllm_proc = None
                    active_vllm_model = None

                result = run_evaluation(
                    config,
                    model_key=model_key,
                    use_gemini=use_gemini,
                    verbose=False,
                    stop_all_on_interrupt=True,
                )

                if result.get("status") == "interrupted":
                    print(f"Модель {model_key} прервана (без сохранения)\n")
                    results_summary.append({
                        "model": model_key,
                        "status": "interrupted",
                        "message": result.get("message", "Прервано пользователем")
                    })
                elif result.get("status") != "error":
                    if result.get("interrupted") and result.get("timeout_reason"):
                        timeout_reason = result.get("timeout_reason")
                        print(f"Модель {model_key} прервана: {timeout_reason}\n")
                        results_summary.append({
                            "model": model_key,
                            "status": "timeout",
                            "timeout_reason": timeout_reason
                        })
                    elif result.get("interrupted"):
                        print(f"Модель {model_key} прервана (без сохранения)\n")
                        results_summary.append({
                            "model": model_key,
                            "status": "interrupted",
                            "message": "Прервано пользователем"
                        })
                    else:
                        results_summary.append({
                            "model": model_key,
                            "status": "success",
                            "multi_agent_mode": result.get("multi_agent_mode"),
                            "avg_speed": result.get("average_response_time_seconds"),
                            "parsing_error_rate": result.get("parsing_error_rate"),
                            "memory_gb": result.get("gpu_memory_during_inference_gb")
                        })
                        print(f"Модель {model_key} успешно оценена\n")
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"Модель {model_key} пропущена из-за ошибки: {error_msg}\n")
                    results_summary.append({
                        "model": model_key,
                        "status": "error",
                        "error": error_msg
                    })
            except (KeyboardInterrupt, StopAllModelsInterrupt):
                print(f"\nПрервано пользователем. Остановка оценки моделей.")
                break
            except Exception as e:
                import traceback
                error_msg = str(e)
                tb_str = traceback.format_exc()
                print(f"Критическая ошибка при оценке {model_key}: {error_msg}")
                print(f"   Детали: {tb_str[:500]}...")
                print(f"Модель {model_key} пропущена\n")
                full_msg = f"{error_msg}\n\n{tb_str}"
                _append_to_model_errors_log(
                    OUTPUT_DIR,
                    "Критическая ошибка при оценке модели (исключение в run_evaluation)",
                    model_key,
                    full_msg,
                )
                logging.error("Оценка модели %s: %s", model_key, error_msg)
                results_summary.append({
                    "model": model_key,
                    "status": "error",
                    "error": error_msg
                })
    finally:
        _terminate_process(vllm_proc, reason="завершение run_all_models")
        _clear_vllm_pid()
    
    # Выводим итоговую сводку
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СВОДКА")
    print(f"{'='*80}\n")
    
    successful = [s for s in results_summary if s['status'] == 'success']
    failed = [s for s in results_summary if s['status'] == 'error']
    timeout_models = [s for s in results_summary if s['status'] == 'timeout']
    interrupted_models = [s for s in results_summary if s['status'] == 'interrupted']
    
    print(f"Общая статистика:")
    print(f"   • Всего моделей: {len(results_summary)}")
    print(f"   • Успешно оценено: {len(successful)}")
    print(f"   • Пропущено из-за ошибок: {len(failed)}")
    print(f"   • Прервано по времени: {len(timeout_models)}")
    print(f"   • Прервано пользователем: {len(interrupted_models)}")
    print()
    
    if timeout_models:
        print(f"МОДЕЛИ, ПРЕРВАННЫЕ ИЗ-ЗА ПРЕВЫШЕНИЯ ВРЕМЕНИ ИНФЕРЕНСА:")
        for summary in timeout_models:
            print(f"   • {summary['model']}: {summary.get('timeout_reason', 'Превышен лимит времени')}")
        print()
    
    if interrupted_models:
        print(f"МОДЕЛИ, ПРЕРВАННЫЕ ПОЛЬЗОВАТЕЛЕМ (без сохранения):")
        for summary in interrupted_models:
            print(f"   • {summary['model']}: {summary.get('message', 'Прервано')}")
        print()
    
    if successful:
        print(f"УСПЕШНО ОЦЕНЕННЫЕ МОДЕЛИ:")
        for summary in successful:
            print(f"   • {summary['model']}")
            mode = summary.get('multi_agent_mode') or 'Одноагентный'
            print(f"     - Режим: {mode}")
            print(f"     - Скорость: {summary['avg_speed']:.3f} сек/ответ")
            print(f"     - Ошибки парсинга: {summary['parsing_error_rate']:.2%}")
            print(f"     - Память: {summary['memory_gb']:.2f} GB")
        print()
    
    if failed:
        print(f"ПРОПУЩЕННЫЕ МОДЕЛИ:")
        for summary in failed:
            print(f"   • {summary['model']}: {summary.get('error', 'Unknown error')[:100]}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск оценки всех моделей")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Запустить оценку только для локальных моделей (исключить API модели)"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Запустить оценку только для ключей *-ollama",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Запустить оценку только для ключей *-vllm",
    )
    parser.add_argument(
        "--vllm-ready-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="Секунд ожидания готовности vLLM после vllm serve (по умолчанию VLLM_READY_TIMEOUT_SEC или 600)",
    )
    parser.add_argument(
        "--multi-agent",
        type=str,
        metavar="MODE",
        help="Режим мультиагентного подхода (simple_4agents, critic_3agents, qa_workflow)"
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Использовать structured output через Pydantic"
    )
    outlines_group = parser.add_mutually_exclusive_group()
    outlines_group.add_argument(
        "--outlines",
        action="store_true",
        help="Использовать outlines со схемой из outlines_schema.py"
    )
    outlines_group.add_argument(
        "--pydantic-outlines",
        action="store_true",
        help="Использовать outlines со схемой из Pydantic (model_json_schema) вместо outlines_schema.py"
    )
    parser.add_argument(
        "--guidance",
        action="store_true",
        help="Constrained decoding через llguidance (по умолчанию схема RUS)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        metavar="NAME",
        help="Название промпта из prompt_config.py (например, DETAILED_INSTR_ZEROSHOT_CD)"
    )
    args = parser.parse_args()

    if args.ollama and args.vllm:
        print("Ошибка: укажите только один из --ollama или --vllm")
        sys.exit(1)
    if args.local_only and (args.ollama or args.vllm):
        print("Ошибка: не комбинируйте --local-only с --ollama / --vllm")
        sys.exit(1)

    log_path = os.path.join(OUTPUT_DIR, "evaluation_summary.log")
    with ConsoleLogCapture(log_path):
        run_all_models(
            local_only=args.local_only,
            multi_agent_mode=args.multi_agent,
            structured_output=args.structured_output,
            use_outlines=args.outlines,
            prompt_template_name=args.prompt,
            pydantic_outlines=args.pydantic_outlines,
            use_guidance=args.guidance,
            ollama_only=args.ollama,
            vllm_only=args.vllm,
            vllm_ready_timeout_sec=args.vllm_ready_timeout,
        )

