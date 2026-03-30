"""
Утилиты для получения информации о GPU
"""
import re
import torch
import subprocess
import platform
from typing import Any, Dict, Optional


def get_gpu_info() -> Dict[str, Any]:
    """
    Получает информацию о видеокарте
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_total_gb": None,
        "gpu_memory_allocated_gb": None,
        "gpu_memory_reserved_gb": None,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    
    # Дополнительная информация через nvidia-smi (если доступно)
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    if len(parts) >= 2:
                        info["gpu_name_detailed"] = parts[0].strip()
                        info["driver_version"] = parts[-1].strip() if len(parts) > 2 else None
    except Exception:
        pass
    
    return info


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Получает текущее использование памяти GPU в GB (через PyTorch).
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    
    return {
        "allocated": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        "reserved": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        "total": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    }


def get_gpu_memory_usage_nvidia_smi() -> Dict[str, float]:
    """
    Получает суммарную занятость GPU через nvidia-smi (все процессы на устройстве).
    Возвращает used_gb, total_gb; при ошибке — used_gb=0, total_gb=0.
    """
    out = {"used_gb": 0.0, "total_gb": 0.0}
    try:
        gpu_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_idx),
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return out
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip().split()[0] for p in line.split(",")]
        if len(parts) >= 2:
            out["used_gb"] = round(float(parts[0]) / 1024.0, 2)
            out["total_gb"] = round(float(parts[1]) / 1024.0, 2)
    except Exception:
        pass
    return out


def get_ollama_inference_vram_gb(gpu_index: Optional[int] = None) -> float:
    """
    Оценка VRAM только процессов Ollama на GPU (nvidia-smi query-compute-apps), в ГБ.
    По смыслу ближе к torch.cuda.memory_allocated: не включает память других приложений на карте,
    в отличие от memory.used по всему GPU.

    Суммируются строки, где process_name содержит «ollama» (регистр не важен).
    Если драйвер не отдаёт compute-apps, возвращает 0.0.
    """
    idx = gpu_index
    if idx is None:
        try:
            idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except Exception:
            idx = 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(idx),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0
        total_mib = 0.0
        pattern = re.compile(r"^(\d+),\s*(.+),\s*(\d+)\s*$")
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue
            _pid, name, mem_s = m.group(1), m.group(2).strip(), m.group(3)
            if "ollama" not in name.lower():
                continue
            try:
                total_mib += float(mem_s)
            except ValueError:
                continue
        return round(total_mib / 1024.0, 2)
    except Exception:
        return 0.0

