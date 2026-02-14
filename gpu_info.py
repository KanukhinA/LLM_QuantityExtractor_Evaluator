"""
Утилиты для получения информации о GPU
"""
import torch
import subprocess
import platform
from typing import Dict, Any


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
    Получает использование памяти GPU через nvidia-smi (без PyTorch).
    Нужно для замера VRAM процесса Ollama: наш процесс не держит модель, GPU занят Ollama.
    Возвращает used_gb, total_gb; при ошибке — used_gb=0, total_gb=0.
    """
    out = {"used_gb": 0.0, "total_gb": 0.0}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
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

