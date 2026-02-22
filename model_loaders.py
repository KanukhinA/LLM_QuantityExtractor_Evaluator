"""
Функции для загрузки различных моделей
"""
import torch
import os
import warnings
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM, AutoProcessor, AutoModelForSeq2SeqLM, AutoModelForImageTextToText, T5ForConditionalGeneration, T5Tokenizer
from typing import Tuple, Any, Optional
from config import HF_TOKEN, GEMINI_API_KEY, USE_FLASH_ATTENTION_2

# Импорт для API моделей
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

# Подавляем предупреждение о нераспознанных ключах в rope_parameters для yarn
warnings.filterwarnings("ignore", message=".*Unrecognized keys in `rope_parameters`.*")

# Настройки для загрузки
HF_HUB_DOWNLOAD_TIMEOUT = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "300"))  # 5 минут по умолчанию


def _get_flash_attn_kwargs() -> dict:
    """
    Возвращает kwargs для использования Flash Attention 2 при загрузке модели.
    Если USE_FLASH_ATTENTION_2 включен и пакет flash-attn доступен, возвращает
    {"attn_implementation": "flash_attention_2"}, иначе пустой словарь.
    """
    if not USE_FLASH_ATTENTION_2:
        return {}
    try:
        import flash_attn  # noqa: F401  # type: ignore
        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        warnings.warn(
            "USE_FLASH_ATTENTION_2 включен, но flash-attn не установлен. "
            "Установите: pip install flash-attn --no-build-isolation (требуется CUDA)."
        )
        return {}


def _apply_chat_template_if_available(tokenizer: Any, prompt: str) -> str:
    """Если у токенизатора есть chat_template (Mistral, Llama и др.), форматирует промпт как user message."""
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        return prompt
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def _generate_with_outlines(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response_schema: Any,
    max_new_tokens: int = 1024,
) -> str:
    """
    Генерация JSON через outlines. Схема берётся из outlines_schema (JSON), не из Pydantic.
    Поддерживает старый (generate.json) и новый (Generator) API.
    """
    prompt = _apply_chat_template_if_available(tokenizer, prompt)
    try:
        import outlines  # type: ignore
        try:
            from outlines import generate  # type: ignore
            use_generate_json = True
        except ImportError:
            from outlines import Generator  # type: ignore
            use_generate_json = False
    except Exception as e:
        raise ImportError(
            f"Не удалось загрузить outlines: {e}. Установите: pip install \"outlines[transformers]\""
        ) from e

    from outlines_schema import get_outlines_schema_str
    schema_str = get_outlines_schema_str()

    outlines_model = outlines.models.transformers.Transformers(model, tokenizer)
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if use_generate_json:
        # whitespace_pattern=r"" — без пробелов между JSON-элементами (многие токенизаторы не имеют отдельного токена пробела)
        generator = generate.json(outlines_model, schema_str, whitespace_pattern=r"")
        generated = generator(prompt, **gen_kwargs)
    else:
        generator = Generator(outlines_model, output_type=schema_str)
        generated = generator(prompt, **gen_kwargs)

    if isinstance(generated, (dict, list)):
        import json as _json
        from structured_schemas import latin_to_cyrillic_output
        converted = latin_to_cyrillic_output(generated)
        return _json.dumps(converted, ensure_ascii=False, indent=2)
    return str(generated).strip()


def _is_outlines_vocabulary_error(exc: BaseException) -> bool:
    """Проверяет, является ли ошибка несовместимостью словаря токенизатора с outlines (кириллица и т.д.)."""
    msg = str(exc).lower()
    return "vocabulary" in msg and ("incompatible" in msg or "incompat" in msg)


def _load_causal_4bit(
    model_name: str,
    model_class: type,
    hyperparameters: Optional[dict] = None,
    **from_pretrained_extra
) -> Tuple[Any, Any]:
    """
    Загрузка causal LM в 4-bit (nf4) через BitsAndBytes. device_map="auto" распределяет по устройствам сам.
    """
    from transformers import BitsAndBytesConfig
    print(f"   Загрузка токенизатора {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        timeout=HF_HUB_DOWNLOAD_TIMEOUT,
        resume_download=from_pretrained_extra.pop("resume_download", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Загрузка модели {model_name} (4-bit nf4)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = model_class.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        **from_pretrained_extra,
    )
    if hasattr(model, "eval"):
        model = model.eval()
    print(f"   Модель загружена в 4-bit (nf4)")
    return model, tokenizer


def load_gemma_3(model_name: str, vram_warning: Optional[str] = None, model_size_warning: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """
    Универсальная функция загрузки моделей Gemma 3 через Gemma3ForCausalLM.
    Используется для всех Gemma 3 моделей (1b, 4b, 12b, 27b).
    
    Args:
        model_name: название модели на HuggingFace (например, "google/gemma-3-4b-it")
        vram_warning: предупреждение о требованиях к VRAM (опционально)
        model_size_warning: предупреждение о размере модели (опционально)
    
    Returns:
        (model, tokenizer)
    """
    hp = hyperparameters or {}
    if hp.get("torch_dtype") in ("nf4", "4bit"):
        raise NotImplementedError(
            "BitsAndBytes 4-bit не поддерживается для Gemma 3 (гибридная архитектура). "
            "Укажите torch_dtype: \"bfloat16\" или используйте API (gemma-3-27b-api)."
        )
    print(f"   Загрузка токенизатора {model_name}...")
    if vram_warning:
        print(f"   ⚠️ Примечание: {vram_warning}")
    
    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            timeout=HF_HUB_DOWNLOAD_TIMEOUT,
            resume_download=True
        )
        elapsed = time.time() - start_time
        print(f"   ✓ Токенизатор загружен за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки токенизатора: {e}")
        print(f"   Возможные причины:")
        print(f"     - Медленное интернет-соединение")
        print(f"     - Проблемы с HuggingFace серверами")
        print(f"     - Неверный или истекший HF_TOKEN")
        print(f"   Попробуйте:")
        print(f"     - Проверить интернет-соединение")
        print(f"     - Проверить HF_TOKEN в config_secrets.py")
        print(f"     - Увеличить таймаут: set HF_HUB_DOWNLOAD_TIMEOUT=600")
        raise
    
    # Устанавливаем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Загрузка модели {model_name} (Gemma3ForCausalLM)...")
    if model_size_warning:
        print(f"   ⚠️ {model_size_warning}")
    
    try:
        start_time = time.time()
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
            **_get_flash_attn_kwargs()
        ).eval()  # Переводим в режим eval для инференса
        elapsed = time.time() - start_time
        if elapsed > 60:
            print(f"   ✓ Модель загружена за {elapsed:.1f}с ({elapsed/60:.1f} минут)")
        else:
            print(f"   ✓ Модель загружена за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        print(f"   Рекомендации:")
        print(f"     - Используйте квантизацию (4-bit или 8-bit) для уменьшения требований к памяти")
        if "api" not in model_name:
            api_model = model_name.replace("google/", "").replace("-it", "-api")
            print(f"     - Рассмотрите использование API версии: {api_model}")
        print(f"     - Проверьте доступную VRAM: python gpu_info.py")
        raise
    
    return model, tokenizer


def load_mistral_3(model_name: str, vram_warning: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """
    Универсальная функция загрузки моделей Mistral 3 через Mistral3ForConditionalGeneration.
    Используется только для репозиториев mistralai/Ministral-*.
    
    ВАЖНО: 
    - Требуется transformers>=4.50.0.dev0: pip install git+https://github.com/huggingface/transformers
    - Требуется mistral-common >= 1.8.6: pip install mistral-common --upgrade
    
    Args:
        model_name: название модели на HuggingFace (например, "mistralai/Ministral-3-8B-Instruct-2512")
        vram_warning: предупреждение о требованиях к VRAM (опционально)
        hyperparameters: опционально; при torch_dtype "nf4"/"4bit" модель загружается в 4-bit
    
    Returns:
        (model, tokenizer)
    """
    name_lower = (model_name or "").lower()
    if "mistralai" not in name_lower and "ministral" not in name_lower:
        return load_standard_model(model_name, hyperparameters=hyperparameters)
    hp = hyperparameters or {}
    if hp.get("torch_dtype") in ("nf4", "4bit"):
        from transformers import Mistral3ForConditionalGeneration
        return _load_causal_4bit(model_name, Mistral3ForConditionalGeneration, hyperparameters)
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    
    print(f"   Загрузка токенизатора {model_name}...")
    if vram_warning:
        print(f"   ⚠️ {vram_warning}")
    
    try:
        start_time = time.time()
        tokenizer = MistralCommonBackend.from_pretrained(model_name, token=HF_TOKEN)
        elapsed = time.time() - start_time
        print(f"   ✓ Токенизатор загружен за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки токенизатора: {e}")
        raise
    
    print(f"   Загрузка модели {model_name}...")
    try:
        start_time = time.time()
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            token=HF_TOKEN,
            **_get_flash_attn_kwargs()
        )
        elapsed = time.time() - start_time
        if elapsed > 60:
            print(f"   ✓ Модель загружена за {elapsed:.1f}с ({elapsed/60:.1f} минут)")
        else:
            print(f"   ✓ Модель загружена за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        print(f"   Возможные причины:")
        if vram_warning:
            print(f"     - Недостаточно VRAM ({vram_warning})")
        print(f"     - Проверьте доступную VRAM: python gpu_info.py")
        raise
    
    return model, tokenizer


def load_standard_model(model_name: str, dtype: Optional[str] = None, torch_dtype: Optional[str] = None, 
                        device_map: str = "auto", trust_remote_code: bool = True, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """
    Универсальная функция загрузки стандартных моделей через AutoTokenizer и AutoModelForCausalLM.
    Используется как fallback, когда индивидуальная функция загрузки не найдена.
    
    Индивидуальные функции загрузки нужны только для особых случаев:
    - Модели, использующие специальные классы (Gemma3ForCausalLM, Mistral3ForConditionalGeneration, T5ForConditionalGeneration)
    - Модели с особыми настройками или обработкой ошибок
    - Модели с предупреждениями о требованиях к VRAM
    
    Для стандартных моделей (Qwen, Gemma 2, и т.д.) эта функция используется автоматически.
    В hyperparameters можно передать torch_dtype: "nf4" для 4-bit квантизации любой модели.
    
    Args:
        model_name: название модели на HuggingFace
        dtype: тип данных для модели (например, "bfloat16", "float16")
        torch_dtype: тип данных для torch (например, "auto", "bfloat16")
        device_map: карта устройств ("auto", "cuda", и т.д.)
        trust_remote_code: доверять ли удаленному коду
        hyperparameters: опционально; при torch_dtype "nf4"/"4bit" модель загружается в 4-bit
    
    Returns:
        (model, tokenizer)
    """
    hp = hyperparameters or {}
    if hp.get("torch_dtype") in ("nf4", "4bit"):
        return _load_causal_4bit(model_name, AutoModelForCausalLM, hyperparameters)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )
    
    # Определяем параметры для загрузки модели
    model_kwargs = {
        "device_map": device_map,
        "token": HF_TOKEN,
        "trust_remote_code": trust_remote_code,
        **_get_flash_attn_kwargs()
    }
    
    # Преобразуем dtype/torch_dtype в нужный формат
    if torch_dtype:
        if torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
    elif dtype:
        if dtype == "bfloat16":
            model_kwargs["dtype"] = torch.bfloat16
        elif dtype == "float16":
            model_kwargs["dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    return model, tokenizer

def load_gemma_2_2b(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка google/gemma-2-2b-it (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "google/gemma-2-2b-it"
    return load_standard_model(name, dtype="bfloat16", device_map="cuda", hyperparameters=hyperparameters)


def load_mistral_3_8b_instruct(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка mistralai/Ministral-3-8B-Instruct-2512 (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "mistralai/Ministral-3-8B-Instruct-2512"
    return load_mistral_3(name, vram_warning="Модель требует ~16GB VRAM для полной загрузки", hyperparameters=hyperparameters)


def load_mistral_3_14b_instruct(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка mistralai/Ministral-3-14B-Instruct-2512 (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "mistralai/Ministral-3-14B-Instruct-2512"
    return load_mistral_3(name, vram_warning="Модель требует ~28GB VRAM для полной загрузки", hyperparameters=hyperparameters)


def load_mistral_3_3b_reasoning(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка mistralai/Ministral-3-3B-Reasoning-2512 (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "mistralai/Ministral-3-3B-Reasoning-2512"
    return load_mistral_3(name, hyperparameters=hyperparameters)

def load_qwen_2_5_3b(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen2.5-3B-Instruct (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "Qwen/Qwen2.5-3B-Instruct"
    return load_standard_model(name, dtype="bfloat16", hyperparameters=hyperparameters)


def load_qwen_3_4b(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen3-4B-Instruct-2507 (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "Qwen/Qwen3-4B-Instruct-2507"
    return load_standard_model(name, dtype="bfloat16", hyperparameters=hyperparameters)


def load_qwen_3_8b(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen3-8B (при torch_dtype nf4/4bit — 4-bit)."""
    name = model_name or "Qwen/Qwen3-8B"
    return load_standard_model(name, torch_dtype="auto", hyperparameters=hyperparameters)


def load_codegemma_7b(model_name: Optional[str] = None, hyperparameters: Optional[dict] = None) -> Tuple[Any, Any]:
    """Загрузка google/codegemma-7b-it (при torch_dtype nf4/4bit — 4-bit)."""
    model_id = model_name or "google/codegemma-7b-it"
    hp = hyperparameters or {}
    if hp.get("torch_dtype") in ("nf4", "4bit"):
        return _load_causal_4bit(model_id, AutoModelForCausalLM, hyperparameters)
    print(f"   Загрузка токенизатора {model_id}...")
    print(f"   ⚠️ Примечание: CodeGemma специализирована для работы с кодом")
    print(f"   (это может занять некоторое время при первом запуске)")
    
    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=HF_TOKEN,
            timeout=HF_HUB_DOWNLOAD_TIMEOUT,
            resume_download=True
        )
        elapsed = time.time() - start_time
        print(f"   ✓ Токенизатор загружен за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки токенизатора: {e}")
        print(f"   Возможные причины:")
        print(f"     - Медленное интернет-соединение")
        print(f"     - Проблемы с HuggingFace серверами")
        print(f"     - Неверный или истекший HF_TOKEN")
        print(f"   Попробуйте:")
        print(f"     - Проверить интернет-соединение")
        print(f"     - Проверить HF_TOKEN в config_secrets.py")
        print(f"     - Увеличить таймаут: set HF_HUB_DOWNLOAD_TIMEOUT=600")
        raise
    
    # Устанавливаем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Загрузка модели {model_id}...")
    print(f"   ⚠️ Это может занять некоторое время из-за размера модели (~7B параметров)")
    try:
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
            trust_remote_code=True,
            **_get_flash_attn_kwargs()
        )
        elapsed = time.time() - start_time
        print(f"   ✓ Модель загружена за {elapsed:.1f}с ({elapsed/60:.1f} минут)")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        print(f"   Возможные причины:")
        print(f"     - Недостаточно VRAM (модель требует ~14GB для полной загрузки)")
        print(f"     - Медленное интернет-соединение")
        print(f"     - Проблемы с HuggingFace серверами")
        print(f"   Рекомендации:")
        print(f"     - Используйте квантизацию (4-bit или 8-bit) для уменьшения требований к памяти")
        print(f"     - Проверьте доступную VRAM: python gpu_info.py")
        raise
    
    return model, tokenizer


def generate_gemma(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 1024, 
    repetition_penalty: float = None,
    structured_output: bool = False,
    response_schema: Any = None,
    use_outlines: bool = False
) -> str:
    """
    Функция генерации для Gemma 3 моделей с использованием правильного формата сообщений
    
    Args:
        model: модель (Gemma3ForCausalLM)
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
        structured_output: флаг для structured output
        response_schema: схема для structured output
        use_outlines: использовать ли outlines для структурированной генерации JSON
    """
    # Если включен outlines-режим для structured output — генерируем JSON напрямую по схеме
    # Работает только для локальных HF-моделей; для API моделей outlines не используется.
    if use_outlines and response_schema is not None:
        try:
            return _generate_with_outlines(model, tokenizer, prompt, response_schema, max_new_tokens)
        except Exception as e:
            if _is_outlines_vocabulary_error(e):
                warnings.warn(
                    f"Outlines несовместим со словарём токенизатора (кириллическая схема). "
                    f"Используем обычную генерацию. Детали: {e}",
                    UserWarning
                )
            else:
                raise RuntimeError(f"Outlines генерация не удалась: {e}") from e
    
    # Для Gemma 3 используем правильный формат сообщений
    # Проверяем, является ли модель Gemma3ForCausalLM
    is_gemma3 = isinstance(model, Gemma3ForCausalLM) or model.__class__.__name__ == 'Gemma3ForCausalLM'
    
    if is_gemma3 and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        # Используем формат сообщений для Gemma 3
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                },
            ],
        ]
        
        # Применяем chat template с правильными параметрами
        inputs_dict = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Применяем device и dtype к тензорам в словаре
        # ВАЖНО: input_ids должны оставаться Long (int64), не конвертируем их в bfloat16
        device = next(model.parameters()).device
        inputs = {}
        for key, value in inputs_dict.items():
            if isinstance(value, torch.Tensor):
                if key == "input_ids":
                    # input_ids должны быть Long (int64), только переносим на device
                    inputs[key] = value.to(device)
                else:
                    # Остальные тензоры (attention_mask и т.д.) могут быть в bfloat16
                    inputs[key] = value.to(device).to(torch.bfloat16)
            else:
                inputs[key] = value
        
        # Генерируем ответ
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        
        # Добавляем eos_token_id, если он есть
        if tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
        
        # Добавляем repetition_penalty, если указан
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)
        
        # Декодируем ответ
        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        full_text = outputs_decoded[0] if outputs_decoded else ""
        
        # Нужно извлечь только новые токены (ответ модели)
        # Для этого декодируем входные данные отдельно
        inputs_decoded = tokenizer.batch_decode(
            inputs['input_ids'] if isinstance(inputs, dict) else inputs, 
            skip_special_tokens=True
        )
        input_text = inputs_decoded[0] if inputs_decoded else ""
        
        # Извлекаем только новую часть ответа
        if full_text.startswith(input_text):
            text = full_text[len(input_text):].strip()
        else:
            # Если не начинается с input_text, пытаемся найти ответ другим способом
            # Просто возвращаем весь текст и пусть парсер разберется
            text = full_text
        
        return text
    
    else:
        # Fallback для старых версий Gemma или если chat template недоступен
        # Пробуем использовать chat template, если он доступен
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Форматируем как диалог
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt
        
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        
        # Добавляем eos_token_id, если он есть
        if tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
        
        # Добавляем repetition_penalty, если указан
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        
        with torch.no_grad():
            output_ids = model.generate(**generate_kwargs)
        
        # Декодируем только новые токены (игнорируя входные)
        input_length = input_ids.shape[1]
        generated_ids = output_ids[0][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Если декодирование новых токенов дало пустой результат, пробуем декодировать весь ответ
        if not text.strip():
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Убираем повтор prompt
            if text.startswith(formatted_prompt):
                text = text[len(formatted_prompt):].strip()
            elif text.startswith(prompt):
                text = text[len(prompt):].strip()
        
        return text.strip()


def generate_standard(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    repetition_penalty: float = None,
    structured_output: bool = False,
    response_schema: Any = None,
    use_outlines: bool = False,
) -> str:
    """
    Стандартная функция генерации для большинства моделей
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
    """
    # Если включен outlines-режим для structured output — генерируем JSON напрямую по схеме
    # Работает только для локальных HF-моделей; для API моделей outlines не используется.
    if use_outlines and response_schema is not None:
        try:
            return _generate_with_outlines(model, tokenizer, prompt, response_schema, max_new_tokens)
        except Exception as e:
            if _is_outlines_vocabulary_error(e):
                warnings.warn(
                    f"Outlines несовместим со словарём токенизатора (кириллическая схема). "
                    f"Используем обычную генерацию. Детали: {e}",
                    UserWarning
                )
            else:
                raise RuntimeError(f"Outlines генерация не удалась: {e}") from e

    formatted_prompt = _apply_chat_template_if_available(tokenizer, prompt)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        try:
            output_ids = model.generate(**generate_kwargs)
        except AttributeError as e:
            if "from_legacy_cache" in str(e):
                generate_kwargs["use_cache"] = False
                output_ids = model.generate(**generate_kwargs)
            else:
                raise

    input_length = input_ids.shape[1]
    generated_ids = output_ids[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if not text.strip():
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if text.startswith(formatted_prompt):
            text = text[len(formatted_prompt):].strip()
        elif text.startswith(prompt):
            text = text[len(prompt):].strip()

    return text.strip()


def generate_qwen(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 512, 
    repetition_penalty: float = None,
    structured_output: bool = False,
    response_schema: Any = None,
    use_outlines: bool = False
) -> str:
    """
    Функция генерации для Qwen с дополнительными стоп-строками
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
        structured_output: флаг для structured output
        response_schema: схема для structured output
        use_outlines: использовать ли outlines для структурированной генерации JSON
    """
    # Если включен outlines-режим для structured output — генерируем JSON напрямую по схеме
    # Работает только для локальных HF-моделей; для API моделей outlines не используется.
    if use_outlines and response_schema is not None:
        try:
            return _generate_with_outlines(model, tokenizer, prompt, response_schema, max_new_tokens)
        except Exception as e:
            if _is_outlines_vocabulary_error(e):
                warnings.warn(
                    f"Outlines несовместим со словарём токенизатора (кириллическая схема). "
                    f"Используем обычную генерацию. Детали: {e}",
                    UserWarning
                )
            else:
                raise RuntimeError(f"Outlines генерация не удалась: {e}") from e
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "stop_strings": ["Human:", "Example"],
        "tokenizer": tokenizer
    }
    
    # Добавляем repetition_penalty, если указан
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)
    
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Убираем повтор prompt
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    
    # Удаляем стоп-строки
    for s in ["Human:", "Example"]:
        if s in text:
            text = text.split(s)[0].strip()
    
    return text.strip()


def generate_t5(
    model, 
    tokenizer_or_processor, 
    prompt: str, 
    max_new_tokens: int = 1024, 
    repetition_penalty: float = None,
    structured_output: bool = False,
    response_schema: Any = None,
    use_outlines: bool = False
) -> str:
    """
    Функция генерации для T5/Seq2Seq моделей
    Поддерживает как processor (AutoProcessor), так и tokenizer (T5Tokenizer)
    
    Args:
        model: модель (AutoModelForImageTextToText, AutoModelForSeq2SeqLM или T5ForConditionalGeneration)
        tokenizer_or_processor: процессор (AutoProcessor) или токенизатор (T5Tokenizer)
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
        structured_output: флаг для structured output (игнорируется для T5)
        response_schema: схема для structured output (игнорируется для T5)
        use_outlines: использовать ли outlines (игнорируется для T5)
    """
    # Определяем, это processor или tokenizer
    # Для T5Gemma processor требует явный параметр text= для текстового ввода
    
    input_ids = None
    decoder = None
    
    try:
        # Пробуем использовать как processor (если это AutoProcessor для T5Gemma)
        # Для T5Gemma нужно использовать text= параметр
        if hasattr(tokenizer_or_processor, '__call__'):
            # Пробуем с явным text= параметром (для T5Gemma)
            try:
                inputs = tokenizer_or_processor(text=prompt, return_tensors="pt")
                if inputs is not None and isinstance(inputs, dict) and 'input_ids' in inputs:
                    input_ids = inputs['input_ids'].to(model.device)
                    decoder = tokenizer_or_processor
            except (TypeError, ValueError):
                # Если не сработало с text=, пробуем без него
                try:
                    inputs = tokenizer_or_processor(prompt, return_tensors="pt")
                    if inputs is not None and isinstance(inputs, dict) and 'input_ids' in inputs:
                        input_ids = inputs['input_ids'].to(model.device)
                        decoder = tokenizer_or_processor
                except Exception:
                    pass
        
        # Если processor не сработал, используем как tokenizer
        if input_ids is None:
            # Проверяем, есть ли у объекта атрибут tokenizer (processor может содержать tokenizer)
            if hasattr(tokenizer_or_processor, 'tokenizer'):
                actual_tokenizer = tokenizer_or_processor.tokenizer
            else:
                actual_tokenizer = tokenizer_or_processor
            
            input_ids = actual_tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            decoder = actual_tokenizer
            
    except Exception as e:
        # Если все не сработало, пробуем последний вариант
        try:
            if hasattr(tokenizer_or_processor, 'tokenizer'):
                actual_tokenizer = tokenizer_or_processor.tokenizer
            else:
                actual_tokenizer = tokenizer_or_processor
            input_ids = actual_tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            decoder = actual_tokenizer
        except Exception as e2:
            raise RuntimeError(f"Не удалось обработать промпт с processor/tokenizer: {e2}") from e2
    
    if input_ids is None:
        raise RuntimeError("Не удалось получить input_ids из processor/tokenizer")
    
    generate_kwargs = {
        "input_ids": input_ids,
        "max_length": input_ids.shape[1] + max_new_tokens,  # T5 использует max_length вместо max_new_tokens
        "do_sample": False,
    }
    
    # Добавляем decoder_start_token_id для T5 моделей
    if decoder is not None:
        if hasattr(decoder, 'pad_token_id') and decoder.pad_token_id is not None:
            generate_kwargs["decoder_start_token_id"] = decoder.pad_token_id
        elif hasattr(decoder, 'tokenizer') and hasattr(decoder.tokenizer, 'pad_token_id'):
            if decoder.tokenizer.pad_token_id is not None:
                generate_kwargs["decoder_start_token_id"] = decoder.tokenizer.pad_token_id
    
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)
    
    # Декодируем ответ
    if decoder is None:
        raise RuntimeError("Decoder не определен для декодирования ответа")
    
    # Проверяем, что output_ids не None и не пустой
    if output_ids is None or len(output_ids) == 0:
        raise RuntimeError("Модель не сгенерировала ответ")
    
    # Для processor может потребоваться использовать tokenizer для декодирования
    if hasattr(decoder, 'decode'):
        text = decoder.decode(output_ids[0], skip_special_tokens=True)
    elif hasattr(decoder, 'tokenizer') and hasattr(decoder.tokenizer, 'decode'):
        text = decoder.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        raise RuntimeError(f"Decoder {type(decoder)} не имеет метода decode")
    
    # Убираем повтор prompt, если он есть
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    
    return text.strip()


def generate_qwen_3(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 32768, 
    repetition_penalty: float = None, 
    enable_thinking: bool = False,
    structured_output: bool = False,
    response_schema: Any = None,
    use_outlines: bool = False
) -> str:
    """
    Функция генерации для Qwen3 с поддержкой thinking mode
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов (по умолчанию 32768 для Qwen3)
        repetition_penalty: штраф за повторения (если None, не используется)
        enable_thinking: включить thinking mode (по умолчанию False)
        structured_output: флаг для structured output (игнорируется для Qwen3)
        response_schema: схема для structured output (игнорируется для Qwen3)
        use_outlines: использовать ли outlines (игнорируется для Qwen3)
    """
    # Подготавливаем сообщения для chat template
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Применяем chat template с thinking mode
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    # Токенизируем
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Параметры генерации
    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    
    # Добавляем eos_token_id, если он есть
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    
    # Добавляем repetition_penalty, если указан
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    
    # Генерируем
    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)
    
    # Извлекаем только новые токены (ответ модели)
    input_length = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_length:].tolist()
    
    # Декодируем ответ
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return text.strip()

