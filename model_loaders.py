"""
Функции для загрузки различных моделей
"""
import torch
import os
import warnings
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM
from typing import Tuple, Any, Optional
from config import HF_TOKEN, GEMINI_API_KEY

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



def load_gemma_2_2b() -> Tuple[Any, Any]:
    """Загрузка google/gemma-2-2b-it"""
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="cuda",
        dtype=torch.bfloat16,
        token=HF_TOKEN
    )
    return model, tokenizer

def load_ministral_3_3b_reasoning_2512() -> Tuple[Any, Any]:
    """
    Загрузка mistralai/Ministral-3-3B-Reasoning-2512 в bfloat16
    
    ВАЖНО: 
    - Требуется transformers>=4.50.0.dev0:
      pip install git+https://github.com/huggingface/transformers
    - Требуется mistral-common >= 1.8.6 для токенизатора:
      pip install mistral-common --upgrade
    """
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    
    model_id = "mistralai/Ministral-3-3B-Reasoning-2512"
    
    # Используем MistralCommonBackend для токенизатора
    tokenizer = MistralCommonBackend.from_pretrained(model_id, token=HF_TOKEN)
    
    # Загружаем модель в bfloat16
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        token=HF_TOKEN
    )
    
    return model, tokenizer

def load_qwen_2_5_1_5b() -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen2.5-1.5B-Instruct"""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device_map="auto",
        dtype=torch.float16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_qwen_2_5_3b() -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen2.5-3B-Instruct с bfloat16"""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        dtype=torch.bfloat16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_qwen_2_5_4b() -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen2.5-4B-Instruct с bfloat16"""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-4B-Instruct",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-4B-Instruct",
        device_map="auto",
        dtype=torch.bfloat16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_qwen_3_8b() -> Tuple[Any, Any]:
    """Загрузка Qwen/Qwen3-8B с автоматическим выбором dtype"""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype="auto",
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_gemma_3_4b_4bit() -> Tuple[Any, Any]:
    """Загрузка google/gemma-3-4b-it с 4-bit quantization"""
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-4b-it",
        token=HF_TOKEN
    )
    # Устанавливаем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        device_map="auto",
        quantization_config=quantization_config,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_mistral_7b_v0_3_bnb_4bit() -> Tuple[Any, Any]:
    """Загрузка unsloth/mistral-7b-v0.3-bnb-4bit (уже квантизированная модель)"""
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        token=HF_TOKEN
    )
    # Устанавливаем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_gemma_3_1b() -> Tuple[Any, Any]:
    """Загрузка google/gemma-3-1b-it БЕЗ квантизации (использует Gemma3ForCausalLM)"""
    model_id = "google/gemma-3-1b-it"
    
    print(f"   Загрузка токенизатора {model_id}...")
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
        print(f"     - Файрвол/прокси блокирует загрузку")
        print(f"   Попробуйте:")
        print(f"     - Проверить интернет-соединение")
        print(f"     - Проверить HF_TOKEN в config_secrets.py")
        print(f"     - Увеличить таймаут: set HF_HUB_DOWNLOAD_TIMEOUT=600")
        raise
    
    # Устанавливаем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Загрузка модели {model_id} (Gemma3ForCausalLM)...")
    try:
        start_time = time.time()
        # Используем Gemma3ForCausalLM для правильной загрузки Gemma 3
        # timeout и resume_download используются только для tokenizer, не для модели
        model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        ).eval()  # Переводим в режим eval для инференса
        elapsed = time.time() - start_time
        print(f"   ✓ Модель загружена за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        raise
    
    return model, tokenizer


def load_gemma_3_4b() -> Tuple[Any, Any]:
    """Загрузка google/gemma-3-4b-it БЕЗ квантизации (требует ~8GB VRAM, использует Gemma3ForCausalLM)"""
    model_id = "google/gemma-3-4b-it"
    
    print(f"   Загрузка токенизатора {model_id}...")
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
        raise
    
    # Устанавливаем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Загрузка модели {model_id} (Gemma3ForCausalLM)...")
    try:
        start_time = time.time()
        # Используем Gemma3ForCausalLM для правильной загрузки Gemma 3
        # timeout и resume_download используются только для tokenizer, не для модели
        model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        ).eval()  # Переводим в режим eval для инференса
        elapsed = time.time() - start_time
        print(f"   ✓ Модель загружена за {elapsed:.1f}с")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        raise
    
    return model, tokenizer


def generate_gemma(model, tokenizer, prompt: str, max_new_tokens: int = 1024, repetition_penalty: float = None) -> str:
    """
    Функция генерации для Gemma 3 моделей с использованием правильного формата сообщений
    
    Args:
        model: модель (Gemma3ForCausalLM)
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
    """
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


def load_chemllm_2b_1_5() -> Tuple[Any, Any]:
    """Загрузка AI4Chem/CHEMLLM-2b-1_5"""
    # CHEMLLM требует trust_remote_code=True для выполнения custom nodes
    tokenizer = AutoTokenizer.from_pretrained(
        "AI4Chem/CHEMLLM-2b-1_5",
        token=HF_TOKEN,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "AI4Chem/CHEMLLM-2b-1_5",
        device_map="auto",
        dtype=torch.float16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_phi_3_5_mini_instruct() -> Tuple[Any, Any]:
    """Загрузка microsoft/Phi-3.5-mini-instruct"""
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        token=HF_TOKEN
    )
    # Устанавливаем pad_token, если его нет (для phi-3.5 pad_token может совпадать с eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="auto",
        dtype=torch.float16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def load_phi_4_mini_instruct() -> Tuple[Any, Any]:
    """Загрузка microsoft/Phi-4-mini-instruct"""
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        device_map="auto",
        dtype=torch.bfloat16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def generate_standard(model, tokenizer, prompt: str, max_new_tokens: int = 1024, repetition_penalty: float = None) -> str:
    """
    Стандартная функция генерации для большинства моделей
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,  # Включаем кэш по умолчанию
    }
    
    # Добавляем eos_token_id, если он есть
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    
    # Добавляем repetition_penalty, если указан
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    
    with torch.no_grad():
        try:
            output_ids = model.generate(**generate_kwargs)
        except AttributeError as e:
            if "from_legacy_cache" in str(e):
                # Если ошибка связана с кэшем, отключаем use_cache
                generate_kwargs["use_cache"] = False
                output_ids = model.generate(**generate_kwargs)
            else:
                raise
    
    # Декодируем только новые токены (игнорируя входные)
    input_length = input_ids.shape[1]
    generated_ids = output_ids[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Если декодирование новых токенов дало пустой результат, пробуем декодировать весь ответ
    if not text.strip():
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Убираем повтор prompt
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
    
    return text.strip()


def generate_phi_3_5(model, tokenizer, prompt: str, max_new_tokens: int = 1024, repetition_penalty: float = None) -> str:
    """
    Функция генерации для Phi-3.5 моделей с отключенным кэшем для обхода ошибки DynamicCache
    и явным указанием attention_mask для избежания предупреждений
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
    """
    # Токенизируем с получением attention_mask
    encoded = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,  # Явно передаем attention_mask
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": False,  # Отключаем кэш для Phi-3.5, чтобы избежать ошибки DynamicCache.from_legacy_cache
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
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
    
    return text.strip()


def generate_qwen(model, tokenizer, prompt: str, max_new_tokens: int = 512, repetition_penalty: float = None) -> str:
    """
    Функция генерации для Qwen с дополнительными стоп-строками
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
    """
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


def generate_qwen_3(model, tokenizer, prompt: str, max_new_tokens: int = 32768, repetition_penalty: float = None, enable_thinking: bool = True) -> str:
    """
    Функция генерации для Qwen3 с поддержкой thinking mode
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов (по умолчанию 32768 для Qwen3)
        repetition_penalty: штраф за повторения (если None, не используется)
        enable_thinking: включить thinking mode (по умолчанию True)
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

