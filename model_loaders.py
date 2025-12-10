"""
Функции для загрузки различных моделей
"""
import torch
import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Any
from config import HF_TOKEN

# Подавляем предупреждение о нераспознанных ключах в rope_parameters для yarn
warnings.filterwarnings("ignore", message=".*Unrecognized keys in `rope_parameters`.*")



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


def load_gemma_3_4b() -> Tuple[Any, Any]:
    """Загрузка google/gemma-3-4b-it БЕЗ квантизации (требует ~8GB VRAM)"""
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
        dtype=torch.float16,  # float16 для экономии памяти
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer


def generate_gemma(model, tokenizer, prompt: str, max_new_tokens: int = 1024, repetition_penalty: float = None) -> str:
    """
    Функция генерации для Gemma моделей с использованием chat template
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: промпт
        max_new_tokens: максимальное количество новых токенов
        repetition_penalty: штраф за повторения (если None, не используется)
    """
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

