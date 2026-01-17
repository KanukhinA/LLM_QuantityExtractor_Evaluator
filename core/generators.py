"""
Реализации стратегий генерации
"""
import torch
from typing import Optional
from .base import BaseGenerator


class APIGenerator(BaseGenerator):
    """Генератор для API моделей (Google Generative AI и OpenRouter)"""
    
    def __init__(self, client, tokenizer=None, model_name: str = "gemma-3-12b-it"):
        """
        Инициализация API генератора
        
        Args:
            client: клиент API (genai.Client или OpenAI)
            tokenizer: не используется, но нужен для совместимости (может быть None)
            model_name: имя модели API
        """
        # Для API моделей model и tokenizer могут быть None
        # Используем client напрямую
        self.client = client
        self.model_name = model_name
        # Устанавливаем model и tokenizer в None для совместимости с BaseGenerator
        self.model = None
        self.tokenizer = None
        
        # Определяем тип API по model_name
        # OpenRouter модели обычно содержат "/" в имени (например, "mistralai/mistral-small-3.1-24b-instruct:free")
        # или ключевые слова "deepseek", "mistral", "openrouter"
        model_name_lower = model_name.lower()
        if ("deepseek" in model_name_lower or 
            "mistral" in model_name_lower or 
            "openrouter" in model_name_lower or 
            "/" in model_name):
            self.api_type = "openrouter"
        else:
            self.api_type = "gemini"
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 1024,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """Генерация через API"""
        prompt = self.prepare_prompt(prompt)
        
        # Выбираем функцию генерации в зависимости от типа API
        if self.api_type == "openrouter":
            from model_loaders_api import generate_openrouter_api
            return generate_openrouter_api(
                client=self.client,
                tokenizer=None,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                model_name=self.model_name,
                repetition_penalty=repetition_penalty
            )
        else:
            from model_loaders_api import generate_gemma_api
            return generate_gemma_api(
                client=self.client,
                tokenizer=None,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                model_name=self.model_name,
                repetition_penalty=repetition_penalty
            )


class StandardGenerator(BaseGenerator):
    """Стандартная стратегия генерации с использованием KV-кэша"""
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 1024,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """Стандартная генерация с кэшем"""
        prompt = self.prepare_prompt(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            **kwargs
        }
        
        if self.tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        
        with torch.no_grad():
            try:
                output_ids = self.model.generate(**generate_kwargs)
            except KeyboardInterrupt:
                # Пробрасываем KeyboardInterrupt наверх для обработки
                raise
            except AttributeError as e:
                if "from_legacy_cache" in str(e):
                    generate_kwargs["use_cache"] = False
                    try:
                        output_ids = self.model.generate(**generate_kwargs)
                    except KeyboardInterrupt:
                        raise
                else:
                    raise
        
        input_length = input_ids.shape[1]
        generated_ids = output_ids[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if not text.strip():
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
        
        return text.strip()
