"""
Реализации стратегий генерации
"""
import torch
from typing import Optional
from .base import BaseGenerator


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
