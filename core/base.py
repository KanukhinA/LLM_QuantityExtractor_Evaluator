"""
Базовые абстрактные классы для ООП архитектуры
"""
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer, PreTrainedModel


class BaseGenerator(ABC):
    """Абстрактный класс для генерации текста"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Генерирует текст на основе промпта
        
        Args:
            prompt: входной промпт
            max_new_tokens: максимальное количество новых токенов
            **kwargs: дополнительные параметры генерации
            
        Returns:
            Сгенерированный текст
        """
        pass
    
    def prepare_prompt(self, prompt: str) -> str:
        """
        Подготавливает промпт (может быть переопределен в подклассах)
        
        Args:
            prompt: исходный промпт
            
        Returns:
            Подготовленный промпт
        """
        return prompt

