"""Тест вида пробела в токенизаторе Gemma"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Gemma использует '▁' для пробела
test_str = " : "
tokens = tokenizer.encode(test_str, add_special_tokens=False)
print(f"Токены для ' : ': {tokens}")
print(f"Декод: {tokenizer.decode(tokens)}")
