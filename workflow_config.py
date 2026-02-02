"""
Конфигурация мультиагентных workflow режимов

Этот модуль содержит конфигурации для различных мультиагентных режимов.
Для добавления нового режима просто добавьте запись в словарь WORKFLOW_CONFIGS.
"""

from typing import Dict, Callable, Any
from prompt_config import (
    NUMERIC_FRAGMENTS_EXTRACTION_PROMPT,
    MASS_FRACTION_EXTRACTION_PROMPT,
    OTHER_PARAMETERS_EXTRACTION_PROMPT,
    JSON_FORMATION_PROMPT,
    DETAILED_INSTR_ONESHOT,
    CRITIC_PROMPT,
    CORRECTOR_PROMPT,
    QA_NUTRIENTS_PROMPT,
    QA_NUTRIENT_PROMPT,
    QA_STANDARD_PROMPT,
    QA_GRADE_PROMPT,
    QA_QUANTITY_PROMPT
)


def _get_simple_4agents_prompts(example_text: str) -> Dict[str, Any]:
    """Генерирует примеры промптов для simple_4agents"""
    return {
        "full_prompt_example": f"""МУЛЬТИАГЕНТНЫЙ РЕЖИМ: simple_4agents

{'='*80}
ПРОМПТ 1: ИЗВЛЕЧЕНИЕ ЧИСЛОВЫХ ФРАГМЕНТОВ
{'='*80}
{NUMERIC_FRAGMENTS_EXTRACTION_PROMPT.format(text=example_text)}

{'='*80}
ПРОМПТ 2: ИЗВЛЕЧЕНИЕ МАССОВЫХ ДОЛЕЙ
{'='*80}
{MASS_FRACTION_EXTRACTION_PROMPT.format(numeric_fragments="[Пример числовых фрагментов из предыдущего шага]")}

{'='*80}
ПРОМПТ 3: ИЗВЛЕЧЕНИЕ ПРОЧИХ ПАРАМЕТРОВ
{'='*80}
{OTHER_PARAMETERS_EXTRACTION_PROMPT.format(numeric_fragments="[Пример числовых фрагментов из предыдущего шага]")}

{'='*80}
ПРОМПТ 4: ФОРМИРОВАНИЕ JSON
{'='*80}
{JSON_FORMATION_PROMPT.format(mass_fractions="[Пример массовых долей из предыдущего шага]", other_parameters="[Пример прочих параметров из предыдущего шага]")}
""",
        "prompt_info": {
            "mode": "simple_4agents",
            "prompts_used": [
                "NUMERIC_FRAGMENTS_EXTRACTION_PROMPT",
                "MASS_FRACTION_EXTRACTION_PROMPT",
                "OTHER_PARAMETERS_EXTRACTION_PROMPT",
                "JSON_FORMATION_PROMPT"
            ],
            "example_numeric_fragments_prompt": NUMERIC_FRAGMENTS_EXTRACTION_PROMPT.format(text=example_text),
            "example_mass_fractions_prompt": MASS_FRACTION_EXTRACTION_PROMPT.format(numeric_fragments="[Пример числовых фрагментов]"),
            "example_other_parameters_prompt": OTHER_PARAMETERS_EXTRACTION_PROMPT.format(numeric_fragments="[Пример числовых фрагментов]"),
            "example_json_formation_prompt": JSON_FORMATION_PROMPT.format(mass_fractions="[Пример массовых долей]", other_parameters="[Пример прочих параметров]")
        },
        "description": "извлечение числовых фрагментов → массовые доли → прочие параметры → формирование JSON"
    }


def _get_critic_3agents_prompts(example_text: str) -> Dict[str, Any]:
    """Генерирует примеры промптов для critic_3agents"""
    example_prompt = DETAILED_INSTR_ONESHOT.format(text=example_text)
    example_response = "[Пример первоначального ответа модели]"
    example_critic_analysis = "[Пример анализа критика]"
    
    return {
        "full_prompt_example": f"""МУЛЬТИАГЕНТНЫЙ РЕЖИМ: critic_3agents

{'='*80}
ПРОМПТ 1: ГЕНЕРАЦИЯ ПЕРВОНАЧАЛЬНОГО ОТВЕТА
{'='*80}
{example_prompt}

{'='*80}
ПРОМПТ 2: АНАЛИЗ ОТВЕТА (КРИТИК)
{'='*80}
{CRITIC_PROMPT.format(prompt=example_prompt, response=example_response)}

{'='*80}
ПРОМПТ 3: ИСПРАВЛЕНИЕ ОШИБОК (ИСПРАВИТЕЛЬ)
{'='*80}
{CORRECTOR_PROMPT.format(
    prompt=example_prompt,
    original_response=example_response,
    critic_analysis=example_critic_analysis
)}
""",
        "prompt_info": {
            "mode": "critic_3agents",
            "prompts_used": [
                "FERTILIZER_EXTRACTION_PROMPT_WITH_EXAMPLE",
                "CRITIC_PROMPT",
                "CORRECTOR_PROMPT"
            ],
            "example_generator_prompt": example_prompt,
            "example_critic_prompt": CRITIC_PROMPT.format(prompt=example_prompt, response=example_response),
            "example_corrector_prompt": CORRECTOR_PROMPT.format(
                prompt=example_prompt,
                original_response=example_response,
                critic_analysis=example_critic_analysis
            )
        },
        "description": "генератор (первоначальный ответ) → критик (анализ ответа) → исправитель (устранение ошибок)"
    }


def _get_qa_workflow_prompts(example_text: str) -> Dict[str, Any]:
    """Генерирует примеры промптов для qa_workflow"""
    example_nutrients = ["N", "P2O5", "K2O"]
    example_substance = "N"
    
    return {
        "full_prompt_example": f"""МУЛЬТИАГЕНТНЫЙ РЕЖИМ: qa_workflow

{'='*80}
ПРОМПТ 1: ИЗВЛЕЧЕНИЕ ПИТАТЕЛЬНЫХ ВЕЩЕСТВ
{'='*80}
{QA_NUTRIENTS_PROMPT.format(text=example_text)}

{'='*80}
ПРОМПТ 2: ИЗВЛЕЧЕНИЕ МАССОВЫХ ДОЛЕЙ (для каждого вещества)
{'='*80}
{QA_NUTRIENT_PROMPT.format(text=example_text, substance=example_substance)}

{'='*80}
ПРОМПТ 3: ИЗВЛЕЧЕНИЕ СТАНДАРТА
{'='*80}
{QA_STANDARD_PROMPT.format(text=example_text)}

{'='*80}
ПРОМПТ 4: ИЗВЛЕЧЕНИЕ МАРКИ
{'='*80}
{QA_GRADE_PROMPT.format(text=example_text)}

{'='*80}
ПРОМПТ 5: ИЗВЛЕЧЕНИЕ КОЛИЧЕСТВ
{'='*80}
{QA_QUANTITY_PROMPT.format(text=example_text)}

{'='*80}
ПРОМПТ 6: СБОРКА ФИНАЛЬНОГО JSON
{'='*80}
(Автоматическая сборка из всех извлеченных данных)
""",
        "prompt_info": {
            "mode": "qa_workflow",
            "prompts_used": [
                "QA_NUTRIENTS_PROMPT",
                "QA_NUTRIENT_PROMPT",
                "QA_STANDARD_PROMPT",
                "QA_GRADE_PROMPT",
                "QA_QUANTITY_PROMPT"
            ],
            "example_nutrients_prompt": QA_NUTRIENTS_PROMPT.format(text=example_text),
            "example_nutrient_prompt": QA_NUTRIENT_PROMPT.format(text=example_text, substance=example_substance),
            "example_standard_prompt": QA_STANDARD_PROMPT.format(text=example_text),
            "example_grade_prompt": QA_GRADE_PROMPT.format(text=example_text),
            "example_quantity_prompt": QA_QUANTITY_PROMPT.format(text=example_text)
        },
        "description": "извлечение питательных веществ → массовые доли для каждого вещества → стандарт → марка → количества → сборка JSON"
    }


# Регистр всех доступных workflow режимов
WORKFLOW_CONFIGS: Dict[str, Dict[str, Any]] = {
    "simple_4agents": {
        "name": "simple_4agents",
        "get_prompts_func": _get_simple_4agents_prompts,
        "description": "4 агента: извлечение числовых фрагментов, массовые доли, прочие параметры, формирование JSON",
        "graph_creator": "create_simple_4agents_graph"  # Имя функции в multi_agent_graph.py
    },
    "critic_3agents": {
        "name": "critic_3agents",
        "get_prompts_func": _get_critic_3agents_prompts,
        "description": "3 агента: генератор (создает первоначальный ответ), критик (анализирует ответ на соответствие промпту), исправитель (устраняет найденные ошибки) (вдохновлено подходом из Reddit: 3agents)",
        "graph_creator": "create_critic_3agents_graph"  # Имя функции в multi_agent_graph.py
    },
    "qa_workflow": {
        "name": "qa_workflow",
        "get_prompts_func": _get_qa_workflow_prompts,
        "description": "6 агентов: извлечение питательных веществ → массовые доли для каждого вещества → стандарт → марка → количества → сборка JSON",
        "graph_creator": "create_qa_workflow_graph"  # Имя функции в multi_agent_graph.py
    }
}


def get_workflow_config(mode: str) -> Dict[str, Any]:
    """
    Получает конфигурацию workflow по режиму
    
    Args:
        mode: имя режима (например, "simple_4agents", "critic_3agents")
    
    Returns:
        Словарь с конфигурацией workflow
    
    Raises:
        ValueError: если режим не найден
    """
    if mode not in WORKFLOW_CONFIGS:
        available_modes = ", ".join(WORKFLOW_CONFIGS.keys())
        raise ValueError(f"Неизвестный режим мультиагентного подхода: {mode}. Доступные режимы: {available_modes}")
    
    return WORKFLOW_CONFIGS[mode]


def get_workflow_prompts(mode: str, example_text: str) -> Dict[str, Any]:
    """
    Получает примеры промптов для указанного режима
    
    Args:
        mode: имя режима
        example_text: пример текста для генерации примеров промптов
    
    Returns:
        Словарь с full_prompt_example, prompt_info и description
    """
    config = get_workflow_config(mode)
    get_prompts_func = config["get_prompts_func"]
    result = get_prompts_func(example_text)
    result["description"] = config.get("description", result.get("description", ""))
    return result


def register_workflow(
    mode: str,
    name: str,
    get_prompts_func: Callable[[str], Dict[str, Any]],
    description: str,
    graph_creator: str
):
    """
    Регистрирует новый workflow режим
    
    Args:
        mode: уникальный идентификатор режима
        name: отображаемое имя режима
        get_prompts_func: функция, которая принимает example_text и возвращает словарь с:
            - full_prompt_example: полный текст всех промптов
            - prompt_info: структурированная информация о промптах
            - description: описание последовательности агентов (опционально)
        description: описание режима для пользователя
        graph_creator: имя функции в multi_agent_graph.py, которая создает граф для этого режима
    
    Example:
        def my_custom_prompts(example_text: str):
            return {
                "full_prompt_example": "...",
                "prompt_info": {...},
                "description": "агент 1 → агент 2"
            }
        
        register_workflow(
            mode="my_custom_mode",
            name="my_custom_mode",
            get_prompts_func=my_custom_prompts,
            description="Мой кастомный режим",
            graph_creator="create_my_custom_graph"
        )
    """
    WORKFLOW_CONFIGS[mode] = {
        "name": name,
        "get_prompts_func": get_prompts_func,
        "description": description,
        "graph_creator": graph_creator
    }



