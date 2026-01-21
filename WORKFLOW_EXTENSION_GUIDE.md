# Руководство по добавлению новых Workflow режимов

Это руководство описывает, как добавить новый мультиагентный workflow режим в систему.

## Структура

Система workflow построена на модульной архитектуре:
- `workflow_config.py` - регистр всех доступных workflow режимов
- `multi_agent_graph.py` - функции создания графов LangGraph для каждого режима
- `model_evaluator.py` - использует конфигурацию для генерации промптов и отчетов

## Шаги для добавления нового workflow

### 1. Создайте функцию генерации промптов

В `workflow_config.py` создайте функцию, которая генерирует примеры промптов:

```python
def _get_my_custom_workflow_prompts(example_text: str) -> Dict[str, Any]:
    """Генерирует примеры промптов для my_custom_workflow"""
    from prompt_config import MY_CUSTOM_PROMPT_1, MY_CUSTOM_PROMPT_2
    
    return {
        "full_prompt_example": f"""МУЛЬТИАГЕНТНЫЙ РЕЖИМ: my_custom_workflow

{'='*80}
ПРОМПТ 1: НАЗВАНИЕ АГЕНТА 1
{'='*80}
{MY_CUSTOM_PROMPT_1.format(text=example_text)}

{'='*80}
ПРОМПТ 2: НАЗВАНИЕ АГЕНТА 2
{'='*80}
{MY_CUSTOM_PROMPT_2.format(previous_result="[Пример результата предыдущего шага]")}
""",
        "prompt_info": {
            "mode": "my_custom_workflow",
            "prompts_used": [
                "MY_CUSTOM_PROMPT_1",
                "MY_CUSTOM_PROMPT_2"
            ],
            "example_prompt_1": MY_CUSTOM_PROMPT_1.format(text=example_text),
            "example_prompt_2": MY_CUSTOM_PROMPT_2.format(previous_result="[Пример]")
        },
        "description": "агент 1 → агент 2"
    }
```

### 2. Создайте функцию создания графа

В `multi_agent_graph.py` создайте функцию, которая создает граф LangGraph:

```python
def create_my_custom_workflow_graph():
    """
    Создает граф LangGraph для my_custom_workflow
    
    Returns:
        Скомпилированный граф
    """
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("agent_1", my_agent_1_function)
    workflow.add_node("agent_2", my_agent_2_function)
    
    # Определяем граф
    workflow.set_entry_point("agent_1")
    workflow.add_edge("agent_1", "agent_2")
    workflow.add_edge("agent_2", END)
    
    return workflow.compile()
```

### 3. Зарегистрируйте workflow

В `workflow_config.py` добавьте запись в словарь `WORKFLOW_CONFIGS`:

```python
WORKFLOW_CONFIGS["my_custom_workflow"] = {
    "name": "my_custom_workflow",
    "get_prompts_func": _get_my_custom_workflow_prompts,
    "description": "Описание вашего workflow для пользователя",
    "graph_creator": "create_my_custom_workflow_graph"  # Имя функции из multi_agent_graph.py
}
```

Или используйте функцию регистрации:

```python
from workflow_config import register_workflow

register_workflow(
    mode="my_custom_workflow",
    name="my_custom_workflow",
    get_prompts_func=_get_my_custom_workflow_prompts,
    description="Описание вашего workflow",
    graph_creator="create_my_custom_workflow_graph"
)
```

### 4. Добавьте промпты в prompt_config.py (если нужны новые)

Если ваш workflow использует новые промпты, добавьте их в `prompt_config.py`:

```python
MY_CUSTOM_PROMPT_1 = """
Ваш промпт для агента 1...
"""

MY_CUSTOM_PROMPT_2 = """
Ваш промпт для агента 2...
{previous_result}
"""
```

## Пример полного workflow

Вот пример полного workflow с 2 агентами:

### 1. prompt_config.py

```python
MY_WORKFLOW_AGENT_1_PROMPT = """
Ты агент 1. Твоя задача: {task}
Текст: {text}
"""

MY_WORKFLOW_AGENT_2_PROMPT = """
Ты агент 2. Твоя задача: {task}
Результат агента 1: {agent1_result}
"""
```

### 2. multi_agent_graph.py

```python
def my_agent_1(state: AgentState) -> AgentState:
    """Агент 1"""
    generator = state.get("generator")
    text = state.get("text", "")
    
    prompt = MY_WORKFLOW_AGENT_1_PROMPT.format(task="извлечь данные", text=text)
    response = generator.generate(prompt=prompt, max_new_tokens=512)
    
    return {**state, "agent1_result": response}

def my_agent_2(state: AgentState) -> AgentState:
    """Агент 2"""
    generator = state.get("generator")
    agent1_result = state.get("agent1_result", "")
    
    prompt = MY_WORKFLOW_AGENT_2_PROMPT.format(task="обработать результат", agent1_result=agent1_result)
    response = generator.generate(prompt=prompt, max_new_tokens=512)
    
    return {**state, "json_result": response}

def create_my_workflow_graph():
    """Создает граф для my_workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_1", my_agent_1)
    workflow.add_node("agent_2", my_agent_2)
    workflow.set_entry_point("agent_1")
    workflow.add_edge("agent_1", "agent_2")
    workflow.add_edge("agent_2", END)
    return workflow.compile()
```

### 3. workflow_config.py

```python
def _get_my_workflow_prompts(example_text: str) -> Dict[str, Any]:
    from prompt_config import MY_WORKFLOW_AGENT_1_PROMPT, MY_WORKFLOW_AGENT_2_PROMPT
    
    return {
        "full_prompt_example": f"""МУЛЬТИАГЕНТНЫЙ РЕЖИМ: my_workflow

{'='*80}
ПРОМПТ 1: АГЕНТ 1
{'='*80}
{MY_WORKFLOW_AGENT_1_PROMPT.format(task="извлечь данные", text=example_text)}

{'='*80}
ПРОМПТ 2: АГЕНТ 2
{'='*80}
{MY_WORKFLOW_AGENT_2_PROMPT.format(task="обработать результат", agent1_result="[Пример результата агента 1]")}
""",
        "prompt_info": {
            "mode": "my_workflow",
            "prompts_used": ["MY_WORKFLOW_AGENT_1_PROMPT", "MY_WORKFLOW_AGENT_2_PROMPT"],
            "example_agent_1_prompt": MY_WORKFLOW_AGENT_1_PROMPT.format(task="извлечь данные", text=example_text),
            "example_agent_2_prompt": MY_WORKFLOW_AGENT_2_PROMPT.format(task="обработать результат", agent1_result="[Пример]")
        },
        "description": "агент 1 (извлечение) → агент 2 (обработка)"
    }

WORKFLOW_CONFIGS["my_workflow"] = {
    "name": "my_workflow",
    "get_prompts_func": _get_my_workflow_prompts,
    "description": "Мой кастомный workflow с 2 агентами",
    "graph_creator": "create_my_workflow_graph"
}
```

## Использование

После регистрации workflow его можно использовать:

```bash
python main.py qwen-2.5-3b --multi-agent my_workflow
```

## Примечания

- Имя функции создания графа в `multi_agent_graph.py` должно совпадать с `graph_creator` в конфигурации
- Функция генерации промптов должна возвращать словарь с ключами: `full_prompt_example`, `prompt_info`, `description`
- `prompt_info` должен содержать ключ `mode` с именем режима
- Описание агентов в `description` используется для вывода в консоль

