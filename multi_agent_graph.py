"""
Мультиагентная система для извлечения данных с использованием LangGraph
"""
import json
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
import time
from prompt_config import (
    NUMERIC_FRAGMENTS_EXTRACTION_PROMPT,
    MASS_FRACTION_EXTRACTION_PROMPT,
    OTHER_PARAMETERS_EXTRACTION_PROMPT,
    JSON_FORMATION_PROMPT,
    DETAILED_INSTR_ONESHOT,
    CRITIC_PROMPT,
    CORRECTOR_PROMPT,
    VALIDATION_FIX_PROMPT,
    QA_NUTRIENTS_PROMPT,
    QA_NUTRIENT_PROMPT,
    QA_STANDARD_PROMPT,
    QA_GRADE_PROMPT,
    QA_QUANTITY_PROMPT
)
from utils import extract_json_from_response, parse_json_safe, is_valid_json
from metrics import validate_with_pydantic
from structured_schemas import latin_to_cyrillic_output, LATIN_TO_CYRILLIC_KEYS


def _clean_repetitive_arrays(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Очищает JSON от повторяющихся значений в массивах.
    Удаляет массивы с сотнями одинаковых элементов.
    """
    if not isinstance(data, dict):
        return data
    
    cleaned = {}
    
    for key, value in data.items():
        if isinstance(value, list):
            cleaned_list = []
            seen_items = set()
            
            for item in value:
                if isinstance(item, dict):
                    # Создаем хеш для проверки уникальности
                    item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if item_str not in seen_items:
                        seen_items.add(item_str)
                        cleaned_list.append(item)
                else:
                    # Для не-словарей тоже проверяем уникальность
                    if item not in cleaned_list:
                        cleaned_list.append(item)
            
            # Если массив слишком длинный (более 20 элементов), обрезаем его
            if len(cleaned_list) > 20:
                # Берем первые 20 уникальных элементов
                cleaned_list = cleaned_list[:20]
            
            cleaned[key] = cleaned_list
        elif isinstance(value, dict):
            cleaned[key] = _clean_repetitive_arrays(value)
        else:
            cleaned[key] = value
    
    return cleaned


def _clean_none_values(parsed_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Удаляет из parsed_json записи с None или [None, None] значениями.
    Такие записи бессмысленны и занижают F1-метрику.
    
    Args:
        parsed_json: распарсенный JSON словарь
        
    Returns:
        dict: очищенный JSON словарь
    """
    if not isinstance(parsed_json, dict):
        return parsed_json
    
    cleaned = {}
    
    # Обрабатываем "массовая доля"
    if "массовая доля" in parsed_json:
        mass_fractions = parsed_json["массовая доля"]
        if isinstance(mass_fractions, list):
            cleaned_mass = []
            for item in mass_fractions:
                if isinstance(item, dict):
                    # Проверяем значение "массовая доля"
                    mass_value = item.get("массовая доля")
                    # Пропускаем записи с None
                    if mass_value is None:
                        continue
                    # Пропускаем [None, None]
                    if isinstance(mass_value, list) and len(mass_value) == 2:
                        if mass_value[0] is None and mass_value[1] is None:
                            continue
                    # Если все значения None, пропускаем
                    if isinstance(mass_value, list) and all(v is None for v in mass_value):
                        continue
                    cleaned_mass.append(item)
                else:
                    # Если это не словарь, оставляем как есть
                    cleaned_mass.append(item)
            cleaned["массовая доля"] = cleaned_mass
        else:
            cleaned["массовая доля"] = mass_fractions
    
    # Обрабатываем "прочее"
    if "прочее" in parsed_json:
        other_params = parsed_json["прочее"]
        if isinstance(other_params, list):
            cleaned_other = []
            for item in other_params:
                if isinstance(item, dict):
                    # Проверяем все значения в словаре
                    has_valid_value = False
                    for key, value in item.items():
                        if value is None:
                            continue
                        if isinstance(value, list):
                            # Пропускаем списки из None
                            if all(v is None for v in value):
                                continue
                            # Пропускаем [None, None]
                            if len(value) == 2 and value[0] is None and value[1] is None:
                                continue
                        # Если есть хотя бы одно непустое значение, оставляем запись
                        if value is not None and value != "":
                            has_valid_value = True
                            break
                    if has_valid_value:
                        cleaned_other.append(item)
                else:
                    # Если это не словарь, оставляем как есть
                    cleaned_other.append(item)
            cleaned["прочее"] = cleaned_other
        else:
            cleaned["прочее"] = other_params
    
    # Копируем остальные ключи, если есть
    for key in parsed_json:
        if key not in ["массовая доля", "прочее"]:
            cleaned[key] = parsed_json[key]
    
    return cleaned


class AgentState(TypedDict):
    """Состояние графа агентов"""
    text: str  # Исходный текст
    numeric_fragments: str  # Извлеченные числовые фрагменты
    numeric_fragments_raw: str  # Полный ответ агента 1 (для отладки)
    mass_fractions: str  # Извлеченные массовые доли
    other_parameters: str  # Извлеченные прочие параметры
    json_result: str  # Финальный JSON
    json_result_raw: str  # Полный ответ агента 4 (для отладки)
    json_parsed: dict  # Распарсенный JSON
    is_valid: bool  # Валидность JSON
    success: bool  # Успешность выполнения
    error: str  # Ошибка (если есть)
    time: float  # Время выполнения
    generator: object  # Генератор для использования
    # Поля для режима critic_3agents
    prompt: str  # Исходный промпт
    initial_response: str  # Первоначальный ответ агента 1
    critic_analysis: str  # Анализ критика
    corrected_response: str  # Исправленный ответ
    # Поля для режима qa_workflow
    nutrients: list  # Список питательных веществ
    nutrient_values: dict  # Словарь {вещество: значение} для массовых долей
    standard: str  # Стандарт
    grade: str  # Марка
    quantities: list  # Список количеств
    # Поля для validation_fix_2agents
    validation_errors: str  # Ошибки валидации Pydantic (для повторной подачи LLM)
    fix_prompt: str  # Промпт агента 2 (исправитель), чтобы в metrics JSON выводить сконкатенированный промпт
    prompt_template_name: str  # Название базового промпта (из --prompt или config) для генерации
    model_key: str  # Ключ модели (для few-shot и т.д.)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def print_agent_response(agent_num: int, response: str, prompt: str = None):
    """Выводит промпт и ответ агента в консоль"""
    if prompt:
        print(f"\n   📝 Промпт агента {agent_num}:")
        print(f"   {'─'*76}")
        for line in prompt.split('\n'):
            print(f"   {line}")
        print(f"   {'─'*76}")
    
    print(f"\n   📋 Ответ агента {agent_num}:")
    print(f"   {'─'*76}")
    if response:
        for line in response.split('\n'):
            print(f"   {line}")
    else:
        print(f"   (пустой ответ)")
    print(f"   {'─'*76}\n")


def extract_fragments_from_instructions(response: str) -> str:
    """Извлекает числовые фрагменты из ответа, содержащего инструкции"""
    instruction_keywords = [
        "пожалуйста выполните", "пожалуйста выполни", "выполните задание", "выполни задание",
        "пожалуйста найдите", "пожалуйста найди", "нужно найти", "необходимо найти",
        "твоя задача", "твоя цель"
    ]
    
    response_lower = response.lower() if response else ""
    if not any(keyword in response_lower for keyword in instruction_keywords):
        return response.strip() if response else ""
    
    lines = response.split('\n')
    extracted_lines = []
    skip_instruction_section = True
    
    for line in lines:
        line_lower = line.lower().strip()
        if any(keyword in line_lower for keyword in instruction_keywords):
            skip_instruction_section = True
            continue
        
        if skip_instruction_section:
            if (any(char.isdigit() for char in line) or 
                any(unit in line_lower for unit in ['%', 'кг', 'т', 'шт', 'л', 'меш', 'вагон', 'гост', 'ту', 'n', 'p', 'k', 's']) or
                'не найдено' in line_lower):
                skip_instruction_section = False
                extracted_lines.append(line)
        else:
            extracted_lines.append(line)
    
    result = '\n'.join(extracted_lines).strip()
    if result and result != response.strip():
        print(f"\n   ⚠️  ВНИМАНИЕ: Ответ агента 1 содержал инструкции вместо результата. Извлечены числовые фрагменты.")
    return result


def print_debug_info(title: str, **kwargs):
    """Выводит отладочную информацию в едином формате"""
    print(f"\n   {'─'*76}")
    print(f"   {title}")
    print(f"   {'─'*76}")
    for key, value in kwargs.items():
        if value is not None and value:
            key_display = key.replace('_', ' ').title()
            print(f"   {key_display}:")
            print(f"   {'─'*76}")
            if isinstance(value, str):
                for line in value.split('\n'):
                    print(f"   {line}")
            else:
                print(f"   {value}")
            print(f"   {'─'*76}")


def handle_agent_error(agent_num: int, error: Exception, elapsed: float, 
                       response: str = None, context_data: Dict[str, str] = None):
    """Обрабатывает ошибку агента и выводит детальную информацию"""
    import traceback
    error_type = type(error).__name__
    error_msg = str(error)
    
    print(f"❌ Ошибка ({elapsed:.2f}с): {error_type}: {error_msg[:100]}")
    
    title = f"🔍 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ОБ ОШИБКЕ АГЕНТА {agent_num}"
    
    debug_info = {
        "Тип ошибки": error_type,
        "Сообщение": error_msg
    }
    
    if context_data:
        debug_info.update(context_data)
    
    if response:
        debug_info["Ответ модели"] = response
    
    debug_info["Полный traceback"] = "\n".join(traceback.format_exc().split('\n'))
    
    print_debug_info(title, **debug_info)
    
    return {
        "success": False,
        "error": f"{error_type}: {error_msg}",
        "time": elapsed
    }


def run_agent_generation(state: AgentState, prompt: str, agent_num: int) -> tuple:
    """Выполняет генерацию ответа агента. max_new_tokens и доп. параметры берутся из state."""
    generator = state.get("generator")
    max_new_tokens = state.get("max_new_tokens", 1792)
    generator_kwargs = state.get("generator_kwargs") or {}
    start_time = time.time()
    try:
        response = generator.generate(prompt=prompt, max_new_tokens=max_new_tokens, **generator_kwargs)
        elapsed = time.time() - start_time
        return response, elapsed, None
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n   ⚠️  Прервано пользователем во время генерации агента {agent_num}")
        raise


def extract_numeric_fragments(state: AgentState) -> AgentState:
    """
    Агент 1: Извлечение числовых фрагментов из текста
    """
    print("   🤖 [Агент 1/4] Извлечение числовых фрагментов...", end=" ", flush=True)
    
    generator = state.get("generator")
    text = state.get("text", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided",
            "time": 0.0
        }
    
    try:
        prompt = NUMERIC_FRAGMENTS_EXTRACTION_PROMPT.format(text=text)
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        
        # Извлекаем фрагменты, убирая инструкции если есть
        numeric_fragments = extract_fragments_from_instructions(response)
        
        # Выводим промпт и ответ всегда
        print_agent_response(1, response, prompt)
        
        # Проверяем на пустой ответ
        if not numeric_fragments or not numeric_fragments.strip():
            print(f"⚠️ ({elapsed:.2f}с) - ПУСТОЙ ОТВЕТ")
            return {
                **state,
                "numeric_fragments": "",
                "numeric_fragments_raw": response if response else "(пустой ответ)",
                "success": False,
                "error": "Agent 1 returned empty response",
                "time": elapsed
            }
        
        print(f"✓ ({elapsed:.2f}с)")
        return {
            **state,
            "numeric_fragments": numeric_fragments,
            "numeric_fragments_raw": response if response else "(пустой ответ)",
            "success": True,
            "error": None,
            "time": elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(1, e, elapsed, 
                                         response if 'response' in locals() else None,
                                         {"Исходный_текст": text})
        return {**state, "numeric_fragments": "", "numeric_fragments_raw": "", **error_result}


def extract_mass_fractions(state: AgentState) -> AgentState:
    """
    Агент 2.1: Извлечение массовых долей из числовых фрагментов
    """
    print("   🤖 [Агент 2/4] Извлечение массовых долей...", end=" ", flush=True)
    
    generator = state.get("generator")
    numeric_fragments = state.get("numeric_fragments", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided"
        }
    
    if not numeric_fragments or "не найдено" in numeric_fragments.lower():
        print("⏭️  Пропущено (числовых фрагментов нет)")
        # Не выводим проблемный ответ агента 1 здесь, так как он уже был выведен после выполнения агента 1
        return {
            **state,
            "mass_fractions": "Массовых долей не найдено",
            "success": True
        }
    
    try:
        prompt = MASS_FRACTION_EXTRACTION_PROMPT.format(
            numeric_fragments=numeric_fragments
        )
        
        response, elapsed, _ = run_agent_generation(state, prompt, 2)
        mass_fractions = response.strip() if response else ""
        
        # Выводим промпт и ответ всегда
        print_agent_response(2, response, prompt)
        
        # Проверяем на пустой ответ
        if not mass_fractions or not mass_fractions.strip() or "не найдено" in mass_fractions.lower():
            print(f"⚠️ ({elapsed:.2f}с) - ПУСТОЙ ОТВЕТ")
            return {**state, "mass_fractions": "", "time": state.get("time", 0.0) + elapsed}
        
        print(f"✓ ({elapsed:.2f}с)")
        return {**state, "mass_fractions": mass_fractions, "time": state.get("time", 0.0) + elapsed}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(2, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Числовые_фрагменты": numeric_fragments})
        return {**state, "mass_fractions": "", **error_result}


def extract_other_parameters(state: AgentState) -> AgentState:
    """
    Агент 2.2: Извлечение прочих параметров из числовых фрагментов
    """
    print("   🤖 [Агент 3/4] Извлечение прочих параметров...", end=" ", flush=True)
    
    generator = state.get("generator")
    numeric_fragments = state.get("numeric_fragments", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided"
        }
    
    if not numeric_fragments or "не найдено" in numeric_fragments.lower():
        print("⏭️  Пропущено (числовых фрагментов нет)")
        # Не выводим проблемный ответ здесь, так как он уже был выведен в агенте 2
        return {
            **state,
            "other_parameters": "Прочих параметров не найдено",
            "success": True
        }
    
    try:
        prompt = OTHER_PARAMETERS_EXTRACTION_PROMPT.format(
            numeric_fragments=numeric_fragments
        )
        
        response, elapsed, _ = run_agent_generation(state, prompt, 3)
        other_parameters = response.strip() if response else ""
        
        # Выводим промпт и ответ всегда
        print_agent_response(3, response, prompt)
        
        # Проверяем на пустой ответ
        if not other_parameters or not other_parameters.strip() or "не найдено" in other_parameters.lower():
            print(f"⚠️ ({elapsed:.2f}с) - ПУСТОЙ ОТВЕТ")
            return {**state, "other_parameters": "", "time": state.get("time", 0.0) + elapsed}
        
        print(f"✓ ({elapsed:.2f}с)")
        return {**state, "other_parameters": other_parameters, "time": state.get("time", 0.0) + elapsed}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(3, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Числовые_фрагменты": numeric_fragments})
        return {**state, "other_parameters": "", **error_result}


def form_json(state: AgentState) -> AgentState:
    """
    Агент 4: Формирование финального JSON из извлеченных данных
    """
    print("   🤖 [Агент 4/4] Формирование JSON...", end=" ", flush=True)
    
    generator = state.get("generator")
    mass_fractions = state.get("mass_fractions", "")
    other_parameters = state.get("other_parameters", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided"
        }
    
    try:
        # Если данные пустые, указываем это явно
        if not mass_fractions or not mass_fractions.strip():
            mass_fractions = "(массовые доли не найдены)"
        if not other_parameters or not other_parameters.strip():
            other_parameters = "(прочие параметры не найдены)"
        
        prompt = JSON_FORMATION_PROMPT.format(
            mass_fractions=mass_fractions,
            other_parameters=other_parameters
        )
        
        response, elapsed, _ = run_agent_generation(state, prompt, 4)
        
        # Извлекаем JSON из ответа
        json_part = extract_json_from_response(response)
        parsed_json = parse_json_safe(json_part)
        
        # Валидация и очистка от повторяющихся значений
        if parsed_json:
            parsed_json = _clean_repetitive_arrays(parsed_json)
            # Очищаем от записей с None или [None, None]
            parsed_json = _clean_none_values(parsed_json)
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        is_valid = is_valid_json(json_part)
        status = "✓" if is_valid else "⚠️"
        print(f"{status} ({elapsed:.2f}с)")
        
        # Выводим промпт и ответ всегда
        print_agent_response(4, response, prompt)
        
        # Если JSON невалидный, дополнительно выводим предупреждение
        if not is_valid:
            print(f"\n   ⚠️  ВНИМАНИЕ: Ответ агента 4 содержит невалидный JSON!")
            print(f"   {'─'*76}\n")
        
        return {
            **state,
            "json_result": json_part,
            "json_result_raw": response,  # Сохраняем полный ответ для отладки
            "json_parsed": parsed_json,
            "is_valid": is_valid,
            "time": state.get("time", 0.0) + elapsed,
            "success": True
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(4, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Массовые_доли": mass_fractions,
                                          "Прочие_параметры": other_parameters,
                                          "Промпт_агента_4": prompt if 'prompt' in locals() else None})
        return {
            **state,
            "json_result": "",
            "json_result_raw": response if 'response' in locals() else "",
            "json_parsed": {},
            "is_valid": False,
            **error_result
        }


def should_continue_after_agent1(state: AgentState) -> str:
    """
    Проверяет, нужно ли продолжать выполнение после агента 1.
    Если агент 1 вернул пустой ответ, пропускаем остальных агентов.
    """
    success = state.get("success", False)
    numeric_fragments = state.get("numeric_fragments", "")
    
    # Если агент 1 не успешен или вернул пустой ответ, завершаем
    if not success or not numeric_fragments or not numeric_fragments.strip():
        print(f"\n   ⏭️  Агенты 2-4 будут пропущены (агент 1 вернул пустой ответ)")
        return "end"
    return "continue"


def should_continue_after_agent3(state: AgentState) -> str:
    """
    Проверяет, нужно ли продолжать выполнение после агента 3.
    Если и массовые доли, и прочие параметры пустые, нет смысла запускать агента 4.
    """
    mass_fractions = state.get("mass_fractions", "")
    other_parameters = state.get("other_parameters", "")
    
    # Проверяем, есть ли хотя бы что-то для формирования JSON
    has_mass_fractions = mass_fractions and mass_fractions.strip() and "не найдено" not in mass_fractions.lower()
    has_other_params = other_parameters and other_parameters.strip() and "не найдено" not in other_parameters.lower()
    
    # Если оба пустые, пропускаем агента 4
    if not has_mass_fractions and not has_other_params:
        print(f"\n   ⏭️  Агент 4 будет пропущен (массовые доли и прочие параметры пустые)")
        return "end"
    return "continue"


# ========== ФУНКЦИИ ДЛЯ РЕЖИМА CRITIC_3AGENTS ==========

def generate_initial_response(state: AgentState) -> AgentState:
    """
    Агент 1: Генерация первоначального ответа на основе промпта
    """
    print("   🤖 [Агент 1/3] Генерация ответа...", end=" ", flush=True)
    
    generator = state.get("generator")
    text = state.get("text", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided",
            "time": 0.0
        }
    
    try:
        # Используем тот же промпт, что и для одноагентного подхода
        from utils import build_prompt3
        prompt = build_prompt3(text)
        
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        
        # Извлекаем JSON из ответа, если есть
        json_part = extract_json_from_response(response)
        
        print(f"✅ ({elapsed:.2f}с)")
        print_agent_response(1, response, prompt)
        
        return {
            **state,
            "prompt": prompt,
            "initial_response": response,
            "json_result": json_part if json_part else response,
            "time": elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = time.time() - time.time()  # 0, так как ошибка произошла до завершения
        error_info = handle_agent_error(1, e, elapsed, context_data={"Исходный текст": text[:200]})
        return {
            **state,
            **error_info
        }


def critique_response(state: AgentState) -> AgentState:
    """
    Агент 2: Критик - анализ ответа на соответствие промпту
    """
    print("   🤖 [Агент 2/3] Анализ ответа (критик)...", end=" ", flush=True)
    
    generator = state.get("generator")
    prompt = state.get("prompt", "")
    initial_response = state.get("initial_response", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided",
            "time": 0.0
        }
    
    if not prompt or not initial_response:
        print("❌ Ошибка: Отсутствует промпт или ответ")
        return {
            **state,
            "success": False,
            "error": "Missing prompt or response",
            "time": 0.0
        }
    
    try:
        critic_prompt = CRITIC_PROMPT.format(prompt=prompt, response=initial_response)
        response, elapsed, _ = run_agent_generation(state, critic_prompt, 2)
        
        # Извлекаем JSON из ответа критика
        critic_json = extract_json_from_response(response)
        if not critic_json:
            critic_json = response
        
        print(f"✅ ({elapsed:.2f}с)")
        print_agent_response(2, response, critic_prompt)
        
        return {
            **state,
            "critic_analysis": response,
            "time": state.get("time", 0.0) + elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = time.time() - time.time()
        error_info = handle_agent_error(2, e, elapsed, context_data={"Промпт": prompt[:200], "Ответ": initial_response[:200]})
        return {
            **state,
            **error_info
        }


def correct_response(state: AgentState) -> AgentState:
    """
    Агент 3: Исправитель - устранение найденных ошибок
    """
    print("   🤖 [Агент 3/3] Исправление ошибок...", end=" ", flush=True)
    
    generator = state.get("generator")
    prompt = state.get("prompt", "")
    initial_response = state.get("initial_response", "")
    critic_analysis = state.get("critic_analysis", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided",
            "time": 0.0
        }
    
    if not prompt or not initial_response:
        print("❌ Ошибка: Отсутствует промпт или ответ")
        return {
            **state,
            "success": False,
            "error": "Missing prompt or response",
            "time": 0.0
        }
    
    try:
        # Проверяем, есть ли ошибки в анализе критика
        try:
            critic_json = parse_json_safe(critic_analysis)
            has_errors = critic_json.get("найдены_ошибки", True)
            
            # Если ошибок нет, возвращаем исходный ответ
            if not has_errors:
                print(f"✅ Ошибок не найдено, возвращаем исходный ответ")
                json_part = extract_json_from_response(initial_response)
                parsed_json = parse_json_safe(json_part if json_part else initial_response)
                # Очищаем от записей с None или [None, None]
                if parsed_json and isinstance(parsed_json, dict):
                    parsed_json = _clean_none_values(parsed_json)
                
                return {
                    **state,
                    "corrected_response": initial_response,
                    "json_result": json_part if json_part else initial_response,
                    "json_parsed": parsed_json,
                    "is_valid": is_valid_json(json_part if json_part else initial_response),
                    "success": True,
                    "time": state.get("time", 0.0)
                }
        except:
            # Если не удалось распарсить анализ критика, все равно пытаемся исправить
            pass
        
        # Если есть ошибки, исправляем
        corrector_prompt = CORRECTOR_PROMPT.format(
            prompt=prompt,
            original_response=initial_response,
            critic_analysis=critic_analysis
        )
        
        response, elapsed, _ = run_agent_generation(state, corrector_prompt, 3)
        
        # Извлекаем JSON из исправленного ответа
        json_part = extract_json_from_response(response)
        if not json_part:
            json_part = response
        
        # Парсим JSON
        parsed_json = parse_json_safe(json_part)
        # Очищаем от записей с None или [None, None]
        if parsed_json and isinstance(parsed_json, dict):
            parsed_json = _clean_none_values(parsed_json)
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        print(f"✅ ({elapsed:.2f}с)")
        print_agent_response(3, response, corrector_prompt)
        
        return {
            **state,
            "corrected_response": response,
            "json_result": json_part,
            "json_result_raw": response,
            "json_parsed": parsed_json,
            "is_valid": is_valid_json(json_part),
            "success": True,
            "time": state.get("time", 0.0) + elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = time.time() - time.time()
        error_info = handle_agent_error(3, e, elapsed, context_data={
            "Промпт": prompt[:200],
            "Исходный ответ": initial_response[:200],
            "Анализ критика": critic_analysis[:200]
        })
        return {
            **state,
            **error_info
        }


# ========== ФУНКЦИИ ДЛЯ РЕЖИМА VALIDATION_FIX_2AGENTS ==========

def generate_and_validate_agent(state: AgentState) -> AgentState:
    """
    Агент 1: Генерация ответа и валидация через Pydantic. При невалидном JSON
    заполняет validation_errors для передачи агенту-исправителю.
    """
    print("   🤖 [Агент 1/2] Генерация и валидация...", end=" ", flush=True)
    generator = state.get("generator")
    text = state.get("text", "")
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {**state, "success": False, "error": "Generator not provided", "time": 0.0}
    try:
        from utils import build_prompt3
        pt_name = state.get("prompt_template_name") or None
        model_key = state.get("model_key") or None
        prompt = build_prompt3(text, prompt_template_name=pt_name, model_key=model_key)
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        json_part = extract_json_from_response(response)
        if not json_part:
            json_part = response
        parsed_json = parse_json_safe(json_part)
        if parsed_json and isinstance(parsed_json, dict) and any(k in LATIN_TO_CYRILLIC_KEYS for k in parsed_json):
            parsed_json = latin_to_cyrillic_output(parsed_json)
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass
        validation = validate_with_pydantic(parsed_json if isinstance(parsed_json, dict) else {}, stage="parsed")
        is_valid = validation.get("is_valid", False)
        if is_valid:
            parsed = validation.get("validated_data") or parsed_json
            print(f"✅ Валидный JSON ({elapsed:.2f}с)")
            return {
                **state,
                "prompt": prompt,
                "initial_response": response,
                "json_result": json_part,
                "json_parsed": parsed if isinstance(parsed, dict) else (parsed_json or {}),
                "is_valid": True,
                "success": True,
                "time": elapsed,
            }
        errors_str = "\n".join(validation.get("errors", []))
        print(f"⚠️ Невалидный JSON, передача исправителю ({elapsed:.2f}с)")
        return {
            **state,
            "prompt": prompt,
            "initial_response": response,
            "json_result": json_part,
            "json_parsed": parsed_json if isinstance(parsed_json, dict) else {},
            "is_valid": False,
            "success": False,
            "validation_errors": errors_str,
            "time": elapsed,
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_info = handle_agent_error(1, e, elapsed, context_data={"Исходный текст": text[:200]})
        return {**state, **error_info}


def fix_validation_agent(state: AgentState) -> AgentState:
    """
    Агент 2: Исправление JSON по ошибкам валидации Pydantic (повторная подача в LLM).
    """
    print("   🤖 [Агент 2/2] Исправление по ошибкам валидации...", end=" ", flush=True)
    generator = state.get("generator")
    text = state.get("text", "")
    invalid_json = state.get("json_result", "")
    validation_errors = state.get("validation_errors", "")
    if not generator or not validation_errors:
        print("❌ Нет генератора или ошибок валидации")
        return {**state, "success": False, "error": "Missing generator or validation_errors", "time": state.get("time", 0.0)}
    try:
        original_prompt = state.get("prompt", "")
        # В original_prompt уже вшит {text} (через build_prompt3(...)).
        # VALIDATION_FIX_PROMPT добавляет {text} ещё раз в секции
        # "Исходный текст для извлечения данных:", поэтому нужно убрать дублирование.
        # Удаляем хвост начиная с "Текст для извлечения данных:" (если он есть).
        text_marker = "Текст для извлечения данных:"
        if text_marker in original_prompt:
            original_prompt_for_fix = original_prompt.split(text_marker, 1)[0].rstrip()
        else:
            # На случай, если в шаблоне другой вариант двоеточия/пробелов
            import re
            m = re.search(r"Текст для извлечения данных\s*[:：]\s*", original_prompt)
            original_prompt_for_fix = (original_prompt[: m.start()].rstrip() if m else original_prompt)
        fix_prompt = VALIDATION_FIX_PROMPT.format(
            original_prompt=original_prompt_for_fix,
            text=text,
            invalid_json=invalid_json,
            validation_errors=validation_errors,
        )
        response, elapsed, _ = run_agent_generation(state, fix_prompt, 2)
        json_part = extract_json_from_response(response)
        if not json_part:
            json_part = response
        parsed_json = parse_json_safe(json_part)
        if parsed_json and isinstance(parsed_json, dict) and any(k in LATIN_TO_CYRILLIC_KEYS for k in parsed_json):
            parsed_json = latin_to_cyrillic_output(parsed_json)
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass
        if parsed_json and isinstance(parsed_json, dict):
            parsed_json = _clean_none_values(parsed_json)
            try:
                json_part = json.dumps(parsed_json, ensure_ascii=False, indent=2)
            except Exception:
                pass
        validation = validate_with_pydantic(parsed_json if isinstance(parsed_json, dict) else {}, stage="parsed")
        is_valid = validation.get("is_valid", False)
        validated_data = validation.get("validated_data")
        total_time = state.get("time", 0.0) + elapsed
        if is_valid:
            print(f"✅ Исправлено ({elapsed:.2f}с)")
        else:
            print(f"⚠️ Всё ещё невалидно ({elapsed:.2f}с)")
        print_agent_response(2, response, fix_prompt)
        return {
            **state,
            "corrected_response": response,
            "json_result": json_part,
            "json_parsed": validated_data if validated_data else (parsed_json or {}),
            "is_valid": is_valid,
            "success": is_valid,
            "time": total_time,
            "fix_prompt": fix_prompt,
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_info = handle_agent_error(2, e, elapsed, context_data={
            "Текст": text[:200],
            "Невалидный JSON": invalid_json[:300],
            "Ошибки": validation_errors[:300],
        })
        return {**state, **error_info}


def _validation_fix_route(state: AgentState) -> str:
    """Маршрутизация после generate_and_validate: при невалидном JSON идём к исправителю."""
    if state.get("is_valid", False):
        return "end"
    if state.get("validation_errors"):
        return "fix_validation"
    return "end"


# ========== ФУНКЦИИ АГЕНТОВ ДЛЯ QA WORKFLOW ==========

def extract_nutrients(state: AgentState) -> AgentState:
    """
    Агент 1: Извлечение списка питательных веществ из текста
    """
    print("   🤖 [QA Агент 1/6] Извлечение питательных веществ...", end=" ", flush=True)
    
    generator = state.get("generator")
    text = state.get("text", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {
            **state,
            "success": False,
            "error": "Generator not provided",
            "time": 0.0
        }
    
    try:
        prompt = QA_NUTRIENTS_PROMPT.format(text=text)
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        
        # Извлекаем JSON из ответа
        json_str = extract_json_from_response(response)
        nutrients = []
        
        if json_str:
            try:
                parsed = parse_json_safe(json_str)
                if isinstance(parsed, list):
                    nutrients = parsed
                elif isinstance(parsed, dict) and "значение" in parsed:
                    nutrients = parsed["значение"] if isinstance(parsed["значение"], list) else []
            except:
                pass
        
        print_agent_response(1, response, prompt)
        
        if not nutrients:
            print(f"⚠️ ({elapsed:.2f}с) - ПИТАТЕЛЬНЫЕ ВЕЩЕСТВА НЕ НАЙДЕНЫ")
            nutrients = []
        
        print(f"✓ ({elapsed:.2f}с) - Найдено веществ: {len(nutrients)}")
        return {
            **state,
            "nutrients": nutrients,
            "nutrient_values": {},
            "time": state.get("time", 0.0) + elapsed,
            "success": True
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(1, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Исходный_текст": text})
        return {**state, "nutrients": [], "nutrient_values": {}, **error_result}


def extract_all_nutrient_values(state: AgentState) -> AgentState:
    """
    Агент 2: Извлечение массовых долей для всех найденных питательных веществ
    """
    generator = state.get("generator")
    text = state.get("text", "")
    nutrients = state.get("nutrients", [])
    nutrient_values = state.get("nutrient_values", {})
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {**state, "success": False, "error": "Generator not provided"}
    
    if not nutrients:
        print("   ⏭️  [QA Агент 2/6] Пропущено (питательные вещества не найдены)")
        return {**state, "nutrient_values": {}}
    
    print(f"   🤖 [QA Агент 2/6] Извлечение массовых долей для {len(nutrients)} веществ...")
    
    total_elapsed = 0.0
    processed_count = 0
    
    for i, substance in enumerate(nutrients, 1):
        if not isinstance(substance, str):
            continue
        
        try:
            print(f"   [{i}/{len(nutrients)}] Обработка вещества: {substance}...", end=" ", flush=True)
            prompt = QA_NUTRIENT_PROMPT.format(text=text, substance=substance)
            response, elapsed, _ = run_agent_generation(state, prompt, 2)
            total_elapsed += elapsed
            
            # Извлекаем JSON из ответа
            json_str = extract_json_from_response(response)
            value = None
            
            if json_str:
                try:
                    parsed = parse_json_safe(json_str)
                    # Формат: {"вещество": "N", "массовая доля": 20}
                    if isinstance(parsed, dict) and "массовая доля" in parsed:
                        value = parsed["массовая доля"]
                except Exception as parse_err:
                    print(f"⚠️ Ошибка парсинга: {parse_err}")
            
            nutrient_values[substance] = value
            processed_count += 1
            
            # Выводим промпт и ответ для первого вещества, для остальных только краткую информацию
            if i == 1:
                print_agent_response(2, f"[Вещество {i}/{len(nutrients)}: {substance}]\n{response}", prompt)
            else:
                # Для остальных веществ выводим только результат
                if value is not None:
                    print(f"✓ значение: {value}")
                else:
                    print(f"⚠️ не найдено")
            
        except KeyboardInterrupt:
            print(f"\n   ⚠️  Прервано пользователем при обработке вещества {substance}")
            nutrient_values[substance] = None
            raise
        except Exception as e:
            nutrient_values[substance] = None
            print(f"❌ Ошибка: {e}")
            # Продолжаем обработку остальных веществ даже при ошибке
    
    print(f"   ✓ Обработано веществ: {processed_count}/{len(nutrients)} ({total_elapsed:.2f}с)")
    return {
        **state,
        "nutrient_values": nutrient_values,
        "time": state.get("time", 0.0) + total_elapsed
    }


def extract_standard(state: AgentState) -> AgentState:
    """
    Агент 3: Извлечение стандарта
    """
    print("   🤖 [QA Агент 3/6] Извлечение стандарта...", end=" ", flush=True)
    
    generator = state.get("generator")
    text = state.get("text", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {**state, "success": False, "error": "Generator not provided"}
    
    try:
        prompt = QA_STANDARD_PROMPT.format(text=text)
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        
        # Извлекаем JSON из ответа
        json_str = extract_json_from_response(response)
        standard = None
        
        if json_str:
            try:
                parsed = parse_json_safe(json_str)
                # Формат: {"параметр": "стандарт", "значение": "ТУ..."}
                if isinstance(parsed, dict) and "параметр" in parsed and parsed["параметр"] == "стандарт" and "значение" in parsed:
                    standard = parsed["значение"]
            except:
                pass
        
        print_agent_response(3, response, prompt)
        
        if standard is None:
            print(f"⚠️ ({elapsed:.2f}с) - СТАНДАРТ НЕ НАЙДЕН")
        else:
            print(f"✓ ({elapsed:.2f}с)")
        
        return {
            **state,
            "standard": standard,
            "time": state.get("time", 0.0) + elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(3, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Исходный_текст": text})
        return {**state, "standard": None, **error_result}


def extract_grade(state: AgentState) -> AgentState:
    """
    Агент 4: Извлечение марки
    """
    print("   🤖 [QA Агент 4/6] Извлечение марки...", end=" ", flush=True)
    
    generator = state.get("generator")
    text = state.get("text", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {**state, "success": False, "error": "Generator not provided"}
    
    try:
        prompt = QA_GRADE_PROMPT.format(text=text)
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        
        # Извлекаем JSON из ответа
        json_str = extract_json_from_response(response)
        grade = None
        
        if json_str:
            try:
                parsed = parse_json_safe(json_str)
                # Формат: {"параметр": "марка", "значение": "N7-P20-K30-S3"}
                if isinstance(parsed, dict) and "параметр" in parsed and parsed["параметр"] == "марка" and "значение" in parsed:
                    grade = parsed["значение"]
            except:
                pass
        
        print_agent_response(4, response, prompt)
        
        if grade is None:
            print(f"⚠️ ({elapsed:.2f}с) - МАРКА НЕ НАЙДЕНА")
        else:
            print(f"✓ ({elapsed:.2f}с)")
        
        return {
            **state,
            "grade": grade,
            "time": state.get("time", 0.0) + elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(4, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Исходный_текст": text})
        return {**state, "grade": None, **error_result}


def extract_quantities(state: AgentState) -> AgentState:
    """
    Агент 5: Извлечение количеств (может быть несколько)
    """
    print("   🤖 [QA Агент 5/6] Извлечение количеств...", end=" ", flush=True)
    
    generator = state.get("generator")
    text = state.get("text", "")
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {**state, "success": False, "error": "Generator not provided"}
    
    try:
        prompt = QA_QUANTITY_PROMPT.format(text=text)
        response, elapsed, _ = run_agent_generation(state, prompt, 1)
        
        # Извлекаем JSON из ответа
        json_str = extract_json_from_response(response)
        quantities = []
        
        if json_str:
            try:
                parsed = parse_json_safe(json_str)
                if isinstance(parsed, dict):
                    # Формат: {"параметр": "масса нетто единицы", "масса": 50, "единица": "кг"}
                    # или {"параметр": "количество мешков", "количество": 1000, "единица": "шт"}
                    if "параметр" in parsed:
                        # Проверяем, что есть хотя бы одно значение (масса, количество или объем)
                        if ("масса" in parsed and parsed["масса"] is not None) or \
                           ("количество" in parsed and parsed["количество"] is not None) or \
                           ("объем" in parsed and parsed["объем"] is not None):
                            quantities.append(parsed)
                elif isinstance(parsed, list):
                    # Если это список объектов
                    quantities = parsed
            except:
                pass
        
        print_agent_response(5, response, prompt)
        
        if not quantities:
            print(f"⚠️ ({elapsed:.2f}с) - КОЛИЧЕСТВА НЕ НАЙДЕНЫ")
        else:
            print(f"✓ ({elapsed:.2f}с) - Найдено количеств: {len(quantities)}")
        
        return {
            **state,
            "quantities": quantities,
            "time": state.get("time", 0.0) + elapsed
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(5, e, elapsed,
                                         response if 'response' in locals() else None,
                                         {"Исходный_текст": text})
        return {**state, "quantities": [], **error_result}


def assemble_qa_json(state: AgentState) -> AgentState:
    """
    Агент 6: Сборка финального JSON из всех извлеченных данных
    """
    print("   🤖 [QA Агент 6/6] Сборка финального JSON...", end=" ", flush=True)
    
    generator = state.get("generator")
    nutrients = state.get("nutrients", [])
    nutrient_values = state.get("nutrient_values", {})
    standard = state.get("standard")
    grade = state.get("grade")
    quantities = state.get("quantities", [])
    
    if not generator:
        print("❌ Ошибка: Generator not provided")
        return {**state, "success": False, "error": "Generator not provided"}
    
    try:
        # Формируем массовые доли
        mass_fractions = []
        for substance in nutrients:
            if isinstance(substance, str) and substance in nutrient_values:
                value = nutrient_values[substance]
                if value is not None:
                    mass_fractions.append({
                        "вещество": substance,
                        "массовая доля": value
                    })
        
        # Формируем прочие параметры
        other_params = []
        
        # Добавляем количества
        for qty in quantities:
            if isinstance(qty, dict) and "параметр" in qty:
                param_name = qty["параметр"]
                
                # Если есть поле "масса"
                if "масса" in qty and qty["масса"] is not None:
                    other_params.append({
                        "параметр": param_name,
                        "масса": qty["масса"],
                        "единица": qty.get("единица")
                    })
                # Если есть поле "количество"
                elif "количество" in qty and qty["количество"] is not None:
                    other_params.append({
                        "параметр": param_name,
                        "количество": qty["количество"],
                        "единица": qty.get("единица")
                    })
                # Если есть поле "объем"
                elif "объем" in qty and qty["объем"] is not None:
                    other_params.append({
                        "параметр": param_name,
                        "объем": qty["объем"],
                        "единица": qty.get("единица")
                    })
                # Если есть поле "значение" (для стандарта и марки)
                elif "значение" in qty:
                    other_params.append({
                        "параметр": param_name,
                        "значение": qty["значение"]
                    })
        
        # Добавляем стандарт
        if standard:
            other_params.append({
                "параметр": "стандарт",
                "значение": standard
            })
        
        # Добавляем марку
        if grade:
            other_params.append({
                "параметр": "марка",
                "значение": grade
            })
        
        # Формируем финальный JSON
        result_json = {
            "массовая доля": mass_fractions,
            "прочее": other_params
        }
        
        # Очищаем от записей с None или [None, None]
        result_json = _clean_none_values(result_json)
        
        json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
        
        # Проверяем валидность
        is_valid = is_valid_json(json_str)
        
        print(f"✓")
        
        # Выводим результирующий JSON
        print(f"\n   📄 Результирующий JSON:")
        print(f"   {'─'*76}")
        # Выводим JSON с отступами для читаемости
        for line in json_str.split('\n'):
            print(f"   {line}")
        print(f"   {'─'*76}\n")
        
        return {
            **state,
            "json_result": json_str,
            "json_result_raw": json_str,
            "json_parsed": result_json,
            "is_valid": is_valid,
            "success": True,
            "error": None
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = 0.0
        error_result = handle_agent_error(6, e, elapsed, None, {})
        return {**state, "json_result": "", "json_parsed": {}, "is_valid": False, **error_result}


def create_qa_workflow_graph():
    """
    Создает граф LangGraph для QA workflow:
    1. Извлечение питательных веществ
    2. Извлечение массовых долей для каждого вещества
    3. Извлечение стандарта
    4. Извлечение марки
    5. Извлечение количеств
    6. Сборка финального JSON
    """
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("extract_nutrients", extract_nutrients)
    workflow.add_node("extract_all_nutrient_values", extract_all_nutrient_values)
    workflow.add_node("extract_standard", extract_standard)
    workflow.add_node("extract_grade", extract_grade)
    workflow.add_node("extract_quantities", extract_quantities)
    workflow.add_node("assemble_qa_json", assemble_qa_json)
    
    # Определяем граф
    workflow.set_entry_point("extract_nutrients")
    
    # Последовательное выполнение всех агентов
    workflow.add_edge("extract_nutrients", "extract_all_nutrient_values")
    workflow.add_edge("extract_all_nutrient_values", "extract_standard")
    workflow.add_edge("extract_standard", "extract_grade")
    workflow.add_edge("extract_grade", "extract_quantities")
    workflow.add_edge("extract_quantities", "assemble_qa_json")
    workflow.add_edge("assemble_qa_json", END)
    
    return workflow.compile()


def create_simple_4agents_graph():
    """
    Создает граф LangGraph для мультиагентной обработки с 4 агентами:
    1. Извлечение числовых фрагментов
    2. Извлечение массовых долей
    3. Извлечение прочих параметров
    4. Формирование JSON
    """
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("extract_numeric_fragments", extract_numeric_fragments)
    workflow.add_node("extract_mass_fractions", extract_mass_fractions)
    workflow.add_node("extract_other_parameters", extract_other_parameters)
    workflow.add_node("form_json", form_json)
    
    # Определяем граф
    workflow.set_entry_point("extract_numeric_fragments")
    
    # После извлечения числовых фрагментов проверяем, нужно ли продолжать
    workflow.add_conditional_edges(
        "extract_numeric_fragments",
        should_continue_after_agent1,
        {
            "continue": "extract_mass_fractions",
            "end": END
        }
    )
    
    # После извлечения массовых долей и прочих параметров последовательно обрабатываем
    workflow.add_edge("extract_mass_fractions", "extract_other_parameters")
    
    # После агента 3 проверяем, есть ли данные для формирования JSON
    workflow.add_conditional_edges(
        "extract_other_parameters",
        should_continue_after_agent3,
        {
            "continue": "form_json",
            "end": END
        }
    )
    
    # После формирования JSON завершаем
    workflow.add_edge("form_json", END)
    
    return workflow.compile()


def create_critic_3agents_graph():
    """
    Создает граф LangGraph для мультиагентной обработки с 3 агентами:
    1. Генератор - создает первоначальный ответ на основе промпта
    2. Критик - анализирует ответ на соответствие промпту
    3. Исправитель - устраняет найденные ошибки
    """
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("generate_initial_response", generate_initial_response)
    workflow.add_node("critique_response", critique_response)
    workflow.add_node("correct_response", correct_response)
    
    # Определяем граф
    workflow.set_entry_point("generate_initial_response")
    
    # Последовательное выполнение: генератор -> критик -> исправитель
    workflow.add_edge("generate_initial_response", "critique_response")
    workflow.add_edge("critique_response", "correct_response")
    workflow.add_edge("correct_response", END)
    
    return workflow.compile()


def create_validation_fix_2agents_graph():
    """
    Граф: генерация ответа -> валидация -> при ошибках валидации повторная подача
    ошибок в LLM (агент-исправитель), иначе конец.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_and_validate", generate_and_validate_agent)
    workflow.add_node("fix_validation", fix_validation_agent)
    workflow.set_entry_point("generate_and_validate")
    workflow.add_conditional_edges(
        "generate_and_validate",
        _validation_fix_route,
        {"fix_validation": "fix_validation", "end": END}
    )
    workflow.add_edge("fix_validation", END)
    return workflow.compile()


def create_multi_agent_graph(mode: str = "simple_4agents"):
    """
    Создает граф LangGraph для мультиагентной обработки
    
    Args:
        mode: режим мультиагентного подхода
    
    Returns:
        Скомпилированный граф
    
    Raises:
        ValueError: если режим не найден или функция создания графа не найдена
    """
    from workflow_config import get_workflow_config
    
    # Получаем конфигурацию workflow
    config = get_workflow_config(mode)
    graph_creator_name = config.get("graph_creator")
    
    if not graph_creator_name:
        raise ValueError(f"Для режима {mode} не указана функция создания графа")
    
    # Получаем функцию создания графа по имени
    graph_creator = globals().get(graph_creator_name)
    
    if graph_creator is None:
        raise ValueError(f"Функция {graph_creator_name} не найдена в multi_agent_graph.py для режима {mode}")
    
    # Вызываем функцию создания графа
    return graph_creator()


def process_with_multi_agent(
    text: str,
    generator,
    max_new_tokens: int = 1792,
    multi_agent_mode: str = "simple_4agents",
    hyperparameters: dict = None,
) -> dict:
    """
    Обрабатывает текст с использованием мультиагентного подхода.
    Гиперпараметры (max_new_tokens, repetition_penalty и т.д.) совпадают со стандартным режимом.
    """
    graph = create_multi_agent_graph(mode=multi_agent_mode)
    hp = hyperparameters or {}
    generator_kwargs = {}
    if hp.get("repetition_penalty") is not None:
        generator_kwargs["repetition_penalty"] = hp["repetition_penalty"]
    if hp.get("max_length") is not None:
        generator_kwargs["max_length"] = hp["max_length"]

    initial_state: AgentState = {
        "text": text,
        "numeric_fragments": "",
        "numeric_fragments_raw": "",
        "mass_fractions": "",
        "other_parameters": "",
        "json_result": "",
        "json_result_raw": "",
        "json_parsed": {},
        "is_valid": False,
        "success": False,
        "error": None,
        "time": 0.0,
        "generator": generator,
        "max_new_tokens": max_new_tokens,
        "generator_kwargs": generator_kwargs,
        # Поля для critic_3agents
        "prompt": "",
        "initial_response": "",
        "critic_analysis": "",
        "corrected_response": "",
        # Поля для qa_workflow
        "nutrients": [],
        "nutrient_values": {},
        "standard": None,
        "grade": None,
        "quantities": [],
        "validation_errors": "",
        "fix_prompt": "",
        "prompt_template_name": hp.get("prompt_template_name") or "",
        "model_key": hp.get("model_key") or ""
    }
    
    try:
        # Запускаем граф
        total_start_time = time.time()
        final_state = graph.invoke(initial_state)
        total_elapsed = time.time() - total_start_time
        
        # Проверяем, был ли пропущен остальной граф из-за пустого ответа агента 1
        if not final_state.get("success", False) and not final_state.get("numeric_fragments", "").strip():
            print(f"   ⏭️  Агенты 2-4 пропущены (агент 1 вернул пустой ответ)")
            print(f"   ⏱️  Общее время мультиагентной обработки: {total_elapsed:.2f}с")
            
            # Возвращаем пустой результат
            return {
                "text": text,
                "response": "",
                "json": "",
                "json_parsed": {},
                "is_valid": False,
                "success": False,
                "error": "Agent 1 returned empty response",
                "time": total_elapsed,
                "numeric_fragments": "",
                "mass_fractions": "",
                "other_parameters": "",
                "prompt": final_state.get("prompt", ""),
                "fix_prompt": "",
                "initial_response": final_state.get("initial_response", ""),
                "validation_errors": final_state.get("validation_errors", ""),
            }
        
        # Проверяем, был ли пропущен агент 4 из-за пустых ответов агентов 2 и 3
        mass_fractions = final_state.get("mass_fractions", "")
        other_parameters = final_state.get("other_parameters", "")
        json_result = final_state.get("json_result", "")
        
        has_mass_fractions = mass_fractions and mass_fractions.strip() and "не найдено" not in mass_fractions.lower()
        has_other_params = other_parameters and other_parameters.strip() and "не найдено" not in other_parameters.lower()
        
        if not json_result and not has_mass_fractions and not has_other_params:
            print(f"   ⏭️  Агент 4 пропущен (массовые доли и прочие параметры пустые)")
            print(f"   ⏱️  Общее время мультиагентной обработки: {total_elapsed:.2f}с")
            
            # Возвращаем пустой результат
            return {
                "text": text,
                "response": "",
                "json": "",
                "json_parsed": {},
                "is_valid": False,
                "success": False,
                "error": "Agents 2 and 3 returned empty responses",
                "time": total_elapsed,
                "numeric_fragments": final_state.get("numeric_fragments", ""),
                "mass_fractions": "",
                "other_parameters": "",
                "prompt": final_state.get("prompt", ""),
                "fix_prompt": "",
                "initial_response": final_state.get("initial_response", ""),
                "validation_errors": final_state.get("validation_errors", ""),
            }
        
        # Выводим итоговую информацию
        print(f"   ⏱️  Общее время мультиагентной обработки: {total_elapsed:.2f}с")
        
        return {
            "text": text,
            "response": final_state.get("json_result", ""),
            "json": final_state.get("json_result", ""),
            "json_parsed": final_state.get("json_parsed", {}),
            "is_valid": final_state.get("is_valid", False),
            "success": final_state.get("success", False),
            "error": final_state.get("error"),
            "time": final_state.get("time", 0.0),
            "numeric_fragments": final_state.get("numeric_fragments", ""),
            "mass_fractions": final_state.get("mass_fractions", ""),
            "other_parameters": final_state.get("other_parameters", ""),
            "prompt": final_state.get("prompt", ""),
            "fix_prompt": final_state.get("fix_prompt", ""),
            "initial_response": final_state.get("initial_response", ""),
            "validation_errors": final_state.get("validation_errors", ""),
        }
    except Exception as e:
        return {
            "text": text,
            "response": "",
            "json": "",
            "json_parsed": {},
            "is_valid": False,
            "success": False,
            "error": str(e),
            "time": 0.0,
            "prompt": "",
            "fix_prompt": "",
            "initial_response": "",
            "validation_errors": "",
        }

