# Evaluation of small LLM - Обработка данных по удобрениям

Проект для автоматизированной оценки различных LLM моделей на датасете по обработке данных об удобрениях.

## Описание

Проект позволяет:
- Загружать и оценивать различные LLM модели из Hugging Face
- Автоматически измерять метрики производительности (скорость, использование памяти)
- Оценивать качество извлечения данных (массовая доля, прочее)
- Анализировать ошибки парсинга JSON
- Получать рекомендации от Gemini API по улучшению моделей
- Переоценивать результаты из сохраненных файлов без повторного запуска модели
- Использовать одноагентный или мультиагентный подход для извлечения данных (на основе LangGraph)

## Архитектура системы

Проект поддерживает два подхода к извлечению данных:

### 1. Одноагентный подход (по умолчанию)

Использует единый промпт для извлечения всех данных за один проход:
- Один запрос к модели с полным промптом
- Модель извлекает все данные (массовые доли и прочие параметры) одновременно
- Формирует JSON ответ

### 2. Мультиагентный подход (LangGraph)

Разделяет задачу на несколько специализированных агентов:

1. **Агент извлечения числовых фрагментов** - находит все числовые упоминания в тексте
2. **Агент извлечения массовых долей** - извлекает только информацию о массовых долях элементов
3. **Агент извлечения прочих параметров** - извлекает массу, количество, стандарты и другие параметры
4. **Агент формирования JSON** - объединяет результаты в финальный JSON

Преимущества мультиагентного подхода:
- Разделение ответственности между агентами
- Более точное извлечение данных за счет специализации
- Возможность независимой настройки каждого агента

### Модуль `core/`

Модуль `core/` содержит минимальную ООП архитектуру для генераторов:

- **`base.py`** - базовый абстрактный класс:
  - `BaseGenerator` - абстрактный класс для генерации текста

- **`generators.py`** - реализация генератора:
  - `StandardGenerator` - стандартная генерация (используется в мультиагентном подходе)

**Текущее использование:**
- `BaseGenerator` и `StandardGenerator` из `core/` используются только в мультиагентном подходе (когда `use_multi_agent=True`)
- Основной код использует функциональный подход из `model_loaders.py`

## Установка

1. Клонируйте репозиторий или скопируйте проект

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте API ключи:

Создайте файл `config_secrets.py` на основе примера:
```bash
cp config_secrets.py.example config_secrets.py
```

Затем откройте `config_secrets.py` и заполните свои ключи:
```python
HF_TOKEN = "your_huggingface_token_here"
GEMINI_API_KEY = "your_gemini_api_key_here"  # опционально
OPENAI_API_KEY = "your_openai_api_key_here"  # опционально
```

**Важно:**
- `HF_TOKEN` - ОБЯЗАТЕЛЬНЫЙ. Без него программа не запустится (нужен для загрузки моделей с Hugging Face).
- `GEMINI_API_KEY` - опциональный, но рекомендуется для:
  - Анализа ошибок через Gemini API (автоматические рекомендации по улучшению)
  - Запуска API моделей Gemma 3 (gemma-3-4b-api, gemma-3-12b-api, gemma-3-27b-api)
  - Без ключа будет пропущен анализ ошибок, но локальные модели будут работать
- `OPENAI_API_KEY` - опциональный, требуется для:
  - Запуска моделей через OpenRouter API (deepseek-r1t-chimera-api, mistral-small-3.1-24b-api)
  - Получите ключ на https://openrouter.ai/
- Файл `config_secrets.py` автоматически игнорируется git (добавлен в .gitignore) для безопасности.

**Альтернатива:** Если не хотите использовать файл, можно установить переменные окружения:
- Windows: `set HF_TOKEN=your_token_here`
- Linux/Mac: `export HF_TOKEN=your_token_here`

**Где получить ключи:**
- Hugging Face Token: https://huggingface.co/settings/tokens
- Google Generative AI API Key (Gemini API Key): https://aistudio.google.com/app/apikey
  - Используется для анализа ошибок и для запуска API моделей Gemma 3
- OpenRouter API Key: https://openrouter.ai/keys
  - Используется для запуска моделей через OpenRouter API (например, deepseek-r1t-chimera-api, mistral-small-3.1-24b-api)

## Структура проекта

```
SmallLLMEvaluator/
├── main.py                 # Основной скрипт для запуска оценки одной модели
├── run_all_models.py       # Скрипт для запуска оценки всех моделей
├── reevaluate.py           # Скрипт для переоценки результатов из файла
├── model_evaluator.py      # Класс для оценки моделей
├── model_loaders.py        # Функции загрузки различных моделей
├── metrics.py              # Функции расчета метрик качества
├── utils.py                # Утилиты для парсинга JSON и построения промптов
├── gemini_analyzer.py      # Интеграция с Gemini API для анализа ошибок
├── gpu_info.py             # Функции для получения информации о GPU
├── config.py               # Конфигурация проекта
├── prompt_config.py        # Конфигурация промптов для агентов
├── multi_agent_graph.py    # Мультиагентная система на LangGraph
├── requirements.txt        # Зависимости проекта
├── core/                   # ООП архитектура для генераторов
│   ├── __init__.py
│   ├── base.py             # Базовый абстрактный класс BaseGenerator
│   └── generators.py       # Реализация StandardGenerator
└── README.md               # Документация
```

## Использование

### Оценка одной модели

Запустите `main.py` с указанием модели:

```bash
python main.py <model_key>
```

Например:
```bash
python main.py qwen-2.5-3b
```

**Запуск в мультиагентном режиме:**

Для запуска модели в мультиагентном режиме используйте флаг `--multi-agent`:

```bash
python main.py <model_key> --multi-agent <mode>
```

Например:
```bash
python main.py qwen-2.5-3b --multi-agent simple_4agents
```

**Доступные режимы мультиагентного подхода:**
- `simple_4agents` - 4 агента: извлечение числовых фрагментов, массовые доли, прочие параметры, формирование JSON
- `critic_3agents` - 3 агента: генератор (создает первоначальный ответ), критик (анализирует ответ на соответствие промпту), исправитель (устраняет найденные ошибки) (вдохновлено подходом из Reddit: 3agents)

**Примеры запуска:**

```bash
# Одноагентный режим (по умолчанию)
python main.py qwen-2.5-3b

# Мультиагентный режим
python main.py qwen-2.5-3b --multi-agent simple_4agents
python main.py qwen-2.5-3b --multi-agent critic_3agents
```

Доступные модели можно посмотреть, запустив `main.py` без аргументов.

### Выбор подхода (одноагентный/мультиагентный)

Подход задается через аргумент командной строки `--multi-agent`.

**Одноагентный подход (по умолчанию):**
- Запуск без флага `--multi-agent`:
```bash
python main.py qwen-2.5-3b
```

**Мультиагентный подход:**
- Запуск с флагом `--multi-agent`:
```bash
python main.py qwen-2.5-3b --multi-agent simple_4agents
```

**Доступные режимы мультиагентного подхода:**
- `simple_4agents` - 4 агента: извлечение числовых фрагментов, массовые доли, прочие параметры, формирование JSON

### Сравнение подходов

**Одноагентный подход:**
- Быстрее (один запрос к модели)
- Меньше использование памяти
- Подходит для простых задач
- Использует единый промпт `FERTILIZER_EXTRACTION_PROMPT_TEMPLATE`

**Мультиагентный подход (`simple_4agents`):**
- Более точное извлечение данных за счет специализации
- Разделение ответственности между агентами
- Больше запросов к модели (4 запроса вместо 1)
- Больше использование памяти и времени
- Использует специализированные промпты для каждого этапа

**Информация о режиме сохраняется в метриках:**
- Поле `multi_agent_mode` в JSON метриках содержит используемый режим (или `null` для одноагентного)
- Поле `prompt_info` содержит информацию о промптах для мультиагентного режима

### Режим вывода (verbose)

Проект поддерживает два режима вывода информации в консоль:

**Подробный вывод (`verbose=True`):**
- Используется при запуске через `main.py`
- Выводит исходный текст для анализа
- Выводит полный ответ модели на каждом этапе
- Выводит детальную статистику каждые 10 текстов
- Полезен для отладки и детального анализа работы модели

**Короткий вывод (`verbose=False`):**
- Используется при запуске через `run_all_models.py`
- Выводит только счетчик текущей итерации
- Выводит краткую статистику (валидные/невалидные JSON)
- Выводит ошибки (полностью для API моделей, обрезанные для локальных)
- Полезен для массовой оценки моделей без перегрузки консоли

**Примеры:**

```bash
# Подробный вывод (verbose=True) - при запуске через main.py
python main.py qwen-2.5-3b

# Короткий вывод (verbose=False) - при запуске через run_all_models.py
python run_all_models.py
```

### Оценка всех моделей

Запустите скрипт для оценки всех настроенных моделей:

```bash
python run_all_models.py
```

Скрипт автоматически:
- Проверит доступность Gemini API
- Запустит оценку для каждой модели
- Пропустит модели, которые не удалось загрузить
- Сохранит результаты для каждой модели

### Ручная оценка результатов

Если нужно пересчитать метрики качества из сохраненного CSV файла без повторного запуска модели:

```bash
python reevaluate.py results/results_model_name_timestamp.csv
```

Или с указанием имени модели:
```bash
python reevaluate.py results/results_model_name_timestamp.csv "model/name"
```

Для включения анализа ошибок через Gemini API добавьте флаг `--gemini`:
```bash
python reevaluate.py results/results_model_name_timestamp.csv "model/name" --gemini
```

**Примечание:** Для использования анализа через Gemini необходим `GEMINI_API_KEY` в `config_secrets.py` или в переменных окружения.

## Доступные модели

Проект поддерживает следующие модели:

### Локальные модели (загружаются с Hugging Face)

- **google/gemma-2-2b-it** - Gemma 2.2B Instruct (ключ: `gemma-2-2b`)
- **google/gemma-3-1b-it** - Gemma 3 1B Instruct (ключ: `gemma-3-1b`)
- **google/gemma-3-4b-it** - Gemma 3 4B Instruct (ключ: `gemma-3-4b`)
- **Qwen/Qwen2.5-1.5B-Instruct** - Qwen 2.5 1.5B (ключ: `qwen-2.5-1.5b`)
- **Qwen/Qwen2.5-3B-Instruct** - Qwen 2.5 3B (ключ: `qwen-2.5-3b`)
- **Qwen/Qwen2.5-4B-Instruct** - Qwen 2.5 4B (ключ: `qwen-2.5-4b`)
- **Qwen/Qwen3-8B** - Qwen 3 8B (ключ: `qwen-3-8b`) - поддерживает thinking mode
- **mistralai/Ministral-3-3B-Reasoning-2512** - Ministral 3 3B Reasoning (ключ: `Ministral-3-3B-Reasoning-2512`)
- **AI4Chem/CHEMLLM-2b-1_5** - CHEMLLM 2B (ключ: `CHEMLLM-2b-1_5`)
- **microsoft/Phi-3.5-mini-instruct** - Phi 3.5 Mini (ключ: `Phi-3.5-mini-instruct`)
- **microsoft/Phi-4-mini-instruct** - Phi 4 Mini (ключ: `phi-4-mini-instruct`)
- **unsloth/mistral-7b-v0.3-bnb-4bit** - Mistral 7B (предквантизированная) (ключ: `mistral-7b-v0.3-bnb-4bit`)

### API модели (через Google Generative AI API)

**Требования:**
- Для моделей Gemma 3 требуется `GEMINI_API_KEY` в `config_secrets.py` или переменных окружения
- Для моделей через OpenRouter требуется `OPENAI_API_KEY` в `config_secrets.py` или переменных окружения
- Библиотека `google-genai` должна быть установлена для моделей Gemma 3
- Библиотека `openai` должна быть установлена для моделей через OpenRouter
- Модели работают через API, не требуют локальной GPU
- Автоматическая обработка rate limits с повторными попытками (до 10 попыток)

**Доступные API модели:**

*Через Google Generative AI API:*
- **gemma-3-4b-it** (ключ: `gemma-3-4b-api`) - Gemma 3 4B через API
- **gemma-3-12b-it** (ключ: `gemma-3-12b-api`) - Gemma 3 12B через API
- **gemma-3-27b-it** (ключ: `gemma-3-27b-api`) - Gemma 3 27B через API

*Через OpenRouter API:*
- **deepseek-r1t-chimera** (ключ: `deepseek-r1t-chimera-api`) - DeepSeek R1T Chimera через OpenRouter API (free tier)
- **mistral-small-3.1-24b-instruct** (ключ: `mistral-small-3.1-24b-api`) - Mistral Small 3.1 24B Instruct через OpenRouter API (free tier)

**Особенности API моделей:**
- Не используют локальную GPU память
- В логах вместо информации о GPU выводится "API"
- Поддерживают автоматическую обработку ошибок rate limit (429)
- Извлекают время ожидания из сообщений об ошибках (например, "Please retry in 12.12324s")
- Все ошибки и ответы выводятся полностью в консоль (без обрезки)

**Пример запуска API модели:**
```bash
# Модели через Google Generative AI API
python main.py gemma-3-4b-api
python main.py gemma-3-12b-api --multi-agent simple_4agents

# Модели через OpenRouter API
python main.py deepseek-r1t-chimera-api
python main.py deepseek-r1t-chimera-api --multi-agent simple_4agents
python main.py mistral-small-3.1-24b-api
python main.py mistral-small-3.1-24b-api --multi-agent simple_4agents
```

## Метрики оценки

Для каждой модели автоматически собираются следующие метрики:

### Производительность
- Средняя скорость ответа (секунды)
- Время загрузки модели
- Общее время инференса
- Использование GPU памяти (среднее, максимальное, минимальное)

### Качество парсинга
- Процент успешно распарсенных JSON ответов
- Список ошибок парсинга с полными ответами моделей

### Качество извлечения данных
Для групп "массовая доля" и "прочее":
- Accuracy (точность)
- Precision (прецизионность)
- Recall (полнота)
- F1-score
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)

## Формат результатов

Результаты сохраняются в директории `results/`:

1. **results_model_name_timestamp.csv** - Детальные результаты для каждого текста:
   - `text` - исходный текст
   - `json` - извлеченный JSON из ответа модели
   - `json_parsed` - распарсенный JSON
   - `is_valid` - валидность JSON

2. **metrics_model_name_timestamp.json** - Сводные метрики:
   - Информация о GPU
   - Метрики производительности
   - Метрики качества
   - Гиперпараметры модели
   - Полный текст промпта
   - Список ошибок парсинга

3. **evaluation_summary.jsonl** - Общий файл со всеми прогонами (JSON Lines формат)

4. **gemini_analysis_model_name_timestamp.json** - Анализ ошибок от Gemini API (если включен)

## Конфигурация

Основные настройки находятся в `config.py`:

### Путь к датасету

Путь к датасету (`DATASET_PATH`) определяется автоматически. Программа ищет файл `results_var3.xlsx` в следующих местах (в порядке приоритета):

1. **Переменная окружения** (наивысший приоритет):
   ```bash
   export DATASET_PATH=/path/to/data/results_var3.xlsx
   ```

2. **Автоматический поиск** (если переменная окружения не установлена):
   - В родительской директории SmallLLMEvaluator: `../data/results_var3.xlsx`
   - На уровень выше: `../../data/results_var3.xlsx`
   - В директории запуска скрипта: `<script_dir>/data/results_var3.xlsx`
   - На уровень выше от директории запуска: `<script_dir>/../data/results_var3.xlsx`
   - В текущей рабочей директории: `<cwd>/data/results_var3.xlsx`
   - На уровень выше от текущей директории: `<cwd>/../data/results_var3.xlsx`
   - Относительный путь: `data/results_var3.xlsx`

**Примечание:** Если файл не найден, программа выведет предупреждение со списком всех проверенных путей, что поможет определить правильное расположение файла.

### Другие настройки

- `GROUND_TRUTH_PATH` - путь к файлу с ground truth (опционально, по умолчанию используется колонка `json_parsed` из датасета)
- `OUTPUT_DIR` - директория для сохранения результатов (по умолчанию `results/`)

### API ключи

**API ключи должны быть установлены через переменные окружения или файл `config_secrets.py`:**

- `HF_TOKEN` - токен Hugging Face (ОБЯЗАТЕЛЬНЫЙ)
- `GEMINI_API_KEY` - ключ Gemini API (опциональный, для анализа ошибок и моделей Gemma 3 через API)
- `OPENAI_API_KEY` - ключ OpenRouter API (опциональный, для моделей через OpenRouter API)

**Приоритет загрузки ключей:**
1. Файл `config_secrets.py` (если существует)
2. Переменные окружения

## Добавление новой модели

1. Добавьте функцию загрузки в `model_loaders.py`:
```python
def load_your_model() -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained("model/name", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "model/name",
        device_map="auto",
        dtype=torch.bfloat16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    return model, tokenizer
```

2. Добавьте функцию генерации (если нужна специализированная):
```python
def generate_your_model(model, tokenizer, prompt: str, max_new_tokens: int = 1024, repetition_penalty: float = None) -> str:
    # Ваша логика генерации
    pass
```

3. Добавьте конфигурацию в `main.py` в словарь `MODEL_CONFIGS`:
```python
"your-model-key": {
    "name": "model/name",
    "load_func": ml.load_your_model,
    "generate_func": ml.generate_standard,  # или ml.generate_your_model
    "hyperparameters": {
        "max_new_tokens": 1024,
        "do_sample": False,
        "dtype": "bfloat16"
    }
}
```

Для запуска в мультиагентном режиме используйте флаг `--multi-agent`:
```bash
python main.py your-model-key --multi-agent simple_4agents
```

**Примечание:** 
- Мультиагентный подход автоматически использует `StandardGenerator` из `core.generators`, который работает с большинством моделей
- Режим работы задается через аргумент командной строки `--multi-agent`
- Информация о режиме сохраняется в метриках (поле `multi_agent_mode`)

## Требования

### Основные требования

- Python 3.8+
- PyTorch с поддержкой CUDA (для локальных моделей)
- Transformers библиотека
- LangGraph (для мультиагентного подхода)
- Другие зависимости из requirements.txt

### Требования для локальных моделей

- CUDA-совместимая GPU (рекомендуется 8GB+ VRAM)
- Hugging Face Token (`HF_TOKEN`)

### Требования для API моделей

- Google Generative AI API Key (`GEMINI_API_KEY`)
- Библиотека `google-genai` (устанавливается через `pip install google-genai`)
- Локальная GPU не требуется

## Работа промптов

### Конфигурация промптов

Все промпты находятся в `prompt_config.py`:

- **`FERTILIZER_EXTRACTION_PROMPT_TEMPLATE`** - основной промпт для одноагентного подхода
- **`NUMERIC_FRAGMENTS_EXTRACTION_PROMPT`** - промпт для агента извлечения числовых фрагментов
- **`MASS_FRACTION_EXTRACTION_PROMPT`** - промпт для агента извлечения массовых долей
- **`OTHER_PARAMETERS_EXTRACTION_PROMPT`** - промпт для агента извлечения прочих параметров
- **`JSON_FORMATION_PROMPT`** - промпт для агента формирования JSON

### Поток данных в мультиагентном подходе

```
Исходный текст
    ↓
[Агент 1: Извлечение числовых фрагментов]
    ↓
Числовые фрагменты
    ↓
    ├─→ [Агент 2.1: Извлечение массовых долей] ──┐
    │                                             │
    └─→ [Агент 2.2: Извлечение прочих параметров] ──┤
                                                      ↓
                                            [Агент 3: Формирование JSON]
                                                      ↓
                                                 Финальный JSON
```

### Формат данных

Все агенты работают с текстовыми данными и извлекают структурированную информацию в формате JSON:

```json
{
  "массовая доля": [
    {"вещество": "N", "массовая доля": [26, 28]},
    {"вещество": "P2O5", "массовая доля": 20}
  ],
  "прочее": [
    {"параметр": "масса нетто единицы", "масса": 50, "единица": "кг"},
    {"параметр": "стандарт", "значение": "ТУ 2184-037-32496445-02"}
  ]
}
```

## Все варианты запуска

### 1. Оценка одной модели (одноагентный режим)

```bash
python main.py <model_key>
```

**Примеры:**
```bash
# Локальные модели
python main.py qwen-2.5-3b
python main.py gemma-2-2b
python main.py Phi-3.5-mini-instruct

# API модели через Google Generative AI (требуется GEMINI_API_KEY)
python main.py gemma-3-4b-api
python main.py gemma-3-12b-api
python main.py gemma-3-27b-api

# API модели через OpenRouter (требуется OPENAI_API_KEY)
python main.py deepseek-r1t-chimera-api
python main.py mistral-small-3.1-24b-api
```

**Примечание:** При запуске через `main.py` используется подробный вывод (verbose=True), который показывает исходный текст и ответ модели на каждом этапе. Это полезно для отладки и детального анализа работы модели.

**Список доступных моделей:**
```bash
python main.py
```
(запуск без аргументов покажет список всех доступных моделей)

### 2. Оценка одной модели (мультиагентный режим)

```bash
python main.py <model_key> --multi-agent <mode>
```

**Доступные режимы:**
- `simple_4agents` - 4 агента: извлечение числовых фрагментов, массовые доли, прочие параметры, формирование JSON
- `critic_3agents` - 3 агента: генератор (создает первоначальный ответ), критик (анализирует ответ на соответствие промпту), исправитель (устраняет найденные ошибки) (вдохновлено подходом из Reddit: 3agents)

**Примеры:**
```bash
# Локальные модели
python main.py qwen-2.5-3b --multi-agent simple_4agents
python main.py qwen-2.5-3b --multi-agent critic_3agents
python main.py gemma-2-2b --multi-agent simple_4agents
python main.py Phi-3.5-mini-instruct --multi-agent simple_4agents

# API модели через Google Generative AI
python main.py gemma-3-4b-api --multi-agent simple_4agents
python main.py gemma-3-12b-api --multi-agent simple_4agents
python main.py gemma-3-27b-api --multi-agent simple_4agents

# API модели через OpenRouter
python main.py deepseek-r1t-chimera-api --multi-agent simple_4agents
python main.py deepseek-r1t-chimera-api --multi-agent critic_3agents
python main.py mistral-small-3.1-24b-api --multi-agent simple_4agents
python main.py mistral-small-3.1-24b-api --multi-agent critic_3agents
```

### 3. Оценка всех моделей подряд

```bash
python run_all_models.py
```

Скрипт автоматически:
- Проверит доступность Gemini API
- Запустит оценку для каждой модели из конфигурации
- Пропустит модели, которые не удалось загрузить
- Сохранит результаты для каждой модели
- Выведет итоговую сводку

**Примечания:**
- Все модели будут запущены в одноагентном режиме. Для мультиагентного режима используйте `main.py` с флагом `--multi-agent`.
- При запуске через `run_all_models.py` используется короткий вывод (verbose=False), который показывает только счетчик итераций, ошибки и краткую статистику.

### 4. Переоценка результатов из сохраненного CSV

```bash
python reevaluate.py <path_to_csv> [model_name] [--gemini]
```

**Примеры:**
```bash
# Переоценка с автоматическим определением модели
python reevaluate.py results/results_qwen-2.5-3b_20240115_103000.csv

# Переоценка с указанием модели
python reevaluate.py results/results_qwen-2.5-3b_20240115_103000.csv "Qwen/Qwen2.5-3B-Instruct"

# Переоценка с анализом через Gemini API
python reevaluate.py results/results_qwen-2.5-3b_20240115_103000.csv "Qwen/Qwen2.5-3B-Instruct" --gemini
```

**Примечания:**
- Переоценка позволяет пересчитать метрики качества без повторного запуска модели
- Флаг `--gemini` включает анализ ошибок через Gemini API с рекомендациями по улучшению
- Для использования `--gemini` необходим `GEMINI_API_KEY` в `config_secrets.py`
- Анализ от Gemini сохраняется в отдельный JSON файл: `gemini_analysis_{model_name}_{timestamp}_reevaluated.json`

### 5. Справка по использованию

```bash
# Показать справку по main.py
python main.py

# Показать справку по reevaluate.py
python reevaluate.py
```

### Сводная таблица команд

| Команда | Описание | Режим | Gemini анализ |
|---------|----------|-------|---------------|
| `python main.py <model_key>` | Оценка одной модели | Одноагентный | По умолчанию |
| `python main.py <model_key> --multi-agent simple_4agents` | Оценка одной модели | Мультиагентный | По умолчанию |
| `python run_all_models.py` | Оценка всех моделей | Одноагентный | По умолчанию |
| `python reevaluate.py <csv_file>` | Переоценка результатов | - | Отключен |
| `python reevaluate.py <csv_file> [model_name] --gemini` | Переоценка с анализом Gemini | - | Включен |
