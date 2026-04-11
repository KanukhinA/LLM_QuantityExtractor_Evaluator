# Evaluation of small LLM - Обработка данных по удобрениям

Проект для автоматизированной оценки различных LLM моделей на датасете по обработке данных об удобрениях.

**Структура документа:** установка (разд. 2) и структура проекта (разд. 3) → режимы работы: одноагентный/мультиагентный, structured output, outlines (разд. 4) → как запускать и примеры команд (разд. 5) → модели (6), метрики (7), промпты (8), добавление модели (9), требования (10), справка (11).

## 1. Описание

Проект позволяет:
- Загружать и оценивать различные LLM модели из Hugging Face
- Автоматически измерять метрики производительности (скорость, использование памяти)
- Оценивать качество извлечения данных (массовая доля, прочее)
- Анализировать ошибки парсинга JSON
- Получать рекомендации от Gemini API по улучшению моделей
- Переоценивать результаты из сохраненных файлов без повторного запуска модели
- Использовать одноагентный или мультиагентный подход для извлечения данных (на основе LangGraph)

## 2. Установка и настройка

### 2.1. Установка зависимостей

1. Клонируйте репозиторий или скопируйте проект

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

### 2.2. Настройка API ключей

Создайте файл `config_secrets.py` на основе примера:
```bash
cp config_secrets.py.example config_secrets.py
```

Затем откройте `config_secrets.py` и заполните свои ключи:
```python
HF_TOKEN = "your_huggingface_token_here"
GEMINI_API_KEY = "your_gemini_api_key_here"  # опционально
OPENAI_API_KEY = "your_openai_api_key_here"  # опционально
GOOGLE_SHEETS_SPREADSHEET_ID = ""  # опционально, для экспорта метрик в Google Таблицу
```

**Важно:**
- `HF_TOKEN` - ОБЯЗАТЕЛЬНЫЙ. Без него программа не запустится (нужен для загрузки моделей с Hugging Face).
- `GEMINI_API_KEY` - опциональный, но рекомендуется для:
  - Анализа ошибок через Gemini API (автоматические рекомендации по улучшению)
  - Запуска API моделей Gemma 3 (gemma-3-4b-api, gemma-3-12b-api, gemma-3-27b-api)
  - Без ключа будет пропущен анализ ошибок, но локальные модели будут работать
- `OPENAI_API_KEY` - опциональный, требуется для:
  - Запуска моделей через OpenRouter API (deepseek-r1t-chimera-api, mistral-small-3.1-24b-api, qwen-3-32b-api)
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
  - Используется для запуска моделей через OpenRouter API (например, deepseek-r1t-chimera-api, mistral-small-3.1-24b-api, qwen-3-32b-api)

### 2.3. Конфигурация проекта

Основные настройки находятся в `config.py`:

**Путь к датасету:**
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

**Другие настройки:**
- `GROUND_TRUTH_PATH` - путь к файлу с ground truth (опционально, по умолчанию используется колонка `json_parsed` из датасета)
- `OUTPUT_DIR` - директория для сохранения результатов (по умолчанию `results/`)
- `PROMPT_TEMPLATE_NAME` - название переменной промпта из `prompt_config.py` для одноагентного подхода (по умолчанию `"DETAILED_INSTR_ZEROSHOT_BASELINE"`):
  - `"DETAILED_INSTR_ZEROSHOT_BASELINE"` - детальный zero-shot промпт без примера (baseline)
  - `"DETAILED_INSTR_ONESHOT"` - детальный промпт с примером текста и ответа (One-shot prompt)
  - `"DETAILED_INSTR_ZEROSHOT_CD"` - zero-shot с примером JSON на латинице (режим CD)
  - `"DETAILED_INSTR_ONESHOT_CD"` - one-shot с примером JSON на латинице (режим CD)
  - `"DETAILED_INSTR_ZEROSHOT_CD_RUS"` - zero-shot с примером JSON на кириллице (CD + RUS схема)
  - `"DETAILED_INSTR_ONESHOT_CD_RUS"` - one-shot с примером JSON на кириллице (CD + RUS схема)
  - `"MINIMAL_INSTR_FIVESHOT"` - минимальный инструктивный few-shot промпт с 5 примерами
  - `"MINIMAL_INSTR_FIVESHOT_APIE"` - few-shot промпт с 5 примерами (APIE): примеры подставляются из XLSX (приоритет) или CSV в `OUTPUT_DIR` (`few_shot_examples_<model_key>_*.xlsx` или `*.csv`). Рекомендуется разметка через `label_few_shot_with_gemini.py` с сохранением в XLSX. **Запуск оценки возможен только при наличии такого файла**; иначе в консоль выводится причина и подсказка

  **Настройка в `config.py`:**
  ```python
  PROMPT_TEMPLATE_NAME = "DETAILED_INSTR_ZEROSHOT_BASELINE"  # или любая другая переменная из prompt_config.py
  ```

  **Переопределение из консоли или models.yaml:**
  - Флаг `--prompt NAME` в main.py и run_all_models.py переопределяет промпт для текущего запуска
  - В `models.yaml` можно указать `hyperparameters.prompt_template_name` для конкретной модели

- **Flash Attention 2 для локальных моделей:** по умолчанию включено (`USE_FLASH_ATTENTION_2=1`). При установленном пакете `flash-attn` все поддерживаемые локальные модели загружаются с `attn_implementation="flash_attention_2"` (экономия VRAM, ускорение). Если `flash-attn` не установлен, выводится предупреждение и модели загружаются без Flash Attention 2. Отключить: `set USE_FLASH_ATTENTION_2=0` (Windows) / `export USE_FLASH_ATTENTION_2=0` (Linux). Установка flash-attn: `pip install flash-attn --no-build-isolation` (требуется CUDA; на Windows сборка часто недоступна, рекомендуется Linux/WSL).

**Приоритет загрузки API ключей:**
1. Файл `config_secrets.py` (если существует)
2. Переменные окружения

### 2.4. Расчёт max_new_tokens по датасету

Чтобы подобрать разумное значение `max_new_tokens` для моделей (в `models.yaml` или при вызове), можно оценить длину ответов в токенах по ground truth из тестового датасета. Скрипт `calc_max_new_tokens.py`:

- загружает датасет (по умолчанию через `find_dataset_path()`, колонка `json_parsed` или `json`);
- сериализует каждый ответ в JSON-строку в том же формате, что и вывод модели (`indent=2`);
- **учитывает разметку JSON:** при парсинге датасета символы `\n` и `\` в сыром выводе модели превращаются в один символ; перед подсчётом токенов строка «разворачивается» (каждый перевод строки — как `\n`, каждый обратный слэш — как `\\`), чтобы оценка соответствовала длине реального вывода модели;
- считает длину в токенах через указанный токенизатор Hugging Face;
- выводит max, mean, перцентили (p90, p95, p99) и рекомендуемое значение (max + запас, с округлением).

Запуск из корня проекта:

```bash
# Один токенизатор (по умолчанию Qwen/Qwen2.5-3B-Instruct), статистика по всем ответам
python calc_max_new_tokens.py

# Прогнать самый длинный ответ через токенизаторы всех локальных моделей из models.yaml
python calc_max_new_tokens.py --all-models
```

Опции: `--dataset PATH` — путь к датасету (Excel или CSV); `--tokenizer NAME` — модель токенизатора (по умолчанию `Qwen/Qwen2.5-3B-Instruct`); `--all-models` — один самый длинный ответ через все локальные токенизаторы (API и Ollama пропускаются); `--margin 0.15` — запас к max (15%); `--min 256` — минимальное рекомендуемое значение; `--round 64` — округлять до кратного 64. Полученное значение можно прописать в `models.yaml` в `hyperparameters.max_new_tokens` (или `max_length` при необходимости) для нужных моделей.

### 2.5. Установка и запуск vLLM

`vLLM` используется как отдельный HTTP-сервер инференса. Проект подключается к нему через OpenAI-совместимый endpoint `/v1/chat/completions`.

Для `main.py` (ключ `*-vllm` или флаг `--vllm`) и для `run_all_models.py --vllm` сервер поднимается автоматически: перед оценкой для каждой `*-vllm` модели запускается `vllm serve`, при смене модели или в конце работы предыдущий автосервер завершается. Логика вынесена в модуль `vllm_autoserver.py`. Вывод `vllm serve` пишется в `OUTPUT_DIR/vllm_autoserver.log` (по умолчанию `results/vllm_autoserver.log`), чтобы при ошибке или таймауте было видно OOM, неверный id модели и т.д.

**Таймаут готовности:** крупные модели или первый запуск (скачивание весов) могут занимать много времени. По умолчанию скрипт ждёт ответа `GET /v1/models` до **600** секунд. Изменить можно переменной окружения `VLLM_READY_TIMEOUT_SEC` (секунды), например: `set VLLM_READY_TIMEOUT_SEC=1200` (Windows) или `export VLLM_READY_TIMEOUT_SEC=1200` (Linux/macOS).

1) Установите vLLM (лучше в отдельном окружении):

```bash
pip install vllm
```

2) Запустите сервер (пример):

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct
```

Пример с квантизацией:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --quantization awq
```

3) При необходимости задайте адрес сервера:
- Windows: `set VLLM_BASE_URL=http://127.0.0.1:8000`
- Linux/Mac: `export VLLM_BASE_URL=http://127.0.0.1:8000`

4) Проверка, что нужные `*-vllm` модели доступны на сервере (удобно, если вы подняли `vllm serve` вручную или проверяете уже запущенный автосервер):

```bash
python check_vllm_models.py
```

Важно: в `models.yaml` поле `vllm_name` для модели должно совпадать с id, поднятым в `vllm serve`.

## 3. Структура проекта

```
SmallLLMEvaluator/
├── main.py                 # Основной скрипт для запуска оценки одной модели
├── run_all_models.py       # Скрипт для запуска оценки всех моделей
├── vllm_autoserver.py      # Автозапуск/остановка vllm serve для main.py и run_all_models.py
├── reevaluate.py           # Скрипт для переоценки результатов из файла
├── calc_max_new_tokens.py  # Расчёт max_new_tokens по длине ground truth в токенах
├── model_evaluator.py      # Класс для оценки моделей
├── model_loaders.py        # Функции загрузки локальных моделей
├── model_loaders_api.py    # Функции загрузки и генерации для API моделей
├── model_config_loader.py  # Загрузчик конфигураций моделей из YAML
├── models.yaml             # Конфигурация всех моделей (YAML)
├── metrics.py              # Функции расчета метрик качества
├── metrics_printer.py      # Класс для вывода метрик в консоль
├── file_manager.py         # Класс для работы с файлами
├── utils.py                # Утилиты для парсинга JSON и построения промптов
├── gemini_analyzer.py      # Интеграция с Gemini API для анализа ошибок
├── gpu_info.py             # Функции для получения информации о GPU
├── config.py               # Конфигурация проекта
├── config_loader.py        # Загрузка API ключей (config_secrets.py или переменные окружения)
├── prompt_config.py        # Конфигурация промптов для агентов
├── multi_agent_graph.py    # Мультиагентная система на LangGraph
├── few_shot_extractor.py   # Модуль для извлечения few-shot примеров на основе Dual-Level Introspective Uncertainty
├── structured_schemas.py   # Pydantic схемы для structured output
├── outlines_schema.py      # JSON-схемы для outlines/guidance (латиница и RUS)
├── workflow_config.py      # Конфигурация мультиагентных workflow
├── requirements.txt        # Зависимости проекта
├── core/                   # ООП архитектура для генераторов
│   ├── __init__.py
│   ├── base.py             # Базовый абстрактный класс BaseGenerator
│   └── generators.py       # Реализация StandardGenerator
└── README.md               # Документация
```

## 4. Режимы работы

В этом разделе собраны все режимы и опции генерации; как именно запускать команды — в разделе 5.

### 4.1. Одноагентный и мультиагентный подход

**Одноагентный (по умолчанию):** один запрос к модели с полным промптом; модель возвращает JSON за один проход. Быстрее, меньше памяти. Промпт задаётся в `config.PROMPT_TEMPLATE_NAME` или через `--prompt`.

**Мультиагентный (LangGraph):** задача разбита на несколько агентов. Используется модуль `core/` (`BaseGenerator`, `StandardGenerator`) и `multi_agent_graph.py`. Режимы:
- **`simple_4agents`** — 4 агента: числовые фрагменты → массовые доли и прочие параметры → формирование JSON.
- **`critic_3agents`** — 3 агента: генератор → критик (анализ ответа) → исправитель.
- **`qa_workflow`** — 6 агентов: питательные вещества → массовые доли по каждому → стандарт → марка → количества → сборка JSON.
- **`validation_fix_2agents`** — 2 агента: генерация ответа (как в одноагентном) → валидация Pydantic; при невалидном JSON ошибки валидации подаются в LLM для исправления (исходный промпт и текст передаются исправителю).

Потоки данных по режимам и список промптов по каждому режиму — в разделе 8.

### 4.2. Structured Output, извлечение и валидация JSON

**Режим `--structured-output`** (отдельный от одно/мультиагентного):
- Промпт из `config.PROMPT_TEMPLATE_NAME` (или `--prompt`); для локальных моделей в конец промпта добавляется JSON Schema из Pydantic.
- Для API Gemini (gemma-3-*-api) схема передаётся в API (`response_schema`), ответ приходит валидным JSON по схеме.

**Извлечение кандидата на JSON** (`extract_json_from_response`). Ответ модели может содержать пояснения, markdown, несколько блоков. Выбирается один фрагмент-кандидат в таком порядке: (1) последний маркер «Ответ:» / «Answer:» — берётся текст после него; внутри него — последний блок в markdown-ограждении (тройные обратные кавычки, метка json) или при его отсутствии подстрока от первого `{` или `[` до конца; (2) если маркера нет — последний такой блок по всему тексту; (3) если и его нет — подстрока от первого `{` или `[` до конца; (4) иначе в парсинг уходит весь ответ. Так снижается влияние пояснений до и после JSON.

**Устойчивый парсинг** (`parse_json_safe`). По строке-кандидату выполняется цепочка починок: выделение JSON-подобного фрагмента (блок в ограждении или от первой скобки до конца); нормализация текста (типографские кавычки → обычные, `None` → `null`, пробельные символы); вставка пропущенных запятых между объектами и удаление лишних перед `}` и `]`; дозакрытие скобок с учётом кавычек (подсчёт глубины вне строк, дописывание недостающих `]` и `}`, закрытие оборванных строк и незавершённых значений); удаление двойных запятых. Затем попытка разбора через `json.loads`; при ошибке — безопасный разбор литералов (`ast.literal_eval` с заменой `null` на `None`). Парсер таким образом справляется с обрезанным выводом, лишними или пропущенными запятыми и частично незакрытыми скобками.

**Валидация по схеме** (`validate_with_pydantic`). Выполняется в двух точках: по сырому тексту ответа (сначала из него извлекается и парсится JSON, затем объект проверяется схемой) и по уже распарсенному объекту (после приведения латинских ключей к кириллице и очистки полей). В обоих случаях проверяется соответствие схеме: наличие обязательных полей, типы, допустимые ключи. При успехе получается нормализованное представление (в т.ч. алиасы ключей); при неудаче — список ошибок для метрик и для `pydantic_errors`. Флаг успешности парсинга JSON и флаги прохождения Pydantic-валидации учитываются отдельно. Для локальных моделей цепочка: ответ → извлечение фрагмента → парсинг с починками → при необходимости приведение ключей к кириллице и очистка → валидация по сырому тексту и по распарсенному объекту. Для API (Gemini с response_schema) ответ обычно уже валидный JSON по схеме; он парсится и при необходимости проверяется схемой для нормализации.

### 4.3. Outlines и Guidance (только локальные модели)

Работают только вместе с `--structured-output`. Взаимоисключающие флаги:
- **`--outlines`** — схема из `outlines_schema.py` (латиница или RUS по промпту).
- **`--pydantic-outlines`** — схема из Pydantic `model_json_schema()`.
- **`--guidance`** — constrained decoding через llguidance (по умолчанию схема RUS, промпт с кириллическим JSON).

Для всех режимов используется одна схема `FertilizerExtractionOutput`; при латинской схеме результат конвертируется в кириллицу. Рекомендуемые промпты: `*_CD` или `--prompt DETAILED_INSTR_ZEROSHOT_CD`.

### 4.4. Сравнение режимов

| Режим | Запросов к модели | Память/скорость | Примечание |
|-------|-------------------|-----------------|------------|
| Одноагентный | 1 | Меньше | Промпт из config / --prompt |
| simple_4agents | 4 | Больше | Специализированные промпты по агентам |
| critic_3agents | 3 | Больше | Генератор → критик → исправитель |
| qa_workflow | 4+ | Больше | Зависит от числа веществ |
| validation_fix_2agents | 1 или 2 | Как одноагентный / +1 при ошибке | Генерация → валидация → при ошибке повторная подача ошибок в LLM |
| Structured output | как выше | — | Добавляет схему в промпт / в API |
| Outlines / Guidance | как одноагентный | — | Constrained decoding, только локальные |

## 5. Использование

### 5.1. Запуск оценки

Запустите `main.py` с указанием модели:

```bash
python main.py <model_key>
```

Например:
```bash
python main.py qwen-2.5-3b
```

**Запуск нескольких моделей:**

Вы можете указать несколько моделей для последовательной оценки. Модели можно указать через запятую или через пробел:

```bash
# Через запятую
python main.py qwen-2.5-3b,qwen-2.5-4b,gemma-3-4b

# Через пробел
python main.py qwen-2.5-3b qwen-2.5-4b gemma-3-4b

# С флагами (применяются ко всем моделям)
python main.py qwen-2.5-3b qwen-2.5-4b --no-gemini
python main.py gemma-3-4b gemma-3-12b-api --structured-output
```

**Примечания:**
- Все указанные модели будут оценены последовательно
- Флаги (например, `--multi-agent`, `--structured-output`, `--no-gemini`) применяются ко всем моделям
- Для каждой модели выводится отдельная сводка результатов
- В конце выводится итоговая сводка по всем моделям

**Мультиагентный режим:** `python main.py <model_key> --multi-agent <mode>`. Режимы: `simple_4agents`, `critic_3agents`, `qa_workflow`, `validation_fix_2agents` (описание — раздел 4.1).

**Основные флаги:** `--multi-agent <mode>`, `--structured-output`, `--outlines`, `--pydantic-outlines`, `--guidance`, `--prompt NAME`, `--no-gemini` (отключить анализ ошибок через Gemini API). Подробности режимов — раздел 4.

**Примеры запуска:**

```bash
# Одноагентный режим (по умолчанию)
python main.py qwen-2.5-3b

# Несколько моделей (через пробел)
python main.py qwen-2.5-3b qwen-2.5-4b gemma-3-4b

# Несколько моделей (через запятую)
python main.py qwen-2.5-3b,qwen-2.5-4b,gemma-3-4b

# Несколько моделей с флагами
python main.py qwen-2.5-3b qwen-2.5-4b --no-gemini
python main.py gemma-3-4b gemma-3-12b-api --structured-output

# Мультиагентный режим
python main.py qwen-2.5-3b --multi-agent simple_4agents
python main.py qwen-2.5-3b --multi-agent critic_3agents
python main.py qwen-2.5-3b --multi-agent qa_workflow
python main.py qwen-2.5-3b --multi-agent validation_fix_2agents

# Qwen3 32B (локальная версия, требует ~64GB+ VRAM)
python main.py qwen-3-32b
python main.py qwen-3-32b --multi-agent simple_4agents

# CodeGemma 7B (специализирована для работы с кодом, требует ~14GB VRAM)
python main.py codegemma-7b
python main.py codegemma-7b --multi-agent simple_4agents

# Запуск без анализа через Gemini API
python main.py qwen-2.5-3b --no-gemini
python main.py qwen-2.5-3b --multi-agent simple_4agents --no-gemini

# Одноагентный режим с structured output
python main.py qwen-2.5-3b --structured-output

# Structured output + outlines (локальные модели)
python main.py qwen-2.5-3b --structured-output --outlines

# Structured output + схема из Pydantic (вместо --outlines)
python main.py qwen-2.5-3b --structured-output --pydantic-outlines

# Structured output + llguidance (по умолчанию схема RUS)
python main.py qwen-2.5-3b --guidance

# Запуск через vLLM (используется ключ <model_key>-vllm)
python main.py qwen-2.5-3b --vllm

# Указание промпта из консоли (переопределяет config.PROMPT_TEMPLATE_NAME)
python main.py qwen-2.5-3b --prompt DETAILED_INSTR_ZEROSHOT_CD --structured-output --outlines

# Мультиагентный режим с structured output
python main.py gemma-3-4b-api --multi-agent simple_4agents --structured-output

# API модель с structured output
python main.py gemma-3-27b-api --structured-output
```

Доступные модели можно посмотреть, запустив `main.py` без аргументов.

### 5.2. Оценка всех моделей

Запустите скрипт для оценки всех настроенных моделей:

```bash
# Оценка всех моделей (локальных и API)
python run_all_models.py

# Оценка только локальных моделей (исключает API модели)
python run_all_models.py --local-only

# Оценка всех моделей в мультиагентном режиме
python run_all_models.py --multi-agent simple_4agents

# Оценка всех моделей с structured output
python run_all_models.py --structured-output

# Оценка всех локальных моделей с structured output и outlines
python run_all_models.py --local-only --structured-output --outlines

# Оценка только vLLM-версий моделей (*-vllm)
python run_all_models.py --vllm

# Оценка только Ollama-версий моделей (*-ollama)
python run_all_models.py --ollama

# Указание промпта из консоли
python run_all_models.py --local-only --prompt DETAILED_INSTR_ZEROSHOT_CD --structured-output --outlines

# Комбинация параметров
python run_all_models.py --local-only --multi-agent qa_workflow --structured-output
```

**Доступные параметры:**
- `--local-only` - запустить оценку только для локальных моделей (исключить API модели)
- `--prompt NAME` - название промпта из prompt_config.py (переопределяет config.PROMPT_TEMPLATE_NAME)
- `--multi-agent MODE` - режим мультиагентного подхода (simple_4agents, critic_3agents, qa_workflow, validation_fix_2agents)
- `--structured-output` - использовать structured output через Pydantic
- `--outlines` - использовать outlines со схемой из outlines_schema.py (только для локальных моделей; взаимоисключающий с --pydantic-outlines)
- `--pydantic-outlines` - использовать outlines со схемой из Pydantic model_json_schema() (взаимоисключающий с --outlines)
- `--guidance` - использовать llguidance для constrained decoding (по умолчанию схема RUS)
- `--vllm` - запускать только `*-vllm` версии моделей (инференс через сервер vLLM)
- `--ollama` - запускать только `*-ollama` версии моделей (инференс через Ollama)

Скрипт автоматически:
- Проверит доступность Gemini API
- Запустит оценку для каждой модели (или только локальных, если указан `--local-only`)
- Применит указанные режимы (multi-agent, structured-output, outlines) ко всем моделям
- Пропустит модели, которые не удалось загрузить
- Сохранит результаты для каждой модели
- Выведет статистику: общее количество моделей, количество локальных/API моделей (при использовании `--local-only`)

### 5.3. Переоценка результатов

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

### 5.4. Извлечение few-shot примеров

Модуль `few_shot_extractor.py` реализует алгоритм **Dual-Level Introspective Uncertainty** для активного обучения и выбора наиболее информативных примеров для аннотации. Сгенерированный CSV используется для разметки; промпт **MINIMAL_INSTR_FIVESHOT_APIE** (режим APIE) подставляет примеры из последнего по времени файла в `OUTPUT_DIR`: сначала ищется **XLSX** (`few_shot_examples_<model_key>_*.xlsx`), при отсутствии — **CSV** (`few_shot_examples_<model_key>_*.csv`). Рекомендуется размечать примеры через Gemini и сохранять в XLSX для ручной проверки (см. ниже). **Перед запуском оценки в режиме APIE необходим файл с примерами**; при отсутствии оценка не запускается.

**Алгоритм:**
1. Фильтрация неразмеченных текстов (удаление текстов, уже присутствующих в размеченном датасете)
2. Кластеризация оставшихся текстов с помощью KMeans и SentenceTransformer
3. Генерация k вариантов ответа для каждого кандидата с использованием sampling (do_sample=True, temperature=0.7, top_k=50)
4. Вычисление трех типов неопределенности:
   - **Generation Disagreement (𝒰_d)**: среднее попарное расстояние Левенштейна между k ответами
   - **Format-Level Uncertainty (𝒰_f)**: Parsing Failure Rate + Structural Disagreement
   - **Content-Level Uncertainty (𝒰_c)**: 1 - средняя Jaccard similarity между извлеченными данными
5. Вычисление общей неопределенности: 𝒰_total = α·𝒰_d + β·𝒰_f + γ·𝒰_c
6. Ранжирование образцов по 𝒰_total и выбор топ-n для аннотации

**Запуск из консоли:**

```bash
python few_shot_extractor.py <model_key> [опции]
```

**Примеры:**

```bash
# Базовый запуск с параметрами по умолчанию
python few_shot_extractor.py gemma-3-4b

# Указание количества примеров и кластеров
python few_shot_extractor.py qwen-2.5-3b --n-examples 20 --n-clusters 50

# Полная настройка всех параметров
python few_shot_extractor.py gemma-3-4b \
    --n-examples 15 \
    --n-clusters 100 \
    --k 3 \
    --temperature 0.7 \
    --top-k 50 \
    --alpha 0.33 \
    --beta 0.33 \
    --gamma 0.34 \
    --max-new-tokens 1024 \
    --verbose \
    --output results/few_shot_examples.csv

# Переопределение путей к корпусам
python few_shot_extractor.py gemma-3-4b \
    --unlabeled-corpus data/unlabeled_corpus.xlsx \
    --labeled-dataset data/labeled_dataset.xlsx
```

**Аргументы командной строки:**

Обязательные:
- `model_key` - ключ модели из `models.yaml` (например, `gemma-3-4b`, `qwen-2.5-3b`)

Опциональные:
- `--n-examples` - количество примеров для извлечения (по умолчанию: 10)
- `--n-clusters` - количество кластеров для кластеризации (по умолчанию: 100)
- `--k` - количество вариантов ответа для каждого текста (по умолчанию: 3)
- `--temperature` - температура для sampling (по умолчанию: 0.7)
- `--top-k` - top_k для sampling (по умолчанию: 50)
- `--alpha` - вес для Generation Disagreement (по умолчанию: 0.33)
- `--beta` - вес для Format Uncertainty (по умолчанию: 0.33)
- `--gamma` - вес для Content Uncertainty (по умолчанию: 0.34)
- `--text-column` - название колонки с текстами (по умолчанию: "text")
- `--max-new-tokens` - максимальное количество новых токенов (по умолчанию: 1024)
- `--unlabeled-corpus` - путь к неразмеченному корпусу (переопределяет значение из config.py)
- `--labeled-dataset` - путь к размеченному датасету (переопределяет значение из config.py)
- `--verbose` - подробный вывод
- `--output` - путь для сохранения результатов в CSV (если не указан, сохраняется в `results/few_shot_examples_<model_key>_<timestamp>.csv`). Файлы в `results/` автоматически подхватываются режимом **MINIMAL_INSTR_FIVESHOT_APIE** при оценке соответствующей модели.

**Настройка путей в config.py:**

Пути к корпусам можно указать в `config.py`:

```python
# Путь к неразмеченному корпусу (Excel файл)
UNLABELED_CORPUS_PATH = "data/unlabeled_corpus.xlsx"

# Путь к размеченному датасету (Excel файл)
# Если не указан, используется стандартный датасет (results_var3.xlsx)
LABELED_DATASET_PATH = "data/labeled_dataset.xlsx"
```

**Возвращаемое значение:**

CSV файл с колонками:
- `text`: исходный текст
- `json`: лучший распарсенный JSON-ответ по примеру (используется в MINIMAL_INSTR_FIVESHOT_APIE как «Ответ примера N»)
- `cluster`: номер кластера
- `generation_disagreement`: метрика Generation Disagreement
- `R_fail`: Parsing Failure Rate
- `structural_disagreement`: Structural Disagreement
- `format_uncertainty`: Format-Level Uncertainty
- `content_uncertainty`: Content-Level Uncertainty
- `total_uncertainty`: общая неопределенность (отсортировано по убыванию)

Колонка `responses` в файл не записывается (остаётся только в памяти при работе скрипта).

**Разметка примеров через Gemini API (XLSX):**

Если примеры неразмечены или нужна разметка для ручной проверки, используйте скрипт `label_few_shot_with_gemini.py`. Он читает CSV (или XLSX) от few_shot_extractor, для каждого текста запрашивает у Gemini API извлечение JSON по той же задаче и сохраняет результат в **XLSX** (удобно просматривать и править вручную). Для работы нужен `GEMINI_API_KEY`.

```bash
# Разметить все примеры из CSV и сохранить в results/few_shot_examples_<model_key>_gemini_<timestamp>.xlsx
python label_few_shot_with_gemini.py results/few_shot_examples_gemma-2-2b_20250108_120000.csv --model-key gemma-2-2b

# Указать выходной файл и ограничить число примеров
python label_few_shot_with_gemini.py results/few_shot_examples_gemma-2-2b_20250108_120000.csv --output results/my_examples.xlsx --model-key gemma-2-2b --max-examples 5
```

Аргументы: `input` — путь к CSV или XLSX с колонкой `text`; `--output` / `-o` — путь к выходному XLSX; `--model-key` — ключ модели для имени файла; `--max-examples` — максимум строк для разметки; `--gemini-model` — модель Gemini (по умолчанию `gemini-2.5-flash`). После сохранения XLSX режим APIE при оценке этой модели подхватит новый файл (приоритет у XLSX перед CSV).

### 5.5. Режим вывода (verbose)

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

### 5.6. Интеграция с Google Таблицами

Модуль `google_sheets_integration.py` обходит папку результатов, для каждой пары (модель, метод) выбирает **последний** запуск (по timestamp или дате файла), формирует таблицу F1 и при наличии настроек загружает её в Google Таблицу.

**Возможности:**
- Обход `results/` (или `config.OUTPUT_DIR`), поиск всех `metrics_*.json`
- Для каждой пары (model_key, method) используется один последний запуск
- Таблица: модели по вертикали, методы (папки) по горизонтали, значения — F1
- Загрузка в Google Таблицу (два листа по умолчанию: «массовая доля» и «прочее») или экспорт в CSV

**Структура таблицы:**
- **По вертикали**: alias моделей (model_key)
- **По горизонтали**: методы (название папки = prompt_template или multi_agent_mode)
- **Значения**: F1 метрики для выбранной группы ("массовая доля" или "прочее")

**Настройка Google Sheets API (где взять JSON):**

1. **Откройте Google Cloud Console:** https://console.cloud.google.com/
2. **Создайте проект** (если ещё нет): сверху нажмите выпадающий список с названием проекта → «Создать проект» → введите имя (например `SmallLLM`) → «Создать».
3. **Включите API:** слева меню «API и сервисы» → «Библиотека» → в поиске введите **Google Sheets API** → откройте → нажмите **«Включить»**. То же для **Google Drive API** (поиск → открыть → «Включить»).
4. **Создайте Service Account и скачайте JSON:**
   - Слева «API и сервисы» → **«Учётные данные»** (Credentials).
   - Вверху вкладка **«Учётные данные»** → кнопка **«+ Создать учётные данные»** → выберите **«Сервисный аккаунт»** (Service account).
   - Имя сервисного аккаунта — любое (например `sheets-export`), нажмите **«Создать и продолжить»** → роль можно не менять → **«Готово»**.
   - Вернётесь на список учётных данных. В блоке **«Сервисные аккаунты»** найдите только что созданный аккаунт, кликните по нему.
   - Откроется страница аккаунта. Вверху вкладка **«Ключи»** (Keys) → **«Добавить ключ»** → **«Создать ключ»** → тип **JSON** → **«Создать»**. Файл сразу скачается в папку «Загрузки».
5. **Переименуйте скачанный файл** в `google_sheets_credentials.json` и положите его в **корень проекта** SmallLLMEvaluator (рядом с `google_sheets_integration.py`). Файл в .gitignore, в репозиторий не попадёт.
6. **Доступ к таблице:** откройте вашу Google Таблицу → «Настройки доступа» (Поделиться) → добавьте **email** вида `something@project-name.iam.gserviceaccount.com` (этот email лежит в JSON в поле `client_email`) с правом «Редактор».

7. **ID таблицы для запуска без аргументов:** в `config_secrets.py` задайте `GOOGLE_SHEETS_SPREADSHEET_ID = "ваш_id_из_url"`. Тогда `python google_sheets_integration.py` без аргументов загрузит данные в эту таблицу.

**Примеры использования:**

```bash
# Запуск без аргументов: обход results/, последний запуск по каждой паре (модель, метод), загрузка в таблицу
# (нужны google_sheets_credentials.json в корне проекта и GOOGLE_SHEETS_SPREADSHEET_ID в config_secrets.py)
python google_sheets_integration.py

# Только собрать и вывести метрики (не загружать в таблицу)
python google_sheets_integration.py --no-upload

# Экспорт в CSV
python google_sheets_integration.py --export-csv results_f1.csv --group "массовая доля"
python google_sheets_integration.py --export-csv results_f1_prochee.csv --group "прочее"

# Загрузка в таблицу с явным указанием credentials и spreadsheet-id
python google_sheets_integration.py --spreadsheet-id YOUR_SPREADSHEET_ID --group "массовая доля"
```

**Аргументы командной строки:**

Для загрузки в Google Таблицу при запуске без аргументов используются:
- **Credentials:** файл `google_sheets_credentials.json` в корне проекта
- **ID таблицы:** `GOOGLE_SHEETS_SPREADSHEET_ID` в `config_secrets.py` или переменная окружения (или `--spreadsheet-id ID`)

Опциональные:
- `--results-dir` - папка с результатами (по умолчанию: из `config.OUTPUT_DIR` или `results`)
- `--worksheet` - название листа (по умолчанию при полном экспорте: «F1 Scores» и «F1 Scores (прочее)»)
- `--group` - экспортировать только одну группу: `"массовая доля"` или `"прочее"` (без флага экспортируются обе на два листа)
- `--export-csv` - экспорт в CSV (укажите путь)
- `--no-upload` - только собрать и вывести метрики, не загружать в таблицу

**Примечания:**
- Метод определяется по структуре папок: `results/{model_key}/{method_folder}/metrics_*.json`
- Для каждой пары (модель, метод) берётся один последний запуск (по timestamp или дате файла)
- Зависимости: `pip install gspread google-auth` (указаны в requirements.txt)

## 6. Доступные модели

Список моделей задаётся в `models.yaml`. Ниже перечислены используемые в проекте модели (по состоянию конфигурации).

### 6.1. Локальные модели (загружаются с Hugging Face)

**Gemma:**
- `gemma-2-2b` — google/gemma-2-2b-it
- `gemma-3-4b` — google/gemma-3-4b-it
- `gemma-3-12b` — google/gemma-3-12b-it
- `codegemma-7b` — google/codegemma-7b-it

**Qwen:**
- `qwen-2.5-3b` — Qwen/Qwen2.5-3B-Instruct
- `qwen-3-4b` — Qwen/Qwen3-4B-Instruct-2507
- `qwen-3-8b` — Qwen/Qwen3-8B
- `qwen-3-14b` — Qwen/Qwen3-14B
- `qwen-3.5-4b` — Qwen/Qwen3.5-4B
- `qwen-3.5-9b` — Qwen/Qwen3.5-9B

**Mistral (Ministral-3):** загрузчик `load_mistral_3` выбирается автоматически по `name`.
- `mistral-3-3b-reasoning` — mistralai/Ministral-3-3B-Reasoning-2512
- `mistral-3-8b-instruct` — mistralai/Ministral-3-8B-Instruct-2512
- `mistral-3-14b-instruct` — mistralai/Ministral-3-14B-Instruct-2512

**Прочие локальные:**
- `gigachat3-10b-1.8b` — ai-sage/GigaChat3-10B-A1.8B-bf16

Для локальных моделей по умолчанию используется Flash Attention 2, если установлен пакет `flash-attn` (см. раздел 3.3 и 10.2).

### 6.2. API модели

**Через Google Generative AI API** (нужен `GEMINI_API_KEY`):
- `gemma-3-4b-api` — gemma-3-4b-it
- `gemma-3-12b-api` — gemma-3-12b-it
- `gemma-3-27b-api` — gemma-3-27b-it

**Через OpenRouter (или другой OpenAI-совместимый endpoint)** (нужен `OPENAI_API_KEY`, при необходимости `OPENAI_BASE_URL`):
- `mistral-small-3.1-24b-api` — mistralai/mistral-small-3.1-24b-instruct:free
- `qwen-3-32b-api` — qwen/qwen3-32b

**Требования для API моделей:**
- Для Gemma 3: `GEMINI_API_KEY`, библиотека `google-genai`
- Для OpenRouter: `OPENAI_API_KEY`, библиотека `openai`
- Модели работают через API, локальная GPU не требуется

**Пример запуска API модели:**
```bash
# Модели через Google Generative AI API
python main.py gemma-3-4b-api
python main.py gemma-3-12b-api --multi-agent simple_4agents

# Модели через OpenRouter API
python main.py mistral-small-3.1-24b-api
python main.py qwen-3-32b-api
python main.py qwen-3-32b-api --multi-agent simple_4agents
```

## 7. Метрики и результаты

### 7.1. Метрики оценки

Для каждой модели автоматически собираются следующие метрики:

**Производительность:**
- Средняя скорость ответа (секунды)
- Время загрузки модели
- Общее время инференса
- Использование GPU памяти (среднее, максимальное, минимальное)

**Качество парсинга:**
- Процент успешно распарсенных JSON ответов
- Список ошибок парсинга с полными ответами моделей

**Качество извлечения данных:**
Для групп "массовая доля" и "прочее":
- Accuracy (точность)
- Precision (прецизионность)
- Recall (полнота)
- F1-score
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)

### 7.2. Формат результатов

Результаты сохраняются в директории `results/`:

1. **results_model_name_prompt_timestamp.csv** - Детальные результаты для каждого текста:
   - `text` - исходный текст
   - `json` - извлеченный JSON из ответа модели
   - `json_parsed` - распарсенный JSON
   - `is_valid` - валидность JSON

2. **metrics_model_name_prompt_timestamp.json** - Сводные метрики:
   - **Информация о GPU:**
     - `gpu_info` - характеристики видеокарты (название, CUDA версия, общая память, версия драйвера)
     - `gpu_memory_after_load_gb` - память после загрузки модели
     - `gpu_memory_during_inference_gb` - средняя память во время инференса
     - `gpu_memory_during_inference_max_gb` - максимальная память во время инференса
     - `gpu_memory_during_inference_min_gb` - минимальная память во время инференса
   - **Метрики производительности:**
     - `average_response_time_seconds` - средняя скорость инференса (секунды на ответ)
     - `api_model` - флаг, является ли модель API-моделью
   - **Метрики качества:**
     - Метрики для групп "массовая доля" и "прочее" (accuracy, precision, recall, f1, tp, fp, fn)
     - `ошибки` - список ошибок с полями: `text_index`, `text`, `response`, `prompt`, `errors`
   - **Другая информация:**
     - Гиперпараметры модели
     - `prompt_template` - техническое название промпта
     - `prompt_designation` - обозначение использованного промпта (из `hyperparameters.prompt_template_name`, `config.PROMPT_TEMPLATE_NAME` или `multi_agent_{mode}`)
     - `prompt_full_text` - полный текст промпта
     - `prompt_info` - дополнительная информация о промптах (для мультиагентного режима)
     - Список ошибок парсинга

3. **raw_metrics_model_name_prompt_timestamp.json** - Метрики для raw output (без умных исправлений):
   - `validation` - статистика валидации через Pydantic
   - `массовая доля` - метрики качества для массовых долей
   - `прочее` - метрики качества для прочих параметров
   - `ошибки` - список ошибок с полями: `text_index`, `text`, `response`, `prompt`, `errors`

4. **evaluation_summary.log** - Консольный вывод последнего запуска оценки (`main.py` или `run_all_models.py`): всё, что выводится в stdout/stderr, дублируется в этот файл в реальном времени. Файл перезаписывается при каждом новом запуске.

5. **gemini_analysis_model_name_timestamp.json** - Анализ ошибок от Gemini API (если включен)

**Структура папок результатов:**

Результаты сохраняются в следующей структуре:
```
results/
  └── <model_key>/
      └── <prompt_folder_name>/
          ├── results_<model_name>_<timestamp>.csv
          ├── metrics_<model_name>_<timestamp>.json
          ├── raw_metrics_<model_name>_<timestamp>.json
          └── quality_errors_<model_name>_<timestamp>.txt
```

**Название папки промпта (`prompt_folder_name`):**
- Для одноагентного режима: название промпта из `hyperparameters.prompt_template_name` или `config.PROMPT_TEMPLATE_NAME`
- Для мультиагентного режима: префикс `MA_` + название режима (капсом). Для режимов с выбором базового промпта (`validation_fix_2agents`, `critic_3agents`) к имени папки добавляется название промпта (например, `MA_VALIDATION_FIX_2AGENTS_DETAILED_INSTR_ZEROSHOT_BASELINE`)
- Если используется `structured_output`:
  - С суффиксом `_structured` (если `structured_output=True`, без outlines/guidance)
  - С суффиксом `_outlines` (если `use_outlines=True`)
  - С суффиксом `_GUIDANCE` в названии папки (если `use_guidance=True`)

**Примеры структуры папок:**
- `results/qwen-2.5-3b/DETAILED_INSTR_ZEROSHOT/` - обычный режим
- `results/qwen-2.5-3b/DETAILED_INSTR_ZEROSHOT_structured/` - с structured output
- `results/qwen-2.5-3b/DETAILED_INSTR_ZEROSHOT_outlines/` - с structured output и outlines
- `results/qwen-2.5-3b/MA_SIMPLE_4AGENTS/` - мультиагентный режим
- `results/qwen-2.5-3b/MA_VALIDATION_FIX_2AGENTS_DETAILED_INSTR_ZEROSHOT_BASELINE/` - validation_fix_2agents с промптом DETAILED_INSTR_ZEROSHOT_BASELINE
- `results/qwen-2.5-3b/MA_SIMPLE_4AGENTS_structured/` - мультиагентный режим с structured output

**Примечание:** Название промпта добавляется в имя файла (например, `metrics_model_name_DETAILED_INSTR_ZEROSHOT_timestamp.json`), а также сохраняется в отдельном поле `prompt_designation` в JSON файле для удобного поиска и фильтрации результатов.

## 8. Промпты и конфигурация

### 8.1. Список промптов (prompt_config.py)

Все промпты находятся в `prompt_config.py`:

**Для одноагентного подхода:**
- **`DETAILED_INSTR_ZEROSHOT_BASELINE`** - детальный zero-shot промпт без примера (baseline)
- **`DETAILED_INSTR_ONESHOT`** - детальный промпт с примером текста и ответа (One-shot prompt)
- **`DETAILED_INSTR_ZEROSHOT_CD`** - zero-shot с примером JSON на латинице (режим CD / --outlines)
- **`DETAILED_INSTR_ONESHOT_CD`** - one-shot с примером JSON на латинице (режим CD)
- **`DETAILED_INSTR_ZEROSHOT_CD_RUS`** - zero-shot с примером JSON на кириллице (--guidance по умолчанию)
- **`DETAILED_INSTR_ONESHOT_CD_RUS`** - one-shot с примером JSON на кириллице
- **`MINIMAL_INSTR_FIVESHOT`** - минимальный инструктивный few-shot промпт с 5 примерами
- **`MINIMAL_INSTR_FIVESHOT_APIE`** - few-shot промпт с 5 примерами (APIE): примеры подставляются из XLSX или CSV (`few_shot_examples_<model_key>_*.xlsx` или `*.csv`). Можно сгенерировать CSV через `few_shot_extractor.py`, затем разметить через `label_few_shot_with_gemini.py` и сохранить в XLSX для ручной проверки. Оценка не запускается при отсутствии файла с примерами

**Для режима `simple_4agents`:**
- **`NUMERIC_FRAGMENTS_EXTRACTION_PROMPT`** - промпт для агента извлечения числовых фрагментов
- **`MASS_FRACTION_EXTRACTION_PROMPT`** - промпт для агента извлечения массовых долей
- **`OTHER_PARAMETERS_EXTRACTION_PROMPT`** - промпт для агента извлечения прочих параметров
- **`JSON_FORMATION_PROMPT`** - промпт для агента формирования JSON

**Для режима `critic_3agents`:**
- Используется промпт из `config.PROMPT_TEMPLATE_NAME` для агента-генератора (первоначальный ответ)
- **`CRITIC_PROMPT`** - промпт для агента-критика (анализ ответа)
- **`CORRECTOR_PROMPT`** - промпт для агента-исправителя (устранение ошибок)

**Для режима `qa_workflow`:**
- **`QA_NUTRIENTS_PROMPT`** - промпт для извлечения списка питательных веществ
- **`QA_NUTRIENT_PROMPT`** - промпт для извлечения массовой доли конкретного вещества (используется для каждого вещества)
- **`QA_STANDARD_PROMPT`** - промпт для извлечения стандарта (ГОСТ, ТУ и т.д.)
- **`QA_GRADE_PROMPT`** - промпт для извлечения марки удобрения
- **`QA_QUANTITY_PROMPT`** - промпт для извлечения количеств (массы, объемы, количество упаковок)

**Для режима `validation_fix_2agents`:**
- Для агента 1 используется тот же промпт, что и в одноагентном режиме (`build_prompt3`, т.е. из `config.PROMPT_TEMPLATE_NAME` или `--prompt`).
- **`VALIDATION_FIX_PROMPT`** - промпт для агента-исправителя: в него подставляются исходный промпт, исходный текст, невалидный JSON и ошибки валидации Pydantic; инструкция по схеме не дублируется, исправитель опирается на исходный промпт.

**Для structured output и outlines:** используется схема `FertilizerExtractionOutput` (раздел 4.3); рекомендуются промпты `*_CD`.

### 8.2. Поток данных в мультиагентном подходе

#### 8.2.1. Режим `simple_4agents`:

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

#### 8.2.2. Режим `critic_3agents`:

```
Исходный текст + промпт
    ↓
[Агент 1: Генератор - создание первоначального ответа]
    ↓
Первоначальный ответ
    ↓
[Агент 2: Критик - анализ ответа на соответствие промпту]
    ↓
Анализ ошибок
    ↓
[Агент 3: Исправитель - устранение найденных ошибок]
    ↓
Исправленный JSON
```

#### 8.2.3. Режим `qa_workflow`:

```
Исходный текст
    ↓
[Агент 1: Извлечение питательных веществ]
    ↓
Список веществ (N, P2O5, K2O, S, ...)
    ↓
[Агент 2: Извлечение массовых долей]
    ├─→ Запрос для N
    ├─→ Запрос для P2O5
    ├─→ Запрос для K2O
    └─→ ... (для каждого вещества)
    ↓
Массовые доли для всех веществ
    ↓
[Агент 3: Извлечение стандарта]
    ↓
Стандарт (ГОСТ, ТУ и т.д.)
    ↓
[Агент 4: Извлечение марки]
    ↓
Марка удобрения
    ↓
[Агент 5: Извлечение количеств]
    ↓
Количества (массы, объемы, упаковки)
    ↓
[Агент 6: Сборка финального JSON]
    ↓
Финальный JSON
```

#### 8.2.4. Режим `validation_fix_2agents`:

```
Исходный текст + промпт (как в одноагентном)
    ↓
[Агент 1: Генерация ответа и валидация Pydantic]
    ↓
    ├─ Валидный JSON → конец
    │
    └─ Невалидный JSON → ошибки валидации
            ↓
       [Агент 2: Исправитель — исходный промпт + текст + невалидный JSON + ошибки валидации]
            ↓
       Исправленный JSON (повторная валидация)
```

### 8.3. Формат данных

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

## 9. Добавление новой модели

Модели добавляются через YAML файл `models.yaml`. Загрузчик и функция генерации определяются автоматически: загрузчик — по полю `name` (или по ключу модели), генерация — по паттернам в ключе.

### 9.1. Добавление локальной модели

1. **Добавьте функцию загрузки в `model_loaders.py`** только если нужна особая логика (специальный класс, свои ошибки и т.д.). Для моделей с `name`, содержащим `gemma-3` или `ministral-3`/`mistral-3`, загрузчик подставляется автоматически.

2. **Добавьте функцию генерации в `model_loaders.py`** (если нужна специализированная):
```python
def generate_your_model(model, tokenizer, prompt: str, max_new_tokens: int = 1024, repetition_penalty: float = None) -> str:
    # Ваша логика генерации
    pass
```

3. **Добавьте конфигурацию в `models.yaml`**. Указывать `load_func` не обязательно: для Gemma 3 и Mistral 3 он выводится из `name`; для остальных — из ключа модели или fallback `load_standard_model`.
```yaml
models:
  your-model-key:
    name: "model/name"
    hyperparameters:
      max_new_tokens: 1024
      do_sample: false
      torch_dtype: "bfloat16"
      # prompt_template_name: "DETAILED_INSTR_ZEROSHOT_CD" — переопределение промпта (опционально)
      # torch_dtype: "nf4" — загрузка в 4-bit (любая поддерживаемая модель)
      # max_cpu_gb_4bit: 8 — лимит CPU RAM при 4-bit (опционально)
```

**Автоматическое определение загрузчика (load_func):**
- Если `load_func` не указан в YAML, загрузчик выбирается по полю `name`:
  - в `name` есть `gemma-3` → `load_gemma_3` (одна функция для всех размеров Gemma 3: 1b, 4b, 12b, 27b)
  - в `name` есть `ministral-3` или `mistral-3` → `load_mistral_3`
  - иначе → `load_{model_key.replace('-', '_').replace('.', '_').lower()}` (например, `load_qwen_2_5_3b`)
- Если функция по имени/ключу не найдена, используется универсальная `load_standard_model` с данным `name`.

**4-bit квантизация (nf4):** для любой локальной модели можно задать в `hyperparameters`:
- `torch_dtype: "nf4"` (или `"4bit"`) — загрузка в 4-bit через BitsAndBytes (экономия VRAM). Опционально: `max_cpu_gb_4bit: 8` — лимит CPU RAM в GB. Переменная окружения `MAX_CPU_GB_4BIT` (или `GEMMA_27B_4BIT_MAX_CPU_GB`) задаёт лимит по умолчанию (12).

**generate_func** определяется автоматически по паттернам:
- `qwen-3` → `generate_qwen_3`
- `qwen` → `generate_qwen`
- `gemma` → `generate_gemma`
- `t5` или `t5gemma` → `generate_t5`
- иначе → `generate_standard`

**Переопределение:** при необходимости укажите `load_func` или `generate_func` явно в YAML.

### 9.2. Добавление API модели

1. **Добавьте функцию загрузки в `model_loaders_api.py`** (если нужна специализированная):
```python
def load_your_api_model() -> Tuple[Any, Any]:
    # Для API моделей обычно возвращаются заглушки или клиенты API
    return None, None
```

2. **Добавьте функцию генерации в `model_loaders_api.py`** (если нужна специализированная):
```python
def generate_your_api_model(model, tokenizer, prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
    # Ваша логика генерации через API
    pass
```

3. **Добавьте конфигурацию в `models.yaml`**:
```yaml
models:
  your-model-api:
    name: "model/name"  # название модели для API
    hyperparameters:
      max_new_tokens: 512
      api_model: true  # обязательно для API моделей
```

**Автоматическое определение для API моделей:**
- Если в ключе модели есть `-api`, автоматически используется модуль `model_loaders_api`
- `generate_func` определяется автоматически:
  - `gemma` → `generate_gemma_api`
  - иначе → `generate_openrouter_api`

### 9.3. Примеры

**Локальная модель:**
```yaml
models:
  my-custom-model:
    name: "organization/my-custom-model"
    hyperparameters:
      max_new_tokens: 1024
      do_sample: false
      torch_dtype: "bfloat16"
```

**API модель:**
```yaml
models:
  my-custom-model-api:
    name: "organization/my-custom-model"
    hyperparameters:
      max_new_tokens: 512
      api_model: true
```

**С явным указанием загрузчика или генерации** (если нужно переопределить автоопределение):
```yaml
models:
  my-custom-model:
    name: "organization/my-custom-model"
    load_func: "load_my_custom_model"   # опционально
    generate_func: "generate_my_custom_model"   # опционально
    hyperparameters:
      max_new_tokens: 1024
      do_sample: false
      torch_dtype: "bfloat16"
```

### 9.4. Запуск новой модели

После добавления модели в `models.yaml` её можно запустить:
```bash
# Одноагентный режим
python main.py your-model-key

# Мультиагентный режим
python main.py your-model-key --multi-agent simple_4agents

# С structured output
python main.py your-model-key --structured-output
```

**Примечание:** Доступные модели можно посмотреть, запустив `python main.py` без аргументов.

## 10. Требования

### 10.1. Основные требования

- Python 3.8+
- PyTorch с поддержкой CUDA (для локальных моделей)
- Transformers библиотека
- LangGraph (для мультиагентного подхода)
- Другие зависимости из requirements.txt

### 10.2. Требования для локальных моделей

- CUDA-совместимая GPU (рекомендуется 8GB+ VRAM)
- Hugging Face Token (`HF_TOKEN`)
- **Flash Attention 2 (опционально, по умолчанию включено):** для ускорения и экономии VRAM установите `pip install flash-attn --no-build-isolation`. Требуется CUDA и совместимый компилятор (обычно доступно в Linux/WSL; на Windows часто недоступно). Если пакет не установлен, локальные модели работают без Flash Attention 2 (выводится предупреждение). Отключить: `set USE_FLASH_ATTENTION_2=0` / `export USE_FLASH_ATTENTION_2=0`.
- **Ошибка `CUDA error: an illegal instruction was encountered` (cudaErrorIllegalInstruction), в т.ч. на Qwen 3.5:** GPU выполняет инструкцию, которую не поддерживает — бинарники (PyTorch, flash-attn) часто собраны под более новую архитектуру (compute capability), чем у вашей видеокарты. Что попробовать: отключить Flash Attention 2 (`export USE_FLASH_ATTENTION_2=0`); проверить совместимость GPU и версии PyTorch; переустановить PyTorch под вашу версию CUDA и архитектуру; обновить драйвер NVIDIA.
- **4-bit (nf4):** при `torch_dtype: "nf4"` в hyperparameters нужны `bitsandbytes` и `accelerate`. Лимит CPU RAM при 4-bit: переменная окружения `MAX_CPU_GB_4BIT` или `GEMMA_27B_4BIT_MAX_CPU_GB` (по умолчанию 12 GB).

### 10.3. Требования для API моделей

- Google Generative AI API Key (`GEMINI_API_KEY`) для моделей Gemma 3
- OpenRouter API Key (`OPENAI_API_KEY`) для моделей через OpenRouter
- Библиотека `google-genai` (устанавливается через `pip install google-genai`) для моделей Gemma 3
- Библиотека `openai` (устанавливается через `pip install openai`) для моделей через OpenRouter

### 10.4. Требования для интеграции с Google Таблицами

Для использования модуля `google_sheets_integration.py`:
- `gspread` и `google-auth` (указаны в requirements.txt)
- Service Account в Google Cloud Console с включёнными Google Sheets API и Google Drive API
- JSON credentials в корне проекта как `google_sheets_credentials.json` (в .gitignore)
- Опционально: `GOOGLE_SHEETS_SPREADSHEET_ID` в `config_secrets.py` для запуска без аргументов

**Установка зависимостей:**
```bash
pip install gspread google-auth
```

**Настройка Service Account:**
1. Создайте проект в Google Cloud Console: https://console.cloud.google.com/
2. Включите Google Sheets API и Google Drive API
3. Создайте Service Account и скачайте JSON файл с credentials
4. Поделитесь Google Таблицей с email из Service Account

### 10.5. Специальные требования

**Для Mistral 3 моделей:**
- `transformers>=4.50.0.dev0`: `pip install git+https://github.com/huggingface/transformers`
- `mistral-common>=1.8.6`: см. `requirements.txt` или `pip install mistral-common --upgrade`

## 11. Справка по использованию

```bash
# Показать справку по main.py
python main.py

# Показать справку по reevaluate.py
python reevaluate.py
```

### 11.1. Сводная таблица команд

| Команда | Описание | Режим | Structured Output | Verbose | Gemini анализ |
|---------|----------|-------|-------------------|---------|---------------|
| `python main.py <model_key>` | Оценка одной модели | Одноагентный | Отключен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key1> <model_key2> ...` | Оценка нескольких моделей | Одноагентный | Отключен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key1>,<model_key2>,...` | Оценка нескольких моделей (через запятую) | Одноагентный | Отключен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key> --multi-agent <mode>` | Оценка одной модели | Мультиагентный | Отключен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key> --structured-output` | Оценка с Pydantic валидацией | Одноагентный | Включен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key> --multi-agent <mode> --structured-output` | Оценка с мультиагентным режимом и Pydantic | Мультиагентный | Включен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key> --structured-output --outlines` | Оценка с Pydantic + outlines (локальные модели) | Одноагентный | Включен | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key> --structured-output --pydantic-outlines` | Оценка с outlines (схема из Pydantic) | Одноагентный | Включен | Включен | По умолчанию |
| `python main.py <model_key> --guidance` | Оценка с llguidance (схема RUS по умолчанию) | Одноагентный | Включен | Включен | По умолчанию |
| `python main.py <model_key> --prompt NAME` | Оценка с указанным промптом | Одноагентный | — | Включен (по умолчанию) | По умолчанию |
| `python main.py <model_key> --no-gemini` | Оценка без анализа Gemini | Одноагентный | Отключен | Включен (по умолчанию) | Отключен |
| `python main.py <model_key> --vllm` | Оценка через vLLM (`<model_key>-vllm`) | Одноагентный | По флагам | Включен (по умолчанию) | По умолчанию |
| `python run_all_models.py` | Оценка всех моделей | Одноагентный | Отключен | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --local-only` | Оценка только локальных моделей | Одноагентный | Отключен | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --vllm` | Оценка всех vLLM-версий (`*-vllm`) | Одноагентный | По флагам | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --ollama` | Оценка всех Ollama-версий (`*-ollama`) | Одноагентный | По флагам | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --multi-agent <mode>` | Оценка всех моделей | Мультиагентный | Отключен | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --structured-output` | Оценка всех моделей | Одноагентный | Включен | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --local-only --structured-output --outlines` | Оценка локальных моделей с outlines | Одноагентный | Включен | Отключен (по умолчанию) | По умолчанию |
| `python run_all_models.py --local-only --structured-output --pydantic-outlines` | Оценка с outlines (схема из Pydantic) | Одноагентный | Включен | Отключен | По умолчанию |
| `python run_all_models.py --local-only --guidance` | Оценка с llguidance (схема RUS) | Одноагентный | Включен | Отключен | По умолчанию |
| `python google_sheets_integration.py` | Экспорт метрик в Google Таблицу (последние запуски) | — | — | — | — |
| `python run_all_models.py --prompt NAME [опции]` | Оценка с указанным промптом | — | — | — | — |
| `python reevaluate.py <csv_file>` | Переоценка результатов | - | - | - | Отключен |
| `python reevaluate.py <csv_file> [model_name] --gemini` | Переоценка с анализом Gemini | - | - | - | Включен |
| `python few_shot_extractor.py <model_key> [опции]` | Извлечение few-shot примеров (для режима APIE — обязательно перед оценкой) | - | - | Опционально (`--verbose`) | - |
