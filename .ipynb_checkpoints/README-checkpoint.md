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
```

**Важно:**
- `HF_TOKEN` - ОБЯЗАТЕЛЬНЫЙ. Без него программа не запустится.
- `GEMINI_API_KEY` - опциональный. Без него будет пропущен анализ ошибок через Gemini API, но оценка моделей будет работать.
- Файл `config_secrets.py` автоматически игнорируется git (добавлен в .gitignore) для безопасности.

**Альтернатива:** Если не хотите использовать файл, можно установить переменные окружения:
- Windows: `set HF_TOKEN=your_token_here`
- Linux/Mac: `export HF_TOKEN=your_token_here`

**Где получить ключи:**
- Hugging Face Token: https://huggingface.co/settings/tokens
- Gemini API Key: https://aistudio.google.com/app/apikey

## Структура проекта

```
SmallLLMEvaluator/
├── main.py                 # Основной скрипт для запуска оценки одной модели
├── run_all_models.py       # Скрипт для запуска оценки всех моделей
├── reevaluate.py           # Скрипт для переоценки результатов из файла
├── model_evaluator.py      # Класс для оценки моделей
├── model_loaders.py         # Функции загрузки различных моделей
├── metrics.py               # Функции расчета метрик качества
├── utils.py                 # Утилиты для парсинга JSON и построения промптов
├── gemini_analyzer.py      # Интеграция с Gemini API для анализа ошибок
├── gpu_info.py             # Функции для получения информации о GPU
├── config.py               # Конфигурация проекта
├── requirements.txt        # Зависимости проекта
└── README.md              # Документация
```

## Использование

### Оценка одной модели

Запустите `main.py` и выберите модель в коде:

```python
# В main.py измените MODEL_CONFIG
MODEL_CONFIG = MODEL_CONFIGS["qwen-2.5-3b"]
```

Затем запустите:
```bash
python main.py
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

### Переоценка результатов

Если нужно пересчитать метрики качества из сохраненного CSV файла без повторного запуска модели:

```bash
python reevaluate.py results/results_model_name_timestamp.csv
```

Или с указанием имени модели:
```bash
python reevaluate.py results/results_model_name_timestamp.csv "model/name"
```

## Доступные модели

Проект поддерживает следующие модели:

- **google/gemma-2-2b-it** - Gemma 2.2B Instruct
- **google/gemma-3-4b-it** - Gemma 3 4B Instruct (с квантизацией и без)
- **Qwen/Qwen2.5-1.5B-Instruct** - Qwen 2.5 1.5B
- **Qwen/Qwen2.5-3B-Instruct** - Qwen 2.5 3B
- **Qwen/Qwen2.5-4B-Instruct** - Qwen 2.5 4B
- **mistralai/Ministral-3-3B-Reasoning-2512** - Ministral 3 3B Reasoning
- **mistralai/Ministral-3-3B-Instruct-2512** - Ministral 3 3B Instruct
- **AI4Chem/CHEMLLM-2b-1_5** - CHEMLLM 2B
- **microsoft/Phi-3.5-mini-instruct** - Phi 3.5 Mini
- **microsoft/Phi-4-mini-instruct** - Phi 4 Mini
- **unsloth/mistral-7b-v0.3-bnb-4bit** - Mistral 7B (предквантизированная)

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

- `DATASET_PATH` - путь к датасету (Excel файл)
- `GROUND_TRUTH_PATH` - путь к файлу с ground truth (опционально, используется колонка json_parsed из датасета)

**API ключи должны быть установлены через переменные окружения:**
- `HF_TOKEN` - токен Hugging Face (ОБЯЗАТЕЛЬНЫЙ)
- `GEMINI_API_KEY` - ключ Gemini API (опциональный, для анализа ошибок)

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

2. Добавьте конфигурацию в `main.py` в словарь `MODEL_CONFIGS`:
```python
"your-model-key": {
    "name": "model/name",
    "load_func": ml.load_your_model,
    "generate_func": ml.generate_standard,  # или ml.generate_qwen, ml.generate_gemma
    "hyperparameters": {
        "max_new_tokens": 1024,
        "do_sample": False,
        "dtype": "bfloat16"
    }
}
```

## Требования

- Python 3.8+
- CUDA-совместимая GPU (рекомендуется 8GB+ VRAM)
- PyTorch с поддержкой CUDA
- Transformers библиотека
- Другие зависимости из requirements.txt

## Лицензия

Проект создан для исследовательских целей.
