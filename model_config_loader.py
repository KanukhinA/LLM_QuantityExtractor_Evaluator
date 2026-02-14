"""
Загрузчик конфигураций моделей из YAML файла
"""
import os
import yaml


def load_model_configs():
    """
    Загружает конфигурации моделей из models.yaml
    Использует ленивый импорт для избежания циклических зависимостей
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "Библиотека PyYAML не установлена. Установите: pip install pyyaml"
        )
    
    # Загружаем YAML файл
    yaml_path = os.path.join(os.path.dirname(__file__), "models.yaml")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # Ленивый импорт модулей для загрузки функций
    model_loaders_module = None
    model_loaders_api_module = None
    
    configs = {}
    for model_key, model_config in yaml_data['models'].items():
        # Автоматическое определение модулей и функций, если не указаны явно
        is_api_model = "-api" in model_key
        
        # Определяем load_module
        if 'load_module' in model_config:
            load_module_name = model_config['load_module']
        else:
            load_module_name = "model_loaders_api" if is_api_model else "model_loaders"
        
        # Определяем generate_module
        if 'generate_module' in model_config:
            generate_module_name = model_config['generate_module']
        else:
            generate_module_name = "model_loaders_api" if is_api_model else "model_loaders"
        
        # Определяем load_func: явно в YAML или по name (для локальных моделей)
        if 'load_func' in model_config:
            load_func_name = model_config['load_func']
        else:
            # По имени модели выбираем загрузчик (только для локальных — API не трогаем)
            name = (model_config.get('name') or '').lower()
            if not is_api_model and 'gemma-3' in name:
                load_func_name = 'load_gemma_3'
            elif not is_api_model and ('mistralai/' in name or 'ministral' in name):
                load_func_name = 'load_mistral_3'
            else:
                load_func_name = f"load_{model_key.replace('-', '_').replace('.', '_').lower()}"
        
        # Определяем generate_func
        if 'generate_func' in model_config:
            generate_func_name = model_config['generate_func']
        else:
            # Автоматическое определение по паттернам
            if is_api_model:
                # Для API моделей
                if "gemma" in model_key:
                    generate_func_name = "generate_gemma_api"
                else:
                    generate_func_name = "generate_openrouter_api"
            else:
                # Для локальных моделей
                if "qwen-3" in model_key:
                    generate_func_name = "generate_qwen_3"
                elif "qwen" in model_key:
                    generate_func_name = "generate_qwen"
                elif "gemma" in model_key:
                    generate_func_name = "generate_gemma"
                elif "t5" in model_key or "t5gemma" in model_key:
                    generate_func_name = "generate_t5"
                else:
                    generate_func_name = "generate_standard"
        
        # Импортируем load_module если нужно
        if load_module_name == "model_loaders" and model_loaders_module is None:
            import model_loaders as model_loaders_module
        elif load_module_name == "model_loaders_api" and model_loaders_api_module is None:
            import model_loaders_api as model_loaders_api_module
        
        # Импортируем generate_module если нужно
        if generate_module_name == "model_loaders" and model_loaders_module is None:
            import model_loaders as model_loaders_module
        elif generate_module_name == "model_loaders_api" and model_loaders_api_module is None:
            import model_loaders_api as model_loaders_api_module
        
        # Преобразуем гиперпараметры (конвертируем строки в нужные типы)
        # Делаем это до создания load_func, чтобы использовать преобразованные значения
        hyperparameters = {}
        for key, value in model_config['hyperparameters'].items():
            if isinstance(value, str):
                # Конвертируем строковые значения
                if value == "bfloat16" or value == "auto":
                    hyperparameters[key] = value
                elif value.lower() == "true":
                    hyperparameters[key] = True
                elif value.lower() == "false":
                    hyperparameters[key] = False
                else:
                    hyperparameters[key] = value
            else:
                hyperparameters[key] = value
        
        # Получаем функции по имени
        # Если индивидуальная функция не найдена, используем универсальную (только для локальных моделей)
        if load_module_name == "model_loaders":
            try:
                raw_load = getattr(model_loaders_module, load_func_name)
                _hp = hyperparameters
                _name = model_config["name"]
                # Загрузчики с именем модели из конфига (не из сигнатуры)
                if load_func_name == "load_gemma_3":
                    load_func = (lambda name, hp: lambda: raw_load(name, hyperparameters=hp))(_name, _hp)
                elif load_func_name == "load_mistral_3":
                    load_func = (lambda name, hp: lambda: raw_load(name, hyperparameters=hp))(_name, _hp)
                else:
                    load_func = (lambda h: lambda: raw_load(h))(_hp)
            except AttributeError:
                # Используем универсальную функцию загрузки как fallback
                model_name = model_config['name']
                _hp = hyperparameters

                def load_func():
                    from model_loaders import load_standard_model
                    return load_standard_model(
                        model_name=model_name,
                        dtype=_hp.get('dtype'),
                        torch_dtype=_hp.get('torch_dtype'),
                        device_map=_hp.get('device_map', 'auto'),
                        trust_remote_code=_hp.get('trust_remote_code', True),
                        hyperparameters=_hp,
                    )
        else:
            load_func = getattr(model_loaders_api_module, load_func_name)
        
        if generate_module_name == "model_loaders":
            generate_func = getattr(model_loaders_module, generate_func_name)
        else:
            generate_func = getattr(model_loaders_api_module, generate_func_name)
        
        configs[model_key] = {
            "name": model_config['name'],
            "load_func": load_func,
            "generate_func": generate_func,
            "hyperparameters": hyperparameters
        }
    
    return configs


# Класс-прокси для ленивой загрузки конфигураций моделей
class ModelConfigsProxy:
    """Прокси-класс для ленивой загрузки MODEL_CONFIGS из YAML"""
    def __init__(self):
        self._configs = None
    
    def _load(self):
        """Загружает конфигурации при первом обращении"""
        if self._configs is None:
            self._configs = load_model_configs()
        return self._configs
    
    def __getitem__(self, key):
        return self._load()[key]
    
    def __iter__(self):
        return iter(self._load())
    
    def keys(self):
        return self._load().keys()
    
    def values(self):
        return self._load().values()
    
    def items(self):
        return self._load().items()
    
    def get(self, key, default=None):
        return self._load().get(key, default)
    
    def __contains__(self, key):
        return key in self._load()
    
    def __len__(self):
        return len(self._load())
    
    def __repr__(self):
        if self._configs is None:
            return "<ModelConfigsProxy (not loaded)>"
        return repr(self._configs)


# Создаем прокси-объект для ленивой загрузки
MODEL_CONFIGS = ModelConfigsProxy()

