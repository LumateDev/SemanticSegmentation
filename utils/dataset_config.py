"""
Управление конфигурациями датасетов
"""

import yaml
from pathlib import Path
import numpy as np


class DatasetConfig:
    """Конфигурация датасета"""
    
    def __init__(self, config_path):
        """
        Args:
            config_path: путь к YAML файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.name = config['name']
        self.description = config.get('description', '')
        
        # Классы
        self.original_classes = config['original_classes']
        self.standard_classes = config['standard_classes']
        self.num_classes = config['num_classes']
        
        # Маппинг
        self.class_mapping = {int(k): int(v) for k, v in config['class_mapping'].items()}
        self.reverse_mapping = {int(k): int(v) for k, v in config['reverse_mapping'].items()}
        
        # Визуализация
        self.class_colors = {int(k): v for k, v in config['class_colors'].items()}
        
        # Признаки
        self.use_features = config.get('use_features', True)
        self.feature_list = config.get('feature_list', ['intensity', 'return_number', 'number_of_returns'])
    
    def map_labels(self, labels):
        """
        Маппинг исходных меток в стандартные категории
        
        Args:
            labels: numpy array с исходными метками
        Returns:
            mapped_labels: numpy array со стандартными метками
        """
        mapped = np.full_like(labels, -1, dtype=np.int32)
        
        for original, standard in self.class_mapping.items():
            mapped[labels == original] = standard
        
        return mapped
    
    def reverse_map_labels(self, labels):
        """
        Обратный маппинг стандартных меток в исходные
        
        Args:
            labels: numpy array со стандартными метками
        Returns:
            original_labels: numpy array с исходными метками
        """
        original = np.zeros_like(labels, dtype=np.uint8)
        
        for standard, orig in self.reverse_mapping.items():
            original[labels == standard] = orig
        
        return original
    
    def get_class_name(self, class_id):
        """Получить название класса"""
        return self.standard_classes.get(class_id, f'Class {class_id}')
    
    def get_class_color(self, class_id):
        """Получить цвет класса"""
        return self.class_colors.get(class_id, [128, 128, 128])
    
    def print_info(self):
        """Вывод информации о конфигурации"""
        print(f"\n{'='*70}")
        print(f"📋 Dataset Configuration: {self.name}")
        print(f"{'='*70}")
        print(f"Description: {self.description}")
        print(f"\n📊 Стандартизированные классы: {self.num_classes}")
        for cls_id, cls_name in self.standard_classes.items():
            print(f"   {cls_id}: {cls_name}")
        
        print(f"\n🔄 Маппинг классов:")
        for orig, std in self.class_mapping.items():
            orig_name = self.original_classes.get(str(orig), f'Class {orig}')
            std_name = self.standard_classes.get(std, f'Class {std}')
            print(f"   {orig} ({orig_name}) → {std} ({std_name})")
        
        print(f"\n🎨 Использование признаков: {self.use_features}")
        if self.use_features:
            print(f"   Признаки: {', '.join(self.feature_list)}")


# Автоопределение конфигурации по файлу
def auto_detect_config(las_file):
    """
    Автоматическое определение конфигурации по имени файла
    
    Args:
        las_file: путь к LAS файлу
    Returns:
        DatasetConfig или None
    """
    las_file = Path(las_file)
    configs_dir = Path('configs/datasets')
    
    # Словарь паттернов
    patterns = {
        'neon': 'neon_sample.yaml',
        'univer': 'univer2019.yaml',
    }
    
    # Поиск по имени файла
    filename_lower = las_file.stem.lower()
    
    for pattern, config_file in patterns.items():
        if pattern in filename_lower:
            config_path = configs_dir / config_file
            if config_path.exists():
                print(f"🔍 Автоопределена конфигурация: {config_file}")
                return DatasetConfig(config_path)
    
    return None


# Список доступных конфигураций
def list_available_configs():
    """Список всех доступных конфигураций"""
    configs_dir = Path('configs/datasets')
    
    if not configs_dir.exists():
        return []
    
    config_files = list(configs_dir.glob('*.yaml'))
    configs = []
    
    for config_file in config_files:
        try:
            config = DatasetConfig(config_file)
            configs.append({
                'file': config_file.name,
                'name': config.name,
                'num_classes': config.num_classes
            })
        except:
            pass
    
    return configs


if __name__ == '__main__':
    # Тестирование
    print("🧪 Тестирование конфигураций датасетов\n")
    
    # Создаем директорию
    Path('configs/datasets').mkdir(parents=True, exist_ok=True)
    
    # Список конфигураций
    print("📋 Доступные конфигурации:")
    configs = list_available_configs()
    
    if not configs:
        print("   ❌ Конфигурации не найдены!")
        print("\n💡 Создайте YAML файлы в configs/datasets/")
    else:
        for cfg in configs:
            print(f"   • {cfg['file']}: {cfg['name']} ({cfg['num_classes']} классов)")
        
        # Тест загрузки
        print("\n" + "="*70)
        for cfg_info in configs:
            config = DatasetConfig(f"configs/datasets/{cfg_info['file']}")
            config.print_info()