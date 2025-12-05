import json
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

def get_model_architectures():
    """Получить список архитектур моделей (.py файлы)"""
    # Создаем логгер для управляющих функций (без WebSocket)
    logger = get_logger('management')
    
    architectures = []
    
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            logger.warning(f"Папка {models_dir} не найдена")
            return architectures
        
        for file in models_dir.iterdir():
            if file.is_file() and file.suffix == '.py' and not file.name.startswith('__'):
                stat = file.stat()
                
                architectures.append({
                    "name": file.stem,
                    "file": file.name,
                    "size": f"{stat.st_size / 1024:.1f} KB",
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                })
        
        logger.info(f"Найдено {len(architectures)} архитектур моделей")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке архитектур: {e}", exc_info=True)
    
    return architectures

def get_trained_models():
    """Получить список обученных моделей (.pth файлы)"""
    logger = get_logger('management')

    trained_models = []
    checkpoints_dir = Path("checkpoints")

    try:
        if not checkpoints_dir.exists():
            logger.warning(f"Папка {checkpoints_dir} не найдена")
            return trained_models

        # Рекурсивно ищем .pth файлы
        for pth_file in checkpoints_dir.rglob("*.pth"):
            stat = pth_file.stat()
            rel_path = pth_file.relative_to(checkpoints_dir)
            model_folder = rel_path.parent

            model_info = get_model_info(pth_file.parent, pth_file.name) 

            model_name_from_config = model_info.get('model_name')
            if model_name_from_config and model_name_from_config != 'Unknown':
                # Если в конфиге есть валидное имя модели, используем его
                display_name_base = model_name_from_config
            else:
                # Иначе используем имя папки как базу
                display_name_base = str(model_folder)

            file_suffix = pth_file.stem

            if file_suffix == 'model':
                display_name = f"{display_name_base} ({file_suffix})"
            else:
                display_name = f"{display_name_base} ({file_suffix})"

            folder_with_base = f"checkpoints/{model_folder.as_posix()}"

            trained_models.append({
                "name": pth_file.name,
                "display_name": display_name,
                "path": str(rel_path),
                "full_path": str(pth_file),
                "size": f"{stat.st_size / 1024 / 1024:.1f} MB",
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "folder": folder_with_base,
                "info": model_info
            })

        # Сортируем по дате (новые сверху)
        trained_models.sort(key=lambda x: x["modified"], reverse=True)
        logger.info(f"Найдено {len(trained_models)} обученных моделей")

    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {e}", exc_info=True)

    return trained_models

def get_model_info(model_dir: Path, model_file: str):
    """Получить информацию о модели из конфигурации"""
    # Создаем логгер для управляющих функций (без WebSocket)
    logger = get_logger('management')
    
    try:
        config_path = model_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {
                    'model_name': config.get('model_name', 'Unknown'),
                    'epochs': config.get('training_config', {}).get('epochs', 'Unknown'),
                    'dataset': config.get('training_config', {}).get('las_file', 'Unknown'),
                    'final_loss': config.get('final_metrics', {}).get('train_loss', 'Unknown')
                }
    except Exception as e:
        logger.error(f"Ошибка чтения config из {model_dir}: {e}")
    
    return {}

def get_datasets_list(subdir: str = None):
    """
    Возвращает .xyz-файлы из поддиректорий datasets/
    Если subdir не указан, возвращает все поддиректории
    """
    # Создаем логгер для управляющих функций (без WebSocket)
    logger = get_logger('management')
    
    datasets_dir = Path("datasets")
    
    if subdir:
        # Возвращаем файлы из конкретной поддиректории
        search_dir = datasets_dir / subdir
        if not search_dir.exists():
            logger.warning(f"Папка {search_dir} не найдена")
            return []
        
        files = []
        for xyz_file in search_dir.glob("*.xyz"):
            stat = xyz_file.stat()
            points_count = get_points_count_xyz(xyz_file)
            files.append({
                "name": xyz_file.stem,
                "file": xyz_file.name,
                "size": f"{stat.st_size / 1024 / 1024:.1f} MB",
                "points": f"{points_count:,}"
            })
        
        logger.info(f"Найдено {len(files)} .xyz файлов в {subdir}")
        return files
    
    else:
        # Возвращаем файлы из всех поддиректорий
        result = {}
        for sub_dir in datasets_dir.iterdir():
            if sub_dir.is_dir():
                subdir_name = sub_dir.name
                files = []
                for xyz_file in sub_dir.glob("*.xyz"):
                    stat = xyz_file.stat()
                    points_count = get_points_count_xyz(xyz_file)
                    files.append({
                        "name": xyz_file.stem,
                        "file": xyz_file.name,
                        "size": f"{stat.st_size / 1024 / 1024:.1f} MB",
                        "points": f"{points_count:,}"
                    })
                result[subdir_name] = files
        
        logger.info(f"Найдены поддиректории: {list(result.keys())}")
        return result

def get_points_count_xyz(file_path: Path) -> int:
    """Считаем строки в .xyz-файле"""
    # Создаем логгер для управляющих функций (без WebSocket)
    logger = get_logger('management')
    
    try:
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Ошибка чтения {file_path}: {e}")
        return 0