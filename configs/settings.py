from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Dict, Set

class Settings(BaseSettings):
    # Пути
    MODELS_DIR: Path = Path("models")
    CHECKPOINTS_DIR: Path = Path("checkpoints")
    DATASETS_DIR: Path = Path("datasets")
    LOGS_DIR: Path = Path("logs")
    
    # ML параметры
    DEFAULT_NUM_POINTS: int = 4096
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_EPOCHS: int = 3
    DEFAULT_LEARNING_RATE: float = 0.001
    
    # API
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    CLASS_MAPPING: dict[int, str] = {
        1: 'default',
        2: 'ground',
        3: 'low_green',
        4: 'mid_green',
        5: 'high_green',
        6: 'roofs',
        7: 'false_point',
        8: 'service_A_points',
    }
    NUM_CLASSES: int = len(CLASS_MAPPING)
    MIN_LABEL: int = min(CLASS_MAPPING.keys())  # 1
    MAX_LABEL: int = max(CLASS_MAPPING.keys())  # 8

    # Папки для логов
    LOG_DIRS: Dict[str, str] = {
        'test': 'logs/test',
        'train': 'logs/train', 
        'predict': 'logs/predict',
        'management': 'logs/management',
        'default': 'logs'
    }
    
    # Модули, для которых отключено логирование в файл
    DISABLE_FILE_LOGGING: Set[str] = {'management'}
    
    # Модули, для которых отключено логирование в консоль
    DISABLE_CONSOLE_LOGGING: Set[str] = set()
    
    # Модули, для которых отключено WebSocket логирование, если они поддерживат его
    DISABLE_WEBSOCKET_LOGGING: Set[str] = set()

    class Config:
        env_file = ".env"

settings = Settings()