import numpy as np
from pathlib import Path
from configs.settings import settings
import logging

logger = logging.getLogger("dataset_stats_service")

def analyze_xyz_dataset(file_path: Path) -> dict:
    """
    Анализирует XYZ файл и возвращает статистику по классам.
    
    Args:
        file_path: Путь к .xyz файлу (class x y z)
        
    Returns:
        dict: Статистика по классам
    """
    logger.info(f"🔍 Анализ датасета: {file_path.name}")
    
    try:
        # Загружаем файл
        data = np.loadtxt(file_path)
        
        if data.shape[1] != 4:
            raise ValueError(
                f"❌ Неверный формат файла: {data.shape[1]} столбцов. Ожидается 4 (class x y z)"
            )
        
        labels = data[:, 0].astype(int)
        total_points = len(labels)
        
        # Подсчитываем уникальные метки и их количество
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Статистика по классам
        class_names = settings.CLASS_MAPPING
        class_stats = {}
        
        for cls, count in zip(unique_labels, counts):
            pct = count / total_points * 100.0
            class_name = class_names.get(int(cls), f"Class {cls}")
            class_stats[class_name] = {
                "label": int(cls),
                "count": int(count),
                "percentage": round(pct, 2)
            }
        
        result = {
            "file": file_path.name,
            "total_points": total_points,
            "unique_classes": len(unique_labels),
            "class_stats": class_stats
        }
        
        logger.info(f"✅ Файл {file_path.name} проанализирован: {total_points:,} точек, {len(unique_labels)} классов")
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа {file_path}: {e}", exc_info=True)
        raise