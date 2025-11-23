import numpy as np
from pathlib import Path
from configs.settings import settings
import logging

logger = logging.getLogger("comparison_service")

def compare_xyz_files(original_path: Path, predicted_path: Path) -> dict:
    """
    Сравнивает два XYZ файла (исходный и предсказанный), включая метки.
    
    Args:
        original_path: Путь к исходному файлу (class x y z)
        predicted_path: Путь к предсказанному файлу (class x y z)
        
    Returns:
        dict: Результаты сравнения
    """
    logger.info(f"🔍 Сравнение файлов: {original_path.name} vs {predicted_path.name}")
    
    # Загружаем исходный файл
    orig_data = np.loadtxt(original_path)
    orig_labels = orig_data[:, 0].astype(int)
    orig_points = orig_data[:, 1:4].astype(np.float32)
    
    # Загружаем предсказанный файл
    pred_data = np.loadtxt(predicted_path)
    pred_labels = pred_data[:, 0].astype(int)
    pred_points = pred_data[:, 1:4].astype(np.float32)
    
    # Проверка совпадения размеров
    if orig_points.shape != pred_points.shape:
        raise ValueError(
            f"❌ Размеры не совпадают: "
            f"оригинал {orig_points.shape}, предсказание {pred_points.shape}"
        )
    
    # Проверка совпадения координат (с небольшой погрешностью)
    coords_match = np.allclose(orig_points, pred_points, atol=1e-6)
    
    if not coords_match:
        logger.warning("⚠️ Координаты в файлах НЕ совпадают!")
        logger.warning("⚠️ Возможно, файлы из разных систем координат.")
    else:
        logger.info("✅ Координаты совпадают")
    
    # Сравнение меток
    total_points = len(orig_labels)
    matching_labels = (orig_labels == pred_labels).sum()
    accuracy = matching_labels / total_points * 100.0
    
    # Статистика по классам
    orig_unique, orig_counts = np.unique(orig_labels, return_counts=True)
    pred_unique, pred_counts = np.unique(pred_labels, return_counts=True)
    
    orig_stats = dict(zip(orig_unique, orig_counts))
    pred_stats = dict(zip(pred_unique, pred_counts))
    
    # Статистика по классам с именами
    class_names = settings.CLASS_MAPPING
    detailed_stats = {}
    
    all_classes = set(orig_unique).union(set(pred_unique))
    
    for cls in sorted(all_classes):
        orig_count = orig_stats.get(cls, 0)
        pred_count = pred_stats.get(cls, 0)
        orig_pct = orig_count / total_points * 100.0
        pred_pct = pred_count / total_points * 100.0
        
        class_name = class_names.get(cls, f"Class {cls}")
        detailed_stats[class_name] = {
            "original": f"{orig_count:,} ({orig_pct:.2f}%)",
            "predicted": f"{pred_count:,} ({pred_pct:.2f}%)",
            "diff": f"{pred_count - orig_count:,} ({(pred_pct - orig_pct):+.2f}%)"
        }
    
    result = {
        "total_points": total_points,
        "coordinates_match": bool(coords_match),
        "matching_labels": int(matching_labels),
        "accuracy": round(accuracy, 2),
        "original_stats": {
            "unique_classes": sorted(orig_unique.tolist()),
            "total_classes": len(orig_unique)
        },
        "predicted_stats": {
            "unique_classes": sorted(pred_unique.tolist()),
            "total_classes": len(pred_unique)
        },
        "detailed_class_stats": detailed_stats
    }
    
    logger.info(f"📊 Точность: {accuracy:.2f}% ({matching_labels}/{total_points})")
    logger.info(f"📊 Уникальных классов: ориг. {len(orig_unique)}, предсказ. {len(pred_unique)}")
    
    return result