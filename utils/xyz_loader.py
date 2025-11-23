import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from configs.settings import settings
import logging

logger = logging.getLogger("xyz_loader")

# ==================== ПРЕОБРАЗОВАНИЕ МЕТОК ====================

def labels_to_indices(labels):
    """Преобразование меток 1-8 → индексы 0-7 для PyTorch"""
    return labels - settings.MIN_LABEL

def indices_to_labels(indices):
    """Преобразование индексов 0-7 → метки 1-8 для сохранения"""
    return indices + settings.MIN_LABEL

# ==================== ЗАГРУЗКА ДАННЫХ ====================

def load_xyz_with_labels(file_path: Path):
    """
    Загрузить файл формата: class x y z
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        points: np.ndarray (N, 3) - координаты XYZ
        labels: np.ndarray (N,) - метки классов (1-8)
    """
    data = np.loadtxt(file_path)
    
    if data.shape[1] != 4:
        raise ValueError(
            f"❌ Неверный формат файла {file_path}: {data.shape[1]} столбцов. "
            f"Ожидается 4 столбца (class x y z)"
        )
    
    labels = data[:, 0].astype(int)
    points = data[:, 1:4].astype(np.float32)

    # Проверка меток
    unique_labels = set(np.unique(labels))
    expected_classes = set(settings.CLASS_MAPPING.keys())  # {1, 2, ..., 8}
    
    if not unique_labels.issubset(expected_classes):
        problematic_classes = unique_labels - expected_classes
        error_msg = (
            f"❌ В файле {file_path.name} найдены недопустимые метки: {sorted(problematic_classes)}. "
            f"Ожидаемые метки: {sorted(expected_classes)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        logger.info(f"✅ Файл {file_path.name} содержит допустимые метки: {sorted(unique_labels)}")
    
    logger.info(f"📂 Загружено {len(points):,} точек с метками")
    logger.info(f"   X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    logger.info(f"   Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    logger.info(f"   Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    return points, labels  # ✅ Возвращаем метки 1-8 (преобразование в датасете)


def load_xyz_no_labels(file_path: Path):
    """
    Загрузить XYZ файл БЕЗ меток или С метками (автоопределение)
    
    Поддерживаемые форматы:
    - x y z (3 столбца)
    - class x y z (4 столбца, метки игнорируются)
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        points: np.ndarray (N, 3) - координаты XYZ
    """
    data = np.loadtxt(file_path)
    
    # Автоматическое определение формата
    if data.shape[1] == 3:
        points = data[:, :3].astype(np.float32)
        logger.info(f"📂 Загружен файл БЕЗ меток: {file_path.name}")
    elif data.shape[1] == 4:
        points = data[:, 1:4].astype(np.float32)
        logger.info(f"📂 Загружен файл С метками (метки проигнорированы): {file_path.name}")
    else:
        raise ValueError(
            f"❌ Неверный формат файла {file_path.name}: {data.shape[1]} столбцов. "
            f"Ожидается 3 (x y z) или 4 (class x y z)"
        )
    
    logger.info(f"   Точек: {len(points):,}")
    logger.info(f"   X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    logger.info(f"   Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    logger.info(f"   Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    return points


def save_xyz_with_labels(file_path: Path, points: np.ndarray, labels: np.ndarray):
    """
    Сохранить файл в формате: class x y z
    
    Args:
        file_path: Путь для сохранения
        points: np.ndarray (N, 3) - координаты XYZ
        labels: np.ndarray (N,) - метки классов (1-8)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if points.shape[0] != labels.shape[0]:
        raise ValueError(
            f"❌ Несовпадение размеров: points={points.shape[0]}, labels={labels.shape[0]}"
        )
    
    if points.shape[1] != 3:
        raise ValueError(
            f"❌ Неверная размерность координат: {points.shape[1]} вместо 3"
        )
    
    # Проверка диапазонов меток
    unique_labels = np.unique(labels)
    expected_range = set(settings.CLASS_MAPPING.keys())
    
    if not set(unique_labels).issubset(expected_range):
        logger.warning(f"⚠️ Метки вне допустимого диапазона: {unique_labels}")
        logger.warning(f"⚠️ Ожидаемые метки: {sorted(expected_range)}")
    
    logger.info(f"💾 Сохранение {len(points):,} точек в {file_path.name}")
    logger.info(f"   Метки: {sorted(unique_labels)}")
    logger.info(f"   X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    logger.info(f"   Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    logger.info(f"   Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Сохранение
    out = np.hstack([labels.reshape(-1, 1), points])
    np.savetxt(file_path, out, fmt="%d %.6f %.6f %.6f", delimiter=' ')
    
    logger.info(f"✅ Файл сохранён: {file_path}")


# ==================== ДАТАСЕТЫ ====================

class MultiXYZDataset(Dataset):
    """
    Датасет для нескольких XYZ файлов с метками
    
    Args:
        file_paths: Список путей к файлам (формат: class x y z)
        num_points: Количество точек в батче
        normalize: Применять ли нормализацию
    """
    def __init__(self, file_paths: list[Path], num_points: int, normalize=True):
        self.points, self.labels = [], []

        for fp in file_paths:
            p, l = load_xyz_with_labels(fp)
            self.points.append(p)
            self.labels.append(l)

        self.points = np.vstack(self.points)
        self.labels = np.hstack(self.labels)
        self.num_points = num_points
        self.normalize = normalize
        
        logger.info(f"📊 Объединённый датасет: {len(self.points):,} точек из {len(file_paths)} файлов")

    def __len__(self):
        return len(self.points) // self.num_points

    def __getitem__(self, idx):
        start = idx * self.num_points
        end = start + self.num_points
        
        pts = self.points[start:end].copy()
        lbl = self.labels[start:end].copy()

        if len(pts) < self.num_points:
            return self[0]

        # ЛОКАЛЬНАЯ нормализация координат
        if self.normalize:
            centroid = pts.mean(axis=0)
            pts -= centroid
            max_dist = np.max(np.sqrt(np.sum(pts**2, axis=1)))
            if max_dist > 0:
                pts /= max_dist

        # ПРЕОБРАЗОВАНИЕ меток 1-8 → 0-7
        lbl = labels_to_indices(lbl)

        return (
            torch.tensor(pts, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.long)
        )