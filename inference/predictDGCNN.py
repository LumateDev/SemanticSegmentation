"""
Применение обученной DGCNN модели к неразмеченным LiDAR данным
Сохранение результатов с предсказанными классами и визуализация
"""

import torch
import numpy as np
import laspy
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modelDGCNN import DGCNN_LiDAR


# ==================== КОНФИГУРАЦИЯ ====================

class Config:
    """Конфигурация для предсказания"""
    
    # Пути
    CHECKPOINT_PATH = None  # Будет установлен из аргументов
    UNLABELED_DIR = Path('datasets/unlabeled')
    PREDICTED_DIR = Path('datasets/predicted')
    VISUALIZATION_DIR = Path('datasets/predicted/visualizations')
    
    # Параметры обработки
    NUM_POINTS = 4096
    BLOCK_SIZE = 50.0
    STRIDE = 25.0  # Перекрытие для сглаживания
    BATCH_SIZE = 16
    USE_FEATURES = True
    FEATURE_DIM = 3
    
    # Voting для перекрывающихся блоков
    USE_VOTING = True  # Усреднение предсказаний в зонах перекрытия
    
    # Визуализация
    VISUALIZE = True
    VIZ_POINT_SIZE = 1
    VIZ_DPI = 150
    VIZ_SAMPLE_POINTS = 50000  # Количество точек для визуализации (для ускорения)
    
    # Маппинг классов
    CLASS_NAMES = {
        0: 'Unclassified',
        1: 'Ground',
        2: 'Vegetation',
        3: 'Building'
    }
    
    CLASS_COLORS = {
        0: [128, 128, 128],  # Серый
        1: [139, 69, 19],     # Коричневый (земля)
        2: [34, 139, 34],     # Зеленый (растительность)
        3: [255, 0, 0]        # Красный (здания)
    }
    
    # Обратный маппинг: 0,1,2,3 -> 1,2,5,6 (для LAS файла)
    CLASS_REVERSE_MAPPING = {0: 1, 1: 2, 2: 5, 3: 6}


# ==================== ЗАГРУЗКА МОДЕЛИ ====================

def load_model(checkpoint_path, device='cuda'):
    """
    Загрузка обученной модели из чекпоинта
    
    Args:
        checkpoint_path: путь к .pth файлу
        device: устройство для инференса
    
    Returns:
        model: загруженная модель в eval режиме
        config: конфигурация обучения
    """
    print(f"\n📦 Загрузка модели из: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # ========== ИСПРАВЛЕНИЕ: PyTorch 2.6 compatibility ==========
    # Загрузка чекпоинта с weights_only=False (безопасно для наших моделей)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Для старых версий PyTorch (< 2.6) где нет параметра weights_only
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Конфигурация
    train_config = checkpoint.get('config', {})
    
    # Создание модели
    model = DGCNN_LiDAR(
        num_classes=train_config.get('num_classes', 4),
        k=train_config.get('k_neighbors', 20),
        use_features=train_config.get('use_features', True),
        feature_dim=train_config.get('feature_dim', 3),
        dropout=0.0  # Dropout выключен для inference
    )
    
    # Загрузка весов
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Информация о модели
    print(f"   ✅ Модель загружена успешно!")
    print(f"   📊 Эпоха: {checkpoint.get('epoch', 'unknown')}")
    
    # Безопасное извлечение метрик
    best_val_acc = checkpoint.get('best_val_acc', None)
    best_val_miou = checkpoint.get('best_val_miou', None)
    
    if best_val_acc is not None:
        print(f"   🎯 Val Accuracy: {best_val_acc:.2f}%")
    if best_val_miou is not None:
        print(f"   🔷 Val mIoU: {best_val_miou:.2f}%")
    
    print(f"   🔧 Классов: {train_config.get('num_classes', 4)}")
    print(f"   📐 K соседей: {train_config.get('k_neighbors', 20)}")
    
    return model, train_config

# ==================== ОБРАБОТКА LAS ФАЙЛА ====================

class LASPredictor:
    """Класс для предсказания классов в LAS файлах"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.class_mapping = {1: 0, 2: 1, 5: 2, 6: 3}  # Для признаков, если есть разметка
    
    def load_las(self, las_file):
        """Загрузка LAS файла"""
        print(f"\n📂 Загрузка файла: {las_file}")
        las = laspy.read(las_file)
        
        # Координаты
        xyz = np.vstack([
            np.array(las.x, dtype=np.float32),
            np.array(las.y, dtype=np.float32),
            np.array(las.z, dtype=np.float32)
        ]).T
        
        print(f"   • Точек: {len(xyz):,}")
        print(f"   • X: {xyz[:, 0].min():.2f} → {xyz[:, 0].max():.2f}")
        print(f"   • Y: {xyz[:, 1].min():.2f} → {xyz[:, 1].max():.2f}")
        print(f"   • Z: {xyz[:, 2].min():.2f} → {xyz[:, 2].max():.2f}")
        
        # Дополнительные признаки
        features = {}
        
        if hasattr(las, 'intensity'):
            features['intensity'] = np.array(las.intensity, dtype=np.float32)
            print(f"   • Intensity: {features['intensity'].min():.0f} → {features['intensity'].max():.0f}")
        
        if hasattr(las, 'return_number') or hasattr(las, 'return_num'):
            return_num = las.return_number if hasattr(las, 'return_number') else las.return_num
            features['return_number'] = np.array(return_num, dtype=np.float32)
        
        if hasattr(las, 'number_of_returns') or hasattr(las, 'num_returns'):
            num_returns = las.number_of_returns if hasattr(las, 'number_of_returns') else las.num_returns
            features['number_of_returns'] = np.array(num_returns, dtype=np.float32)
        
        return las, xyz, features
    
    def create_blocks(self, xyz):
        """Создание блоков для обработки"""
        print(f"\n🔨 Создание блоков...")
        
        x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
        x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
        
        # Нормализация координат
        xyz_normalized = xyz.copy()
        xyz_normalized[:, 0] -= x_min
        xyz_normalized[:, 1] -= y_min
        
        blocks = []
        x_start = 0
        
        while x_start < (x_max - x_min):
            y_start = 0
            while y_start < (y_max - y_min):
                # Маска точек в блоке
                mask = (
                    (xyz_normalized[:, 0] >= x_start) &
                    (xyz_normalized[:, 0] < x_start + self.config.BLOCK_SIZE) &
                    (xyz_normalized[:, 1] >= y_start) &
                    (xyz_normalized[:, 1] < y_start + self.config.BLOCK_SIZE)
                )
                
                indices = np.where(mask)[0]
                
                if len(indices) >= 10:  # Минимум 10 точек
                    blocks.append({
                        'indices': indices,
                        'x_start': x_start,
                        'y_start': y_start,
                        'center_x': x_start + self.config.BLOCK_SIZE / 2,
                        'center_y': y_start + self.config.BLOCK_SIZE / 2
                    })
                
                y_start += self.config.STRIDE
            x_start += self.config.STRIDE
        
        print(f"   • Создано блоков: {len(blocks)}")
        
        return blocks, xyz_normalized, (x_min, y_min)
    
    def prepare_block(self, xyz_norm, features, indices):
        """Подготовка блока для модели"""
        block_xyz = xyz_norm[indices].copy()
        
        # Центрирование
        centroid = block_xyz[:, :2].mean(axis=0)
        block_xyz[:, 0] -= centroid[0]
        block_xyz[:, 1] -= centroid[1]
        
        # Сэмплирование
        if len(block_xyz) >= self.config.NUM_POINTS:
            choice = np.random.choice(len(block_xyz), self.config.NUM_POINTS, replace=False)
        else:
            choice = np.random.choice(len(block_xyz), self.config.NUM_POINTS, replace=True)
        
        block_xyz = block_xyz[choice]
        selected_indices = indices[choice]
        
        # Добавление признаков
        if self.config.USE_FEATURES:
            feature_list = []
            
            if 'intensity' in features:
                intensity = features['intensity'][selected_indices] / 255.0
                feature_list.append(intensity.reshape(-1, 1))
            
            if 'return_number' in features:
                return_num = features['return_number'][selected_indices]
                feature_list.append(return_num.reshape(-1, 1))
            
            if 'number_of_returns' in features:
                num_returns = features['number_of_returns'][selected_indices]
                feature_list.append(num_returns.reshape(-1, 1))
            
            if feature_list:
                feats = np.concatenate(feature_list, axis=1)
                block_xyz = np.concatenate([block_xyz, feats], axis=1)
        
        # Нормализация координат
        centroid_xyz = block_xyz[:, :3].mean(axis=0)
        block_xyz[:, :3] -= centroid_xyz
        max_dist = np.max(np.sqrt(np.sum(block_xyz[:, :3]**2, axis=1)))
        if max_dist > 0:
            block_xyz[:, :3] /= max_dist
        
        return torch.FloatTensor(block_xyz), selected_indices
    
    @torch.no_grad()
    def predict(self, las_file, output_file=None):
        """
        Предсказание классов для всего LAS файла
        
        Args:
            las_file: путь к входному LAS файлу
            output_file: путь для сохранения (если None, автоматически)
        
        Returns:
            output_file: путь к сохраненному файлу
            predictions: массив предсказанных классов
        """
        # Загрузка данных
        las_original, xyz, features = self.load_las(las_file)
        
        # Создание блоков
        blocks, xyz_normalized, (x_min, y_min) = self.create_blocks(xyz)
        
        # Инициализация массивов для предсказаний
        if self.config.USE_VOTING:
            # Для voting храним сумму вероятностей и количество голосов
            predictions_sum = np.zeros((len(xyz), 4), dtype=np.float32)
            predictions_count = np.zeros(len(xyz), dtype=np.int32)
        else:
            predictions = np.zeros(len(xyz), dtype=np.int32)
        
        # Обработка блоков батчами
        print(f"\n🔮 Предсказание классов...")
        
        num_batches = (len(blocks) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for batch_idx in tqdm(range(num_batches), desc="Обработка батчей"):
            start_idx = batch_idx * self.config.BATCH_SIZE
            end_idx = min(start_idx + self.config.BATCH_SIZE, len(blocks))
            batch_blocks = blocks[start_idx:end_idx]
            
            batch_data = []
            batch_indices = []
            
            for block in batch_blocks:
                block_tensor, selected_indices = self.prepare_block(
                    xyz_normalized, features, block['indices']
                )
                batch_data.append(block_tensor)
                batch_indices.append(selected_indices)
            
            # Stack в батч
            batch_tensor = torch.stack(batch_data).to(self.device)
            
            # Предсказание
            outputs = self.model(batch_tensor)  # (B, N, num_classes)
            
            if self.config.USE_VOTING:
                # Softmax для вероятностей
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()  # (B, N, num_classes)
                
                for i, indices in enumerate(batch_indices):
                    predictions_sum[indices] += probs[i]
                    predictions_count[indices] += 1
            else:
                # Просто argmax
                preds = outputs.argmax(dim=-1).cpu().numpy()  # (B, N)
                
                for i, indices in enumerate(batch_indices):
                    predictions[indices] = preds[i]
        
        # Финальные предсказания
        if self.config.USE_VOTING:
            # Усреднение вероятностей
            mask = predictions_count > 0
            predictions_sum[mask] /= predictions_count[mask, np.newaxis]
            predictions = predictions_sum.argmax(axis=1)
            
            # Для точек без предсказаний (не должно быть, но на всякий случай)
            predictions[~mask] = 0
        
        print(f"\n✅ Предсказание завершено!")
        
        # Статистика предсказаний
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\n📊 Распределение предсказанных классов:")
        total = len(predictions)
        for cls, count in zip(unique, counts):
            percent = 100.0 * count / total
            class_name = self.config.CLASS_NAMES.get(int(cls), f'Class {cls}')
            print(f"   {class_name}: {count:,} точек ({percent:.2f}%)")
        
        # Сохранение результатов
        if output_file is None:
            output_file = self.config.PREDICTED_DIR / Path(las_file).name
        
        output_file = self.save_predictions(las_original, predictions, output_file)
        
        return output_file, predictions, xyz
    
    def save_predictions(self, las_original, predictions, output_file):
        """Сохранение LAS файла с предсказанными классами"""
        print(f"\n💾 Сохранение результатов...")
        
        # Создание выходной директории
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Создание нового LAS файла
        las_output = laspy.LasData(las_original.header)
        
        # Копирование всех точек
        las_output.x = las_original.x
        las_output.y = las_original.y
        las_output.z = las_original.z
        
        # Копирование других атрибутов
        if hasattr(las_original, 'intensity'):
            las_output.intensity = las_original.intensity
        if hasattr(las_original, 'return_number'):
            las_output.return_number = las_original.return_number
        elif hasattr(las_original, 'return_num'):
            las_output.return_num = las_original.return_num
        if hasattr(las_original, 'number_of_returns'):
            las_output.number_of_returns = las_original.number_of_returns
        elif hasattr(las_original, 'num_returns'):
            las_output.num_returns = las_original.num_returns
        
        # Установка предсказанных классов
        # Обратный маппинг: 0,1,2,3 -> 1,2,5,6
        predictions_remapped = np.array([
            self.config.CLASS_REVERSE_MAPPING[p] for p in predictions
        ], dtype=np.uint8)
        
        las_output.classification = predictions_remapped
        
        # Сохранение
        las_output.write(str(output_file))
        
        print(f"   ✅ Файл сохранен: {output_file}")
        print(f"   📦 Размер: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_file


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize_predictions(xyz, predictions, config, output_path=None, title="Predicted Classes"):
    """
    Визуализация облака точек с предсказанными классами
    
    Args:
        xyz: координаты точек (N, 3)
        predictions: предсказанные классы (N,)
        config: конфигурация
        output_path: путь для сохранения
        title: заголовок графика
    """
    print(f"\n🎨 Создание визуализации...")
    
    # Сэмплирование для ускорения визуализации
    if len(xyz) > config.VIZ_SAMPLE_POINTS:
        indices = np.random.choice(len(xyz), config.VIZ_SAMPLE_POINTS, replace=False)
        xyz_viz = xyz[indices]
        pred_viz = predictions[indices]
    else:
        xyz_viz = xyz
        pred_viz = predictions
    
    # Цвета для каждой точки
    colors = np.array([config.CLASS_COLORS[p] for p in pred_viz]) / 255.0
    
    # Создание 3D графика
    fig = plt.figure(figsize=(20, 15))
    
    # 3D вид сверху
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(xyz_viz[:, 0], xyz_viz[:, 1], xyz_viz[:, 2], 
                c=colors, s=config.VIZ_POINT_SIZE, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View (Top)')
    ax1.view_init(elev=90, azim=-90)
    
    # 3D вид сбоку
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(xyz_viz[:, 0], xyz_viz[:, 1], xyz_viz[:, 2], 
                c=colors, s=config.VIZ_POINT_SIZE, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D View (Side)')
    ax2.view_init(elev=10, azim=-45)
    
    # 2D вид сверху (XY)
    ax3 = fig.add_subplot(223)
    ax3.scatter(xyz_viz[:, 0], xyz_viz[:, 1], 
                c=colors, s=config.VIZ_POINT_SIZE, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('2D View (Top - XY)')
    ax3.set_aspect('equal')
    
    # Легенда с классами
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # Статистика по классам
    unique, counts = np.unique(pred_viz, return_counts=True)
    total = len(pred_viz)
    
    legend_text = f"{title}\n\n"
    legend_text += f"Total points visualized: {len(xyz_viz):,}\n\n"
    legend_text += "Class Distribution:\n"
    legend_text += "-" * 40 + "\n"
    
    y_pos = 0.9
    for cls, count in zip(unique, counts):
        percent = 100.0 * count / total
        class_name = config.CLASS_NAMES.get(int(cls), f'Class {cls}')
        color = np.array(config.CLASS_COLORS[cls]) / 255.0
        
        # Цветной квадратик
        ax4.add_patch(plt.Rectangle((0.1, y_pos - 0.02), 0.05, 0.05, 
                                     facecolor=color, edgecolor='black'))
        
        # Текст
        ax4.text(0.2, y_pos, f"{class_name}: {count:,} ({percent:.1f}%)", 
                fontsize=12, verticalalignment='center')
        
        y_pos -= 0.1
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Сохранение
    if output_path is None:
        output_path = config.VISUALIZATION_DIR / f"prediction_{Path(title).stem}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Визуализация сохранена: {output_path}")
    
    return output_path


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='DGCNN LiDAR Prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input LAS file (if None, process all in unlabeled/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output LAS file path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Конфигурация
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.VISUALIZE = not args.no_visualize
    
    # Устройство
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"{'🎯 DGCNN LIDAR PREDICTION':^80}")
    print(f"{'='*80}")
    print(f"\n🖥️  Device: {device}")
    
    # Загрузка модели
    model, train_config = load_model(args.checkpoint, device)
    
    # Создание предиктора
    predictor = LASPredictor(model, config, device)
    
    # Определение входных файлов
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_files = list(config.UNLABELED_DIR.glob('*.las'))
        if not input_files:
            print(f"\n❌ Не найдено LAS файлов в: {config.UNLABELED_DIR}")
            return
    
    print(f"\n📁 Найдено файлов для обработки: {len(input_files)}")
    
    # Обработка файлов
    for las_file in input_files:
        print(f"\n{'='*80}")
        print(f"📄 Обработка: {las_file.name}")
        print(f"{'='*80}")
        
        # Предсказание
        output_file, predictions, xyz = predictor.predict(
            las_file,
            output_file=args.output
        )
        
        # Визуализация
        if config.VISUALIZE:
            viz_path = config.VISUALIZATION_DIR / f"{las_file.stem}_predicted.png"
            visualize_predictions(
                xyz, predictions, config,
                output_path=viz_path,
                title=f"Predictions: {las_file.name}"
            )
    
    print(f"\n{'='*80}")
    print(f"✅ ВСЕ ФАЙЛЫ ОБРАБОТАНЫ")
    print(f"{'='*80}")
    print(f"\n📁 Результаты:")
    print(f"   • Predicted LAS: {config.PREDICTED_DIR}")
    if config.VISUALIZE:
        print(f"   • Visualizations: {config.VISUALIZATION_DIR}")


if __name__ == '__main__':
    main()