"""
Применение обученной DGCNN модели к неразмеченным LiDAR данным
Сохранение результатов с предсказанными классами и визуализация
Поддержка форматов LAS и XYZ с конфигурационными файлами
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modelDGCNN import DGCNN_LiDAR
from utils.dataset_config import DatasetConfig, auto_detect_config

# ==================== КОНФИГУРАЦИЯ ====================

class PredictConfig:
    """Конфигурация для предсказания"""
    
    # Пути
    UNLABELED_DIR = Path('datasets/unlabeled')
    PREDICTED_DIR = Path('datasets/predicted')
    VISUALIZATION_DIR = Path('datasets/predicted/visualizations')
    
    # Параметры обработки
    NUM_POINTS = 4096
    BLOCK_SIZE = 50.0
    STRIDE = 25.0
    BATCH_SIZE = 16
    
    # Voting для перекрывающихся блоков
    USE_VOTING = True
    
    # Визуализация
    VISUALIZE = True
    VIZ_POINT_SIZE = 1
    VIZ_DPI = 150
    VIZ_SAMPLE_POINTS = 50000

# ==================== ЗАГРУЗКА МОДЕЛИ ====================

def load_model(checkpoint_path, device='cuda'):
    """
    Загрузка обученной модели из чекпоинта
    """
    print(f"\n📦 Загрузка модели из: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Конфигурация обучения
    train_config = checkpoint.get('config', {})
    
    # Используем количество классов из конфига обучения
    num_classes = train_config.get('num_classes', 4)
    feature_dim = train_config.get('feature_dim', 3)
    
    print(f"   🔧 Классов: {num_classes}")
    print(f"   🔧 Используются координаты + признаки ({3 + feature_dim} каналов)")
    
    # Создание модели
    model = DGCNN_LiDAR(
        num_classes=num_classes,
        k=train_config.get('k_neighbors', 20),
        use_features=True,
        feature_dim=feature_dim,
        dropout=0.0
    )
    
    # Загрузка весов
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Модель загружена успешно!")
    print(f"   📊 Эпоха: {checkpoint.get('epoch', 'unknown')}")
    
    best_val_acc = checkpoint.get('best_val_acc', None)
    best_val_miou = checkpoint.get('best_val_miou', None)
    
    if best_val_acc is not None:
        print(f"   🎯 Val Accuracy: {best_val_acc:.2f}%")
    if best_val_miou is not None:
        print(f"   🔷 Val mIoU: {best_val_miou:.2f}%")
    
    print(f"   📐 K соседей: {train_config.get('k_neighbors', 20)}")
    
    return model, train_config, num_classes

# ==================== ОБРАБОТКА ФАЙЛОВ ====================

class LidarPredictor:
    """Универсальный класс для предсказания классов в LAS и XYZ файлах"""
    
    def __init__(self, model, config, dataset_config, device='cuda'):
        self.model = model
        self.config = config
        self.dataset_config = dataset_config
        self.device = device
        self.num_classes = dataset_config.num_classes
    
    def detect_file_type(self, file_path):
        """Определение типа файла"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.las', '.laz']:
            return 'las'
        elif file_ext in ['.xyz', '.txt']:
            return 'xyz'
        else:
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    parts = first_line.replace(',', ' ').split()
                    if len(parts) >= 4 and all(self._is_float(x) for x in parts):
                        return 'xyz'
            except:
                pass
            return 'las'
    
    def _is_float(self, x):
        """Проверка, можно ли преобразовать в float"""
        try:
            float(x)
            return True
        except ValueError:
            return False
    
    def load_xyz(self, xyz_file):
        """Загрузка XYZ файла с использованием конфигурации датасета"""
        print(f"\n📂 Загрузка XYZ файла: {xyz_file}")
        
        data = []
        unique_classes = set()
        
        with open(xyz_file, 'r') as f:
            for line in tqdm(f, desc="   Чтение строк"):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.replace(',', ' ').split()
                if len(parts) >= 4:
                    try:
                        class_id = int(float(parts[0]))
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        data.append([x, y, z, class_id])
                        unique_classes.add(class_id)
                    except ValueError:
                        continue
        
        if not data:
            raise ValueError("XYZ файл не содержит валидных данных")
        
        data = np.array(data, dtype=np.float32)
        xyz = data[:, :3]
        original_classes = data[:, 3].astype(np.int32)
        
        # Применяем маппинг классов из конфигурации
        mapped_classes = self.dataset_config.map_labels(original_classes)
        
        print(f"   • Точек: {len(xyz):,}")
        print(f"   • X: {xyz[:, 0].min():.2f} → {xyz[:, 0].max():.2f}")
        print(f"   • Y: {xyz[:, 1].min():.2f} → {xyz[:, 1].max():.2f}")
        print(f"   • Z: {xyz[:, 2].min():.2f} → {xyz[:, 2].max():.2f}")
        print(f"   • Исходные классы: {sorted(unique_classes)}")
        print(f"   • После маппинга: {np.unique(mapped_classes)}")
        
        # XYZ файлы не содержат дополнительных признаков
        features = {}
        
        return None, xyz, mapped_classes, features
    
    def load_las(self, las_file):
        """Загрузка LAS файла с использованием конфигурации датасета"""
        print(f"\n📂 Загрузка LAS файла: {las_file}")
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
        
        # Получаем и маппим классы из LAS
        if hasattr(las, 'classification'):
            original_classes = np.array(las.classification)
            mapped_classes = self.dataset_config.map_labels(original_classes)
            print(f"   • Исходные классы LAS: {np.unique(original_classes)}")
            print(f"   • После маппинга: {np.unique(mapped_classes)}")
        else:
            mapped_classes = np.zeros(len(xyz), dtype=np.int32)
            print("   • Классы не найдены, используем Other")
        
        # Дополнительные признаки
        features = {}
        
        if hasattr(las, 'intensity') and self.dataset_config.use_features:
            features['intensity'] = np.array(las.intensity, dtype=np.float32)
            print(f"   • Intensity: {features['intensity'].min():.0f} → {features['intensity'].max():.0f}")
        
        if (hasattr(las, 'return_number') or hasattr(las, 'return_num')) and self.dataset_config.use_features:
            return_num = las.return_number if hasattr(las, 'return_number') else las.return_num
            features['return_number'] = np.array(return_num, dtype=np.float32)
        
        if (hasattr(las, 'number_of_returns') or hasattr(las, 'num_returns')) and self.dataset_config.use_features:
            num_returns = las.number_of_returns if hasattr(las, 'number_of_returns') else las.num_returns
            features['number_of_returns'] = np.array(num_returns, dtype=np.float32)
        
        return las, xyz, mapped_classes, features
    
    def load_data(self, file_path):
        """Загрузка данных в зависимости от типа файла"""
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'xyz':
            return self.load_xyz(file_path)
        else:
            return self.load_las(file_path)
    
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
                mask = (
                    (xyz_normalized[:, 0] >= x_start) &
                    (xyz_normalized[:, 0] < x_start + self.config.BLOCK_SIZE) &
                    (xyz_normalized[:, 1] >= y_start) &
                    (xyz_normalized[:, 1] < y_start + self.config.BLOCK_SIZE)
                )
                
                indices = np.where(mask)[0]
                
                if len(indices) >= 10:
                    blocks.append({
                        'indices': indices,
                        'x_start': x_start,
                        'y_start': y_start,
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
        feature_list = []
        
        if not features and self.dataset_config.use_features:
            # Для XYZ файлов добавляем нулевые признаки
            zeros = np.zeros((len(block_xyz), 3), dtype=np.float32)
            feature_list.append(zeros)
        elif self.dataset_config.use_features:
            # Для LAS файлов используем реальные признаки
            if 'intensity' in features:
                intensity = features['intensity'][selected_indices] / 255.0
                feature_list.append(intensity.reshape(-1, 1))
            else:
                feature_list.append(np.zeros((len(block_xyz), 1), dtype=np.float32))
            
            if 'return_number' in features:
                return_num = features['return_number'][selected_indices]
                feature_list.append(return_num.reshape(-1, 1))
            else:
                feature_list.append(np.zeros((len(block_xyz), 1), dtype=np.float32))
            
            if 'number_of_returns' in features:
                num_returns = features['number_of_returns'][selected_indices]
                feature_list.append(num_returns.reshape(-1, 1))
            else:
                feature_list.append(np.zeros((len(block_xyz), 1), dtype=np.float32))
        
        # Объединяем признаки
        if feature_list and self.dataset_config.use_features:
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
    def predict(self, input_file, output_file=None):
        """
        Предсказание классов для всего файла
        """
        file_type = self.detect_file_type(input_file)
        
        # Загрузка данных
        original_data, xyz, original_classes, features = self.load_data(input_file)
        
        print(f"   🔧 Тип файла: {file_type.upper()}")
        print(f"   🔧 Модель поддерживает классов: {self.num_classes}")
        print(f"   🔧 Конфигурация: {self.dataset_config.name}")
        
        # Создание блоков
        blocks, xyz_normalized, (x_min, y_min) = self.create_blocks(xyz)
        
        # Инициализация массивов для предсказаний с правильным количеством классов
        if self.config.USE_VOTING:
            predictions_sum = np.zeros((len(xyz), self.num_classes), dtype=np.float32)
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
            
            batch_tensor = torch.stack(batch_data).to(self.device)
            outputs = self.model(batch_tensor)
            
            if self.config.USE_VOTING:
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                for i, indices in enumerate(batch_indices):
                    predictions_sum[indices] += probs[i]
                    predictions_count[indices] += 1
            else:
                preds = outputs.argmax(dim=-1).cpu().numpy()
                for i, indices in enumerate(batch_indices):
                    predictions[indices] = preds[i]
        
        # Финальные предсказания
        if self.config.USE_VOTING:
            mask = predictions_count > 0
            predictions_sum[mask] /= predictions_count[mask, np.newaxis]
            predictions = predictions_sum.argmax(axis=1)
            predictions[~mask] = 0
        
        print(f"\n✅ Предсказание завершено!")
        
        # Статистика предсказаний
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\n📊 Распределение предсказанных классов:")
        total = len(predictions)
        for cls, count in zip(unique, counts):
            percent = 100.0 * count / total
            class_name = self.dataset_config.get_class_name(cls)
            print(f"   {class_name}: {count:,} точек ({percent:.2f}%)")
        
        # Сохранение результатов
        if output_file is None:
            output_file = self.config.PREDICTED_DIR / Path(input_file).name
        
        if file_type == 'las':
            output_file = self.save_las_predictions(original_data, predictions, output_file)
        else:
            output_file = self.save_xyz_predictions(xyz, predictions, output_file)
        
        return output_file, predictions, xyz
    
    def save_las_predictions(self, las_original, predictions, output_file):
        """Сохранение LAS файла с предсказанными классами"""
        print(f"\n💾 Сохранение результатов в LAS...")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            header = laspy.LasHeader(version="1.2", point_format=las_original.header.point_format.id)
            header.offsets = las_original.header.offsets
            header.scales = las_original.header.scales
            
            las_output = laspy.LasData(header)
            
        except Exception as e:
            print(f"   ⚠️  Ошибка создания заголовка: {e}")
            las_output = laspy.create(point_format=las_original.header.point_format.id, version="1.2")
        
        # Копирование всех точек
        las_output.x = las_original.x
        las_output.y = las_original.y
        las_output.z = las_original.z
        
        # Копирование других атрибутов
        if hasattr(las_original, 'intensity'):
            las_output.intensity = las_original.intensity
        
        # Установка предсказанных классов с обратным маппингом
        predictions_remapped = self.dataset_config.reverse_map_labels(predictions)
        
        las_output.classification = predictions_remapped
        
        # Сохранение
        try:
            las_output.write(str(output_file))
            print(f"   ✅ Файл сохранен: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"   ❌ Ошибка сохранения LAS: {e}")
            xyz_output = output_file.with_suffix('.xyz')
            print(f"   🔄 Сохранение в XYZ: {xyz_output}")
            return self.save_xyz_predictions(
                np.vstack([las_original.x, las_original.y, las_original.z]).T,
                predictions,
                xyz_output
            )
    
    def save_xyz_predictions(self, xyz, predictions, output_file):
        """Сохранение XYZ файла с предсказанными классами"""
        print(f"\n💾 Сохранение результатов в XYZ...")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for i in tqdm(range(len(xyz)), desc="   Запись точек"):
                f.write(f"{predictions[i]} {xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]}\n")
        
        print(f"   ✅ Файл сохранен: {output_file}")
        return output_file

# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize_predictions(xyz, predictions, dataset_config, output_path=None, title="Predicted Classes"):
    """
    Визуализация облака точек с предсказанными классами
    """
    print(f"\n🎨 Создание визуализации...")
    
    # Сэмплирование для ускорения визуализации
    sample_points = 50000
    if len(xyz) > sample_points:
        indices = np.random.choice(len(xyz), sample_points, replace=False)
        xyz_viz = xyz[indices]
        pred_viz = predictions[indices]
    else:
        xyz_viz = xyz
        pred_viz = predictions
    
    # Цвета для каждой точки
    colors = np.array([dataset_config.get_class_color(p) for p in pred_viz]) / 255.0
    
    # Создание 3D графика
    fig = plt.figure(figsize=(20, 15))
    
    # 3D вид сверху
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(xyz_viz[:, 0], xyz_viz[:, 1], xyz_viz[:, 2], 
                c=colors, s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View (Top)')
    ax1.view_init(elev=90, azim=-90)
    
    # 3D вид сбоку
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(xyz_viz[:, 0], xyz_viz[:, 1], xyz_viz[:, 2], 
                c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D View (Side)')
    ax2.view_init(elev=10, azim=-45)
    
    # 2D вид сверху (XY)
    ax3 = fig.add_subplot(223)
    ax3.scatter(xyz_viz[:, 0], xyz_viz[:, 1], 
                c=colors, s=1, alpha=0.6)
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
    
    y_pos = 0.9
    for cls, count in zip(unique, counts):
        percent = 100.0 * count / total
        class_name = dataset_config.get_class_name(cls)
        color = np.array(dataset_config.get_class_color(cls)) / 255.0
        
        # Цветной квадратик
        ax4.add_patch(plt.Rectangle((0.1, y_pos - 0.02), 0.05, 0.05, 
                                     facecolor=color, edgecolor='black'))
        
        # Текст
        ax4.text(0.2, y_pos, f"{class_name}: {count:,} ({percent:.1f}%)", 
                fontsize=10, verticalalignment='center')
        
        y_pos -= 0.05
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Сохранение
    if output_path is None:
        output_path = Path('datasets/predicted/visualizations') / f"prediction_{Path(title).stem}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Визуализация сохранена: {output_path}")
    
    return output_path

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='DGCNN LiDAR Prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input file (LAS or XYZ). If None, process all in unlabeled/')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Path to dataset config YAML file (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Конфигурация предсказания
    config = PredictConfig()
    config.BATCH_SIZE = args.batch_size
    config.VISUALIZE = not args.no_visualize
    
    # Устройство
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"{'🎯 DGCNN LIDAR PREDICTION':^80}")
    print(f"{'='*80}")
    print(f"\n🖥️  Device: {device}")
    
    # Загрузка модели
    model, train_config, num_classes = load_model(args.checkpoint, device)
    
    # Определение конфигурации датасета
    if args.dataset_config and os.path.exists(args.dataset_config):
        # Используем указанную конфигурацию
        dataset_config = DatasetConfig(args.dataset_config)
        print(f"📋 Используется конфигурация: {dataset_config.name}")
    elif args.input:
        # Автоопределение по имени файла
        dataset_config = auto_detect_config(args.input)
        if dataset_config:
            print(f"📋 Автоопределена конфигурация: {dataset_config.name}")
        else:
            # Используем конфигурацию из обучения
            print("📋 Используется конфигурация из обучения модели")
            # Создаем минимальную конфигурацию на основе обучения
            dataset_config = DatasetConfig('configs/datasets/neon_sample.yaml')  # Fallback
    else:
        # Используем конфигурацию из обучения
        print("📋 Используется конфигурация из обучения модели")
        dataset_config = DatasetConfig('configs/datasets/neon_sample.yaml')  # Fallback
    
    # Вывод информации о конфигурации
    dataset_config.print_info()
    
    # Создание предиктора
    predictor = LidarPredictor(model, config, dataset_config, device)
    
    # Определение входных файлов
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_files = []
        for ext in ['*.las', '*.laz', '*.xyz', '*.txt']:
            input_files.extend(list(config.UNLABELED_DIR.glob(ext)))
        
        if not input_files:
            print(f"\n❌ Не найдено файлов в: {config.UNLABELED_DIR}")
            print(f"   Поддерживаемые форматы: .las, .laz, .xyz, .txt")
            return
    
    print(f"\n📁 Найдено файлов для обработки: {len(input_files)}")
    
    # Обработка файлов
    for input_file in input_files:
        print(f"\n{'='*80}")
        print(f"📄 Обработка: {input_file.name}")
        print(f"{'='*80}")
        
        output_file, predictions, xyz = predictor.predict(
            input_file,
            output_file=args.output
        )
        
        if config.VISUALIZE:
            viz_path = config.VISUALIZATION_DIR / f"{input_file.stem}_predicted.png"
            visualize_predictions(
                xyz, predictions, dataset_config,
                output_path=viz_path,
                title=f"Predictions: {input_file.name}"
            )
    
    print(f"\n{'='*80}")
    print(f"✅ ВСЕ ФАЙЛЫ ОБРАБОТАНЫ")
    print(f"{'='*80}")
    print(f"\n📁 Результаты:")
    print(f"   • Predicted files: {config.PREDICTED_DIR}")
    if config.VISUALIZE:
        print(f"   • Visualizations: {config.VISUALIZATION_DIR}")

if __name__ == '__main__':
    main()