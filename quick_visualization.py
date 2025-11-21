"""
Быстрая визуализация облака точек для проверки работы
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import laspy
import sys
import os
from tqdm import tqdm

# Конфигурация визуализации
CLASS_NAMES = {
    0: 'Unclassified',
    1: 'Ground', 
    2: 'Vegetation',
    3: 'Building'
}

CLASS_COLORS = {
    0: [128, 128, 128],  # Серый
    1: [139, 69, 19],     # Коричневый
    2: [34, 139, 34],     # Зеленый
    3: [255, 0, 0]        # Красный
}

def load_sample_data(file_path, max_points=50000):
    """Загрузка небольшой выборки точек для визуализации"""
    print(f"📂 Загрузка данных из: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.las', '.laz']:
        # Загрузка LAS файла
        try:
            las = laspy.read(file_path)
            xyz = np.vstack([las.x, las.y, las.z]).T
            
            # Получаем классы если есть
            if hasattr(las, 'classification'):
                labels = np.array(las.classification)
                print(f"   • Найдены классы: {np.unique(labels)}")
            else:
                labels = np.zeros(len(xyz), dtype=int)
                print("   • Классы не найдены, используем Unclassified")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки LAS: {e}")
            return None, None
            
    elif file_ext in ['.xyz', '.txt']:
        # Загрузка XYZ файла
        data = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_points:  # Ограничиваем количество точек
                    break
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        class_id = int(float(parts[0]))
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        data.append([class_id, x, y, z])
                    except:
                        continue
        
        if not data:
            print("❌ Не удалось загрузить данные из XYZ файла")
            return None, None
            
        data = np.array(data)
        labels = data[:, 0].astype(int)
        xyz = data[:, 1:4]
        
    else:
        print("❌ Неподдерживаемый формат файла")
        return None, None
    
    # Сэмплируем точки если их слишком много
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        labels = labels[indices]
        print(f"   • Загружено {max_points} точек (случайная выборка)")
    else:
        print(f"   • Загружено {len(xyz)} точек")
    
    print(f"   • X: {xyz[:, 0].min():.2f} → {xyz[:, 0].max():.2f}")
    print(f"   • Y: {xyz[:, 1].min():.2f} → {xyz[:, 1].max():.2f}") 
    print(f"   • Z: {xyz[:, 2].min():.2f} → {xyz[:, 2].max():.2f}")
    
    return xyz, labels

def visualize_point_cloud(xyz, labels, title="Облако точек", output_path=None):
    """Визуализация облака точек с классами"""
    print(f"\n🎨 Создание визуализации...")
    
    # Цвета для каждого класса
    colors = np.array([CLASS_COLORS.get(l, [128, 128, 128]) for l in labels]) / 255.0
    
    # Создание фигуры
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D вид сверху
    ax1 = fig.add_subplot(221, projection='3d')
    scatter1 = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                          c=colors, s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.set_title('3D вид сверху')
    ax1.view_init(elev=90, azim=-90)
    
    # 2. 3D вид сбоку
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                          c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D вид сбоку')
    ax2.view_init(elev=15, azim=-45)
    
    # 3. 2D вид сверху (XY)
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(xyz[:, 0], xyz[:, 1], 
                          c=colors, s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('2D вид сверху (XY)')
    ax3.set_aspect('equal')
    
    # 4. Статистика и легенда
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # Статистика по классам
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_points = len(labels)
    
    # Заголовок
    ax4.text(0.1, 0.95, f"{title}", fontsize=16, fontweight='bold', 
             transform=ax4.transAxes)
    ax4.text(0.1, 0.90, f"Всего точек: {total_points:,}", fontsize=12,
             transform=ax4.transAxes)
    
    # Распределение классов
    y_pos = 0.80
    ax4.text(0.1, y_pos, "Распределение классов:", fontsize=12, fontweight='bold',
             transform=ax4.transAxes)
    y_pos -= 0.05
    
    for i, (cls, count) in enumerate(zip(unique_labels, counts)):
        percent = 100.0 * count / total_points
        class_name = CLASS_NAMES.get(cls, f'Class {cls}')
        color = np.array(CLASS_COLORS.get(cls, [128, 128, 128])) / 255.0
        
        # Цветной маркер
        ax4.plot(0.1, y_pos, 's', markersize=10, color=color, transform=ax4.transAxes)
        
        # Текст
        ax4.text(0.15, y_pos, f"{class_name}: {count:,} ({percent:.1f}%)", 
                fontsize=10, transform=ax4.transAxes, verticalalignment='center')
        
        y_pos -= 0.05
    
    plt.tight_layout()
    
    # Сохранение
    if output_path is None:
        output_path = Path('datasets/predicted/visualizations/quick_test.png')
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Визуализация сохранена: {output_path}")
    return output_path

def quick_analysis(xyz, labels):
    """Быстрый анализ данных"""
    print(f"\n📊 Быстрый анализ данных:")
    print(f"   • Всего точек: {len(xyz):,}")
    print(f"   • Размерность: {xyz.shape}")
    
    # Статистика по координатам
    print(f"   • Диапазон X: {xyz[:, 0].min():.2f} - {xyz[:, 0].max():.2f}")
    print(f"   • Диапазон Y: {xyz[:, 1].min():.2f} - {xyz[:, 1].max():.2f}") 
    print(f"   • Диапазон Z: {xyz[:, 2].min():.2f} - {xyz[:, 2].max():.2f}")
    
    # Статистика по классам
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   • Уникальные классы: {unique}")
    print(f"   • Распределение классов:")
    for cls, count in zip(unique, counts):
        percent = 100.0 * count / len(labels)
        class_name = CLASS_NAMES.get(cls, f'Class {cls}')
        print(f"      {class_name}: {count:,} точек ({percent:.2f}%)")

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Быстрая визуализация облака точек')
    parser.add_argument('--input', type=str, required=True, 
                       help='Путь к файлу (LAS или XYZ)')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения визуализации')
    parser.add_argument('--max_points', type=int, default=50000,
                       help='Максимальное количество точек для визуализации')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 БЫСТРАЯ ВИЗУАЛИЗАЦИЯ ОБЛАКА ТОЧЕК")
    print("=" * 80)
    
    # Проверка файла
    if not os.path.exists(args.input):
        print(f"❌ Файл не найден: {args.input}")
        return
    
    # Загрузка данных
    xyz, labels = load_sample_data(args.input, args.max_points)
    
    if xyz is None:
        print("❌ Не удалось загрузить данные")
        return
    
    # Быстрый анализ
    quick_analysis(xyz, labels)
    
    # Визуализация
    filename = Path(args.input).stem
    output_path = args.output or f'datasets/predicted/visualizations/quick_{filename}.png'
    
    visualize_point_cloud(xyz, labels, 
                         title=f"Быстрая визуализация: {filename}",
                         output_path=output_path)
    
    print(f"\n✅ Готово! Визуализация сохранена: {output_path}")

if __name__ == '__main__':
    main()