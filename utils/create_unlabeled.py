import os
import laspy
import numpy as np
from pathlib import Path

def detect_file_type(file_path):
    """Определение типа файла"""
    ext = Path(file_path).suffix.lower()
    if ext in ['.las', '.laz']:
        return 'las'
    elif ext in ['.xyz', '.txt']:
        return 'xyz'
    else:
        return 'las'  # по умолчанию

def list_data_files(folder):
    """Список всех поддерживаемых файлов"""
    if not os.path.exists(folder):
        print(f"❌ Папка {folder} не существует.")
        return []
    
    extensions = ['.las', '.laz', '.xyz', '.txt']
    files = []
    for ext in extensions:
        files.extend([f for f in os.listdir(folder) if f.lower().endswith(ext)])
    return files

def create_unlabeled_xyz(input_path, output_path):
    """Создание немаркированной версии XYZ файла"""
    print(f"Очистка {input_path} → {output_path}...")
    
    try:
        # Чтение XYZ
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 4:
                    # Заменяем класс на 0 (Unclassified)
                    x, y, z = parts[1], parts[2], parts[3]
                    data.append(f"0 {x} {y} {z}\n")
        
        # Сохранение
        with open(output_path, 'w') as f:
            f.writelines(data)
        
        print(f"✅ Создан немаркированный XYZ: {output_path}")
        print(f"   Обработано строк: {len(data)}")
        
    except Exception as e:
        print(f"❌ Ошибка обработки XYZ: {e}")

def create_unlabeled_las(input_path, output_path):
    """Создание немаркированной версии LAS файла (существующий код)"""
    print(f"Очистка {input_path} → {output_path}...")
    
    # Чтение исходного файла
    with laspy.open(input_path) as f:
        las = f.read()

    # Определяем совместимую версию
    target_version = "1.2"
    point_format = las.header.point_format

    # Создаём новый LAS
    new_las = laspy.create(point_format=point_format, file_version=target_version)
    
    # Копируем координаты
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z

    # Копируем другие поля, кроме классификации
    for dim_name in las.point_format.dimension_names:
        if 'class' in dim_name.lower():
            continue
        if hasattr(las, dim_name):
            try:
                setattr(new_las, dim_name, getattr(las, dim_name))
            except Exception as e:
                print(f"⚠️  Не удалось скопировать поле '{dim_name}': {e}")

    new_las.write(output_path)
    print(f"✅ Сохранено: {output_path}")

def main():
    raw_dir = 'datasets/raw'
    out_dir = 'datasets/unlabeled'
    
    files = list_data_files(raw_dir)
    if not files:
        print("Нет поддерживаемых файлов в datasets/raw/")
        return

    print("Доступные файлы:")
    for i, f in enumerate(files, 1):
        file_type = detect_file_type(f)
        print(f"{i}. {f} [{file_type.upper()}]")
    
    try:
        idx = int(input("Выберите номер файла: ")) - 1
        if idx < 0 or idx >= len(files):
            print("Неверный номер.")
            return
    except ValueError:
        print("Введите число.")
        return

    input_path = os.path.join(raw_dir, files[idx])
    output_path = os.path.join(out_dir, files[idx])
    os.makedirs(out_dir, exist_ok=True)

    file_type = detect_file_type(input_path)
    
    if file_type == 'xyz':
        create_unlabeled_xyz(input_path, output_path)
    else:
        create_unlabeled_las(input_path, output_path)

if __name__ == '__main__':
    main()