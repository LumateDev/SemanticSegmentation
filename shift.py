import os

# Пути к файлам
input_path = r"C:\Users\Alex\Desktop\SemanticSegmentation\datasets\raw\001.xyz"
output_path = r"C:\Users\Alex\Desktop\SemanticSegmentation\datasets\raw\001_shift.xyz"

# Смещения по осям (в метрах или тех же единицах, что и координаты)
shift_x = 100.0
shift_y = -200.0
shift_z = 50.0

# Создаём выходную директорию, если её нет
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Чтение, смещение и запись
with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # пропускаем некорректные строки

        x = float(parts[0]) + shift_x
        y = float(parts[1]) + shift_y
        z = float(parts[2]) + shift_z

        # Сохраняем остальные столбцы без изменений (например, RGB, intensity, label и т.д.)
        rest = parts[3:] if len(parts) > 3 else []
        new_line = f"{x} {y} {z}" + (" " + " ".join(rest) if rest else "") + "\n"
        outfile.write(new_line)