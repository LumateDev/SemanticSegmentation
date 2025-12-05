export interface ModelArchitecture {
  name: string;
  file: string;
  size: string;
  modified: string;
}

export interface TrainedModel {
  name: string;
  display_name: string;
  size: string;
  modified: string;
  folder?: string;
  path: string;
  full_path: string;
}

// Файл датасета (когда запрашиваем с subdir)
export interface DatasetFile {
  name: string; // без расширения
  file: string; // полное имя файла с расширением
  size: string;
  points: string;
}

// Ответ без subdir - объект с ключами-папками
export type DatasetsRootResponse = Record<string, DatasetFile[]>;

// Ответ С subdir - массив файлов
export type DatasetsSubdirResponse = DatasetFile[];

export interface ClassStat {
  label: number;
  count: number;
  percentage: number;
}

export interface DatasetStats {
  file: string;
  total_points: number;
  unique_classes: number;
  class_stats: Record<string, ClassStat>;
}
