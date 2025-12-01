import { apiClient } from '@/api/apiClient';
import type { ModelArchitecture, TrainedModel } from './apiTypes';

export const Api = {
  // Получить архитектуры моделей
  getModelArchitectures: (): Promise<ModelArchitecture[]> => {
    return apiClient.get('/api/model-architectures');
  },

  // Получить обученные модели
  getTrainedModels: (): Promise<TrainedModel[]> => {
    return apiClient.get('/api/trained-models');
  },

  // Получить датасеты
  getDatasets: (subdir?: string) => {
    const params = subdir ? { subdir } : {};
    return apiClient.get('/api/datasets', { params });
  },

  // Сравнить XYZ файлы
  compareXYZ: (originalFile: string, predictedFile: string) => {
    return apiClient.get('/api/compare-xyz', {
      params: {
        original_file: originalFile,
        predicted_file: predictedFile,
      },
    });
  },

  // Получить статистику датасета
  getDatasetStats: (filePath: string, subdir: string = 'raw') => {
    return apiClient.get('/api/dataset-stats', {
      params: {
        file_path: filePath,
        subdir,
      },
    });
  },
};
