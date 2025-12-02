import { apiClient } from '@/api/apiClient';
import type {
  ModelArchitecture,
  TrainedModel,
  DatasetsRootResponse,
  DatasetsSubdirResponse,
  DatasetStats,
} from './apiTypes';

export const Api = {
  // ===== Модели =====
  getModelArchitectures: (): Promise<ModelArchitecture[]> => {
    return apiClient.get('/api/model-architectures');
  },

  getTrainedModels: (): Promise<TrainedModel[]> => {
    return apiClient.get('/api/trained-models');
  },

  // ===== Датасеты =====
  // Без subdir возвращает { "raw": [...], "predicted": [...] }
  getDatasetsRoot: (): Promise<DatasetsRootResponse> => {
    return apiClient.get('/api/datasets');
  },

  // С subdir возвращает массив файлов
  getDatasetsSubdir: (subdir: string): Promise<DatasetsSubdirResponse> => {
    return apiClient.get('/api/datasets', { params: { subdir } });
  },

  getDatasetStats: (filePath: string, subdir: string = 'raw'): Promise<DatasetStats> => {
    return apiClient.get('/api/dataset-stats', {
      params: { file_path: filePath, subdir },
    });
  },
};
