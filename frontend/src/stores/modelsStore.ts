import { defineStore } from 'pinia';
import { Api } from '@/api/Api';
import type { ModelArchitecture, TrainedModel } from '@/api/apiTypes';

export const useModelsStore = defineStore('useModelsStore', () => {
  const architectures = ref<ModelArchitecture[]>([]);
  const trainedModels = ref<TrainedModel[]>([]);
  const loading = ref({
    architectures: false,
    trained: false,
  });

  // Api
  const loadArchitectures = async () => {
    loading.value.architectures = true;
    try {
      const data = await Api.getModelArchitectures();
      architectures.value = data;
    } catch (err) {
      ElMessage.error('Ошибка загрузки архитектур');
      architectures.value = [];
    } finally {
      loading.value.architectures = false;
    }
  };

  const loadTrainedModels = async () => {
    loading.value.trained = true;
    try {
      const data = await Api.getTrainedModels();
      trainedModels.value = data;
    } catch (err) {
      ElMessage.error('Ошибка загрузки обученных моделей');
      trainedModels.value = [];
    } finally {
      loading.value.trained = false;
    }
  };

  const loadAllModels = () => {
    loadArchitectures();
    loadTrainedModels();
  };

  return {
    architectures,
    trainedModels,
    loading,

    loadArchitectures,
    loadTrainedModels,
    loadAllModels,
  };
});
