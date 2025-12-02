import { defineStore } from 'pinia';
import { ref, computed, shallowRef } from 'vue';
import { Api } from '@/api/Api';
import type { DatasetFile, DatasetStats } from '@/api/apiTypes';

export interface FolderInfo {
  name: string;
  filesCount: number;
}

export const useDatasetsStore = defineStore('datasets', () => {
  // ===== State =====
  const files = shallowRef<DatasetFile[]>([]);
  const folders = shallowRef<FolderInfo[]>([]);
  const currentSubdir = ref<string | null>(null);
  const selectedFile = ref<DatasetFile | null>(null);
  const currentStats = ref<DatasetStats | null>(null);

  const isLoading = ref(false);
  const isLoadingStats = ref(false);

  const error = ref<string | null>(null);
  const statsError = ref<string | null>(null);

  // ===== Getters =====
  const hasFiles = computed(() => files.value.length > 0);
  const hasFolders = computed(() => folders.value.length > 0);
  const isRootLevel = computed(() => currentSubdir.value === null);

  const classStatsArray = computed(() => {
    if (!currentStats.value?.class_stats) return [];

    return Object.entries(currentStats.value.class_stats)
      .map(([name, stats]) => ({
        name,
        ...stats,
      }))
      .sort((a, b) => b.count - a.count);
  });

  // ===== Actions =====

  async function fetchDatasets(subdir?: string): Promise<void> {
    isLoading.value = true;
    error.value = null;

    try {
      if (subdir) {
        // Запрос файлов из конкретной папки
        const response = await Api.getDatasetsSubdir(subdir);
        files.value = response || [];
        folders.value = [];
        currentSubdir.value = subdir;
      } else {
        // Запрос корневого уровня - получаем папки
        const response = await Api.getDatasetsRoot();

        // Преобразуем { "raw": [...], "predicted": [...] } в массив папок
        const foldersList: FolderInfo[] = Object.entries(response).map(([name, filesList]) => ({
          name,
          filesCount: filesList.length,
        }));

        folders.value = foldersList;
        files.value = [];
        currentSubdir.value = null;
      }

      selectedFile.value = null;
      currentStats.value = null;
    } catch (err: any) {
      error.value = err?.response?.data?.detail || err.message || 'Ошибка загрузки датасетов';
      files.value = [];
      folders.value = [];
      console.error('Ошибка загрузки датасетов:', err);
    } finally {
      isLoading.value = false;
    }
  }

  async function navigateToFolder(folderName: string): Promise<void> {
    await fetchDatasets(folderName);
  }

  async function navigateUp(): Promise<void> {
    await fetchDatasets();
  }

  function selectFile(file: DatasetFile): void {
    selectedFile.value = file;
  }

  async function fetchStats(fileName: string, subdir: string = 'raw'): Promise<void> {
    isLoadingStats.value = true;
    statsError.value = null;

    try {
      currentStats.value = await Api.getDatasetStats(fileName, subdir);
    } catch (err: any) {
      statsError.value = err?.response?.data?.detail || err.message || 'Ошибка загрузки статистики';
      currentStats.value = null;
      console.error('Ошибка загрузки статистики:', err);
    } finally {
      isLoadingStats.value = false;
    }
  }

  function clearStats(): void {
    currentStats.value = null;
    statsError.value = null;
  }

  function reset(): void {
    files.value = [];
    folders.value = [];
    currentSubdir.value = null;
    selectedFile.value = null;
    currentStats.value = null;
    error.value = null;
    statsError.value = null;
  }

  return {
    // State
    files,
    folders,
    currentSubdir,
    selectedFile,
    currentStats,
    isLoading,
    isLoadingStats,
    error,
    statsError,

    // Getters
    hasFiles,
    hasFolders,
    isRootLevel,
    classStatsArray,

    // Actions
    fetchDatasets,
    navigateToFolder,
    navigateUp,
    selectFile,
    fetchStats,
    clearStats,
    reset,
  };
});
