<!-- src/views/DatasetsView.vue -->
<template>
  <div class="datasets-view">
    <div class="datasets-view__container">
      <h2 class="datasets-view__title">
        <el-icon><Files /></el-icon>
        Управление датасетами
      </h2>

      <DatasetsList @show-stats="handleShowStats" />
    </div>

    <DatasetStats v-model="statsDialogVisible" :file-name="selectedFileName" :subdir="selectedSubdir" />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { storeToRefs } from 'pinia';
import { Files } from '@element-plus/icons-vue';
import DatasetsList from '@/components/datasets/DatasetsList.vue';
import DatasetStats from '@/components/datasets/DatasetStats.vue';
import { useDatasetsStore } from '@/stores/datasetsStore';
import type { DatasetFile } from '@/api/apiTypes';

const datasetsStore = useDatasetsStore();
const { currentSubdir } = storeToRefs(datasetsStore);

const statsDialogVisible = ref(false);
const selectedFileName = ref('');
const selectedSubdir = ref('raw');

function handleShowStats(file: DatasetFile): void {
  selectedFileName.value = file.file;
  selectedSubdir.value = currentSubdir.value || 'raw';
  statsDialogVisible.value = true;
}
</script>

<style lang="scss" scoped>
.datasets-view {
  padding: 24px;
  min-height: 100%;
  background: var(--el-bg-color-page);

  &__container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    background: var(--el-bg-color);
    border-radius: 8px;
    box-shadow: var(--el-box-shadow-light);
  }

  &__title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 24px;
    font-size: 20px;
    font-weight: 600;
    color: var(--el-text-color-primary);

    .el-icon {
      color: var(--el-color-primary);
    }
  }
}
</style>
