<template>
  <div class="models-page">
    <!-- Архитектуры -->
    <el-card class="models-page__section-card">
      <template #header>
        <div class="models-page__section-header">
          <div class="models-page__section-title">
            <h3>🏗️ Архитектуры моделей</h3>
          </div>
          <el-button
            type="primary"
            :icon="Refresh"
            @click="modelsStore.loadAllModels"
            :loading="modelsStore.loading.architectures || modelsStore.loading.trained"
            class="models-page__refresh-btn"
          >
            Обновить все
          </el-button>
        </div>
      </template>
      <div v-if="modelsStore.loading.architectures" class="models-page__loading-state">
        <el-skeleton :rows="3" animated />
      </div>
      <div v-else-if="modelsStore.architectures.length === 0" class="models-page__empty-state">
        ❌ Архитектуры моделей не найдены
      </div>
      <el-table v-else :data="modelsStore.architectures" class="models-page__table" size="small" stripe>
        <el-table-column prop="name" label="Название" min-width="180">
          <template #default="{ row }">
            <el-text type="primary" class="model-name">{{ row.name }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="file" label="Файл" width="160" />
        <el-table-column prop="size" label="Размер" width="100" />
        <el-table-column prop="modified" label="Дата" width="160" />
      </el-table>
    </el-card>

    <!-- Обученные модели -->
    <el-card class="models-page__section-card">
      <template #header>
        <div class="models-page__section-header">
          <div class="models-page__section-title">
            <el-icon><Finished /></el-icon>
            <h3>Обученные модели</h3>
          </div>
        </div>
      </template>
      <div v-if="modelsStore.loading.trained" class="models-page__loading-state">
        <el-skeleton :rows="4" animated />
      </div>
      <div v-else-if="modelsStore.trainedModels.length === 0" class="models-page__empty-state">
        ❌ Обученные модели не найдены
      </div>
      <el-table v-else :data="modelsStore.trainedModels" class="models-page__table" size="small" stripe>
        <el-table-column prop="display_name" label="Отображаемое имя" min-width="200">
          <template #default="{ row }">
            <el-text type="success" class="model-name">{{ row.display_name }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="Файл" width="160" />
        <el-table-column prop="size" label="Размер" width="100" />
        <el-table-column prop="modified" label="Дата" width="160" />
        <el-table-column label="Папка" min-width="280">
          <template #default="{ row }">
            <el-tag v-if="row.folder" type="info" size="small" effect="plain">
              {{ row.folder }}
            </el-tag>
            <span v-else>—</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { Finished, Refresh } from '@element-plus/icons-vue';
import { useModelsStore } from '@/stores/modelsStore';

const modelsStore = useModelsStore();

onMounted(() => {
  modelsStore.loadAllModels();
});
</script>
<style lang="scss" scoped>
.models-page {
  padding: 1.5rem;

  &__section-card {
    margin-bottom: 1.5rem;
    background: var(--el-fill-color-light);
    border-bottom: 1px solid var(--el-border-color-lighter);
  }

  &__section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
  }

  &__section-title {
    display: flex;
    align-items: center;
    gap: 8px;

    h3 {
      margin: 0;
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }
  }

  &__loading-state,
  &__empty-state {
    padding: 1rem;
    text-align: center;
    color: var(--el-text-color-secondary);
  }

  &__table {
    :deep(.model-name) {
      font-weight: 600;
    }
  }
}
</style>
