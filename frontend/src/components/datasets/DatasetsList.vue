<template>
  <div class="datasets-list">
    <!-- Header -->
    <div class="datasets-list__header">
      <div class="datasets-list__breadcrumb">
        <el-breadcrumb separator="/">
          <el-breadcrumb-item>
            <el-button type="primary" link :disabled="isRootLevel" @click="handleNavigateUp">
              <el-icon><HomeFilled /></el-icon>
              Datasets
            </el-button>
          </el-breadcrumb-item>
          <el-breadcrumb-item v-if="currentSubdir">
            {{ currentSubdir }}
          </el-breadcrumb-item>
        </el-breadcrumb>
      </div>

      <el-button type="primary" :icon="Refresh" :loading="isLoading" @click="handleRefresh"> Обновить </el-button>
    </div>

    <!-- Error -->
    <el-alert v-if="error" :title="error" type="error" show-icon closable class="datasets-list__error" />

    <!-- Loading -->
    <div v-if="isLoading" v-loading="true" class="datasets-list__loading" />

    <!-- Empty -->
    <el-empty v-else-if="!hasFolders && !hasFiles" description="Нет данных" class="datasets-list__empty" />

    <!-- Content with scroll -->
    <el-scrollbar v-else class="datasets-list__content">
      <!-- Folders (root level) -->
      <div v-if="hasFolders" class="datasets-list__folders">
        <h4 class="datasets-list__section-title">Папки</h4>
        <el-row :gutter="16">
          <el-col v-for="folder in folders" :key="folder.name" :xs="24" :sm="12" :md="8" :lg="6">
            <el-card class="datasets-list__folder-card" shadow="hover" @click="handleFolderClick(folder.name)">
              <div class="datasets-list__folder-content">
                <el-icon class="datasets-list__folder-icon" :size="32">
                  <Folder />
                </el-icon>
                <div class="datasets-list__folder-info">
                  <span class="datasets-list__folder-name">{{ folder.name }}</span>
                  <span class="datasets-list__folder-meta"> {{ folder.filesCount }} файлов </span>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </div>

      <!-- Files (inside folder) -->
      <div v-else-if="hasFiles" class="datasets-list__files">
        <h4 class="datasets-list__section-title">
          Файлы
          <el-tag type="info" size="small">{{ files.length }}</el-tag>
        </h4>

        <el-table
          :data="files"
          stripe
          highlight-current-row
          class="datasets-list__table"
          @row-click="handleRowClick"
          max-height="600"
        >
          <el-table-column prop="file" label="Имя файла" min-width="200">
            <template #default="{ row }">
              <div class="datasets-list__file-name">
                <el-icon><Document /></el-icon>
                <span>{{ row.file }}</span>
              </div>
            </template>
          </el-table-column>

          <el-table-column prop="size" label="Размер" width="120" />

          <el-table-column prop="points" label="Точек" width="140">
            <template #default="{ row }">
              <el-tag type="success" size="small">{{ row.points }}</el-tag>
            </template>
          </el-table-column>

          <el-table-column label="Действия" width="140" fixed="right">
            <template #default="{ row }">
              <el-button size="small" type="primary" :icon="PieChart" @click.stop="handleShowStats(row)">
                Статистика
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-scrollbar>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import { HomeFilled, Folder, Document, Refresh, PieChart } from '@element-plus/icons-vue';
import { useDatasetsStore } from '@/stores/datasetsStore';
import type { DatasetFile } from '@/api/apiTypes';

const emit = defineEmits<{
  (e: 'show-stats', file: DatasetFile): void;
}>();

const datasetsStore = useDatasetsStore();
const { files, folders, currentSubdir, isLoading, error, hasFiles, hasFolders, isRootLevel } =
  storeToRefs(datasetsStore);

function handleRefresh(): void {
  datasetsStore.fetchDatasets(currentSubdir.value || undefined);
}

function handleNavigateUp(): void {
  datasetsStore.navigateUp();
}

function handleFolderClick(folderName: string): void {
  datasetsStore.navigateToFolder(folderName);
}

function handleRowClick(row: DatasetFile): void {
  datasetsStore.selectFile(row);
}

function handleShowStats(file: DatasetFile): void {
  emit('show-stats', file);
}

onMounted(() => {
  datasetsStore.fetchDatasets();
});
</script>

<style lang="scss" scoped>
.datasets-list {
  &__header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--el-border-color-lighter);
  }

  &__error {
    margin-bottom: 16px;
  }

  &__loading {
    min-height: 200px;
  }

  &__empty {
    padding: 60px 0;
  }

  &__section-title {
    margin: 0 0 16px;
    font-size: 16px;
    font-weight: 500;
    color: var(--el-text-color-primary);

    .el-tag {
      margin-left: 8px;
      vertical-align: middle;
    }
  }

  &__folders {
    overflow: hidden !important;
    margin-bottom: 24px;
  }

  &__folder-card {
    cursor: pointer;
    transition:
      transform 0.2s,
      box-shadow 0.2s;
    margin-bottom: 16px;

    &:hover {
      transform: translateY(-2px);
      box-shadow: var(--el-box-shadow);
    }
  }

  &__folder-content {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  &__folder-icon {
    color: var(--el-color-warning);
    flex-shrink: 0;
  }

  &__folder-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }

  &__folder-name {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  &__folder-meta {
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  &__table {
    :deep(.el-table__row) {
      cursor: pointer;
    }
  }

  &__file-name {
    display: flex;
    align-items: center;
    gap: 8px;

    .el-icon {
      color: var(--el-color-primary);
    }
  }
}
</style>
