<template>
  <div class="train-page">
    <div class="train-page__container">
      <!-- Конфигурационная панель -->
      <div class="train-page__config">
        <h3 class="train-page__config-title">⚙️ Конфигурация обучения</h3>

        <!-- Название модели -->
        <el-form-item label="Название модели:">
          <el-input v-model="form.model_name" placeholder="Например: my_model_v1" clearable />
          <div class="train-page__name-hint">
            <small>Оставьте пустым для автоматического названия</small>
          </div>
        </el-form-item>

        <!-- Выбор датасетов -->
        <el-form-item label="Датасеты (.xyz):">
          <div v-if="datasetsStore.isLoading" class="train-page__loading-datasets">
            <el-skeleton :rows="3" animated />
          </div>
          <el-checkbox-group v-else v-model="selectedDatasetFiles" class="train-page__dataset-list">
            <el-checkbox
              v-for="ds in unlabeledDatasets"
              :key="ds.file"
              :label="`datasets/raw/${ds.file}`"
              class="train-page__dataset-item"
            >
              {{ ds.name }} ({{ ds.points }} точек)
            </el-checkbox>
          </el-checkbox-group>
          <div v-if="!datasetsStore.isLoading && unlabeledDatasets.length === 0" class="train-page__no-datasets">
            Нет доступных датасетов
          </div>
          <div class="train-page__dataset-actions">
            <el-button size="small" @click="selectAllDatasets">Выбрать все</el-button>
            <el-button size="small" @click="deselectAllDatasets">Снять все</el-button>
          </div>
        </el-form-item>

        <!-- Чекпоинт для дообучения -->
        <el-form-item label="Чекпоинт для дообучения (опционально):">
          <el-select
            v-model="form.resume_from"
            clearable
            placeholder="Не дообучать (новая модель)"
            :loading="modelsStore.loading.trained"
            style="width: 100%"
          >
            <el-option
              v-for="model in modelsStore.trainedModels"
              :key="model.path"
              :value="model.full_path"
              :label="`${model.display_name} (${model.modified})`"
            />
          </el-select>
        </el-form-item>

        <!-- Batch Size -->
        <el-form-item label="Batch Size:">
          <el-input-number v-model="form.batch_size" :min="1" :max="32" controls-position="right" style="width: 100%" />
        </el-form-item>

        <!-- Epochs -->
        <el-form-item label="Количество эпох:">
          <el-input-number v-model="form.epochs" :min="1" :max="100" controls-position="right" style="width: 100%" />
        </el-form-item>

        <!-- Learning Rate -->
        <el-form-item label="Learning Rate:">
          <el-input-number
            v-model="form.learning_rate"
            :min="0.0001"
            :max="0.1"
            :step="0.0001"
            controls-position="right"
            style="width: 100%"
          />
        </el-form-item>

        <!-- Кнопка запуска -->
        <el-button
          class="train-page__train-btn"
          type="primary"
          :disabled="isTraining || !canStart"
          :loading="isTraining"
          @click="startTraining"
        >
          🚀 Начать обучение
        </el-button>

        <!-- Примечание -->
        <div class="train-page__note">
          <small>
            <strong>💡 Примечание:</strong><br />
            • Модель сохраняется в <code>checkpoints/DGCNN/[название]/</code><br />
            • Автоматическое имя: <code>DGCNN_YYYYMMDD_HHMMSS</code><br />
            • Ручное имя: <code>ваше_название_YYYYMMDD_HHMMSS</code>
          </small>
        </div>
      </div>

      <!-- Логи -->
      <div class="train-page__logs">
        <WebSocketLogBox ref="logBoxRef" title="📊 Логи обучения" :logs="logs" :status="wsStatus" @clear="clearLogs" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { ElMessage } from 'element-plus';
import WebSocketLogBox from '@/components/WebSocketLogBox.vue';
import { useWebSocket } from '@/composables/useWebSocket';
import { useWebSocketHandlers } from '@/composables/useWebSocketHandlers';
import { useModelsStore } from '@/stores/modelsStore';
import { useDatasetsStore } from '@/stores/datasetsStore';
import type { LogEntry } from '@/composables/useWebSocketHandlers';
import type { WebSocketStatus } from '@/composables/useWebSocket';

// === Stores ===
const modelsStore = useModelsStore();
const datasetsStore = useDatasetsStore();

// === Reactive state ===
interface TrainingForm {
  model_name: string;
  batch_size: number;
  epochs: number;
  learning_rate: number;
  resume_from: string | null;
  xyz_files: string[]; // будет заполнено при отправке
}

const form = ref<TrainingForm>({
  model_name: '',
  batch_size: 8,
  epochs: 3,
  learning_rate: 0.001,
  resume_from: null,
  xyz_files: [],
});

const wsBaseUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://127.0.0.1:8000';
const wsUrl = `${wsBaseUrl}/api/ws/train-model`;
const logs = ref<LogEntry[]>([]);
const isTraining = ref(false);
const selectedDatasetFiles = ref<string[]>([]);
const logBoxRef = ref<InstanceType<typeof WebSocketLogBox> | null>(null);

// === Unlabeled datasets ===
const unlabeledDatasets = computed(() => {
  return datasetsStore.files;
});

// === WebSocket ===
const {
  ws,
  status: wsStatus,
  connect,
  disconnect,
} = useWebSocket({
  url: wsUrl,
  onOpen: () => {
    const payload = {
      ...form.value,
      xyz_files: [...selectedDatasetFiles.value],
      resume_from: form.value.resume_from || null,
    };

    ws.value?.send(JSON.stringify(payload));
  },
  onClose: () => {
    finishTraining();
  },
  onError: () => {
    logs.value.push({
      timestamp: new Date().toLocaleTimeString('ru-RU'),
      message: '❌ Ошибка соединения с WebSocket',
      level: 'error',
    });
    finishTraining();
  },
});

// === Логи ===
const logsContainerRef = computed(() => logBoxRef.value?.logsContainerRef ?? null);
const { handleWebSocketMessage, clearLogs } = useWebSocketHandlers(logs, logsContainerRef);

watch(ws, newWs => {
  if (newWs) {
    newWs.onmessage = handleWebSocketMessage;
  }
});

// === Lifecycle ===
onMounted(async () => {
  await modelsStore.loadTrainedModels();
  await datasetsStore.fetchDatasets('raw');
});

// === Вычисляемые свойства ===
const canStart = computed(() => {
  return selectedDatasetFiles.value.length > 0;
});

// === Методы ===
const selectAllDatasets = () => {
  selectedDatasetFiles.value = unlabeledDatasets.value.map(ds => `datasets/raw/${ds.file}`);
};

const deselectAllDatasets = () => {
  selectedDatasetFiles.value = [];
};

const startTraining = () => {
  if (isTraining.value) {
    ElMessage.warning('Обучение уже запущено!');
    return;
  }

  if (selectedDatasetFiles.value.length === 0) {
    ElMessage.warning('Выберите хотя бы один датасет для обучения!');
    return;
  }

  logs.value = [];
  isTraining.value = true;
  connect();
};

const finishTraining = () => {
  isTraining.value = false;
  disconnect();
};
</script>

<style lang="scss" scoped>
.train-page {
  padding: 20px;
  font-family: Arial, sans-serif;

  &__container {
    display: flex;
    gap: 20px;
  }

  &__config {
    flex: 1;
    padding: 20px;
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    background: var(--el-bg-color-page);

    &-title {
      margin: 0 0 20px;
      color: var(--el-text-color-primary);
    }
  }

  &__name-hint {
    margin-top: 5px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__loading-datasets,
  &__no-datasets {
    padding: 8px 0;
    color: var(--el-text-color-secondary);
  }

  &__dataset-list {
    display: flex;
    flex-direction: column;
    max-height: 200px;
    overflow-y: auto;
    padding: 8px 0;
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    background: var(--el-fill-color-light);
  }

  &__dataset-item {
    padding: 8px;
    margin: 4px 0;
    &:hover {
      background: var(--el-fill-color);
      border-radius: var(--el-border-radius-base);
    }
  }

  &__dataset-actions {
    margin-top: 10px;
    display: flex;
    gap: 10px;
  }

  &__train-btn {
    width: 100%;
    margin: 15px 0;
    height: 40px;
    font-size: 16px;
  }

  &__note {
    margin-top: 20px;
    padding: 10px;
    background: var(--el-fill-color-light);
    border-radius: var(--el-border-radius-base);
    font-size: 12px;
    color: var(--el-text-color-secondary);

    code {
      background: var(--el-fill-color);
      padding: 2px 4px;
      border-radius: 3px;
      font-family: monospace;
    }
  }

  &__logs {
    flex: 2;
    height: 800px;
  }
}
</style>
