<template>
  <div class="predict-page">
    <div class="predict-page__container">
      <!-- Конфигурационная панель -->
      <div class="predict-page__config">
        <h3 class="predict-page__config-title">⚙️ Конфигурация предсказания</h3>

        <!-- Выбор модели -->
        <el-form-item label="Модель:">
          <el-select
            v-model="form.checkpoint_path"
            placeholder="Выберите модель..."
            :loading="modelsStore.loading.trained"
            style="width: 100%"
          >
            <el-option
              v-for="model in modelsStore.trainedModels"
              :key="model.path"
              :value="model.full_path"
              :label="`${model.display_name} (${model.size})`"
            />
          </el-select>
        </el-form-item>

        <!-- Выбор датасетов -->
        <el-form-item label="Датасеты:">
          <el-checkbox-group v-model="selectedDatasetFiles" class="predict-page__dataset-list">
            <el-checkbox
              v-for="ds in unlabeledDatasets"
              :key="ds.file"
              :label="`datasets/raw/${ds.file}`"
              class="predict-page__dataset-item"
            >
              {{ ds.name }} ({{ ds.points }} точек)
            </el-checkbox>
          </el-checkbox-group>
          <div class="predict-page__dataset-actions">
            <el-button size="small" @click="selectAllDatasets">Выбрать все</el-button>
            <el-button size="small" @click="deselectAllDatasets">Снять все</el-button>
          </div>
        </el-form-item>

        <!-- Batch Size -->
        <el-form-item label="Batch Size:">
          <el-input-number v-model="form.batch_size" :min="1" :max="32" controls-position="right" style="width: 100%" />
        </el-form-item>

        <!-- Output Dir -->
        <el-form-item label="Выходная папка:">
          <el-input v-model="form.output_dir" readonly />
        </el-form-item>

        <!-- Кнопка запуска -->
        <el-button
          class="predict-page__predict-btn"
          type="primary"
          :disabled="isPredicting || !canStart"
          :loading="isPredicting"
          @click="startPrediction"
        >
          🔮 Запустить предсказание
        </el-button>

        <!-- Статистика -->
        <div v-if="showStats" class="predict-page__stats-panel">
          <div class="predict-page__stats-header">📊 Агрегированная статистика:</div>
          <div class="predict-page__stats-content">
            <div v-for="(item, idx) in statsLines" :key="idx" class="predict-page__stats-item" v-html="item" />
          </div>
        </div>

        <!-- Примечание -->
        <div class="predict-page__note">
          <small>
            <strong>💡 Примечание:</strong><br />
            • Используется обученная модель<br />
            • Обрабатываются unlabeled датасеты<br />
            • Результаты сохраняются в {{ form.output_dir }}<br />
            • <strong>Новое:</strong> поддержка множественных датасетов!
          </small>
        </div>
      </div>

      <!-- Логи -->
      <div class="predict-page__logs">
        <WebSocketLogBox
          ref="logBoxRef"
          title="📊 Логи предсказания"
          :logs="logs"
          :status="wsStatus"
          @clear="clearLogs"
        />
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
interface PredictionForm {
  checkpoint_path: string;
  batch_size: number;
  output_dir: string;
  input_file?: string;
  input_files?: string[];
}

const form = ref<PredictionForm>({
  checkpoint_path: '',
  batch_size: 16,
  output_dir: 'datasets/predicted',
});

const wsBaseUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://127.0.0.1:8000';
const wsUrl = `${wsBaseUrl}/api/ws/predict`;
const logs = ref<LogEntry[]>([]);
const isPredicting = ref(false);
const selectedDatasetFiles = ref<string[]>([]);
const showStats = ref(false);
const statsLines = ref<string[]>([]);

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
    const payload: PredictionForm = { ...form.value };

    if (selectedDatasetFiles.value.length === 1) {
      payload.input_file = selectedDatasetFiles.value[0];
    } else if (selectedDatasetFiles.value.length > 1) {
      payload.input_files = [...selectedDatasetFiles.value];
    }

    ws.value?.send(JSON.stringify(payload));
  },
  onClose: () => {
    finishPrediction();
  },
  onError: () => {
    logs.value.push({
      timestamp: new Date().toLocaleTimeString('ru-RU'),
      message: '❌ Ошибка соединения с WebSocket',
      level: 'error',
    });
    finishPrediction();
  },
});

// === Логи ===
const logsContainerRef = computed(() => logBoxRef.value?.logsContainerRef ?? null);
const { handleWebSocketMessage, clearLogs } = useWebSocketHandlers(logs, logsContainerRef);

// Привязываем обработчик сообщений к WebSocket
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
  return form.value.checkpoint_path !== '' && selectedDatasetFiles.value.length > 0;
});

// === Методы ===
const selectAllDatasets = () => {
  selectedDatasetFiles.value = unlabeledDatasets.value.map(ds => `datasets/raw/${ds.file}`);
};

const deselectAllDatasets = () => {
  selectedDatasetFiles.value = [];
};

const startPrediction = () => {
  if (isPredicting.value) {
    ElMessage.warning('Предсказание уже запущено!');
    return;
  }

  if (!form.value.checkpoint_path) {
    ElMessage.warning('Выберите модель для предсказания!');
    return;
  }

  if (selectedDatasetFiles.value.length === 0) {
    ElMessage.warning('Выберите хотя бы один датасет!');
    return;
  }

  showStats.value = false;
  logs.value = [];
  isPredicting.value = true;
  connect();
};

const finishPrediction = () => {
  isPredicting.value = false;
  disconnect();
};
</script>

<style lang="scss" scoped>
.predict-page {
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

  &__predict-btn {
    width: 100%;
    margin: 15px 0;
    height: 40px;
    font-size: 16px;
  }

  &__stats-panel {
    margin-top: 20px;
    padding: 15px;
    background: var(--el-fill-color-light);
    border-radius: var(--el-border-radius-base);
  }

  &__stats-header {
    font-weight: bold;
    margin-bottom: 10px;
    color: var(--el-text-color-primary);
  }

  &__stats-item {
    padding: 5px;
    margin-bottom: 5px;
    background: var(--el-bg-color);
    border-radius: var(--el-border-radius-small);
    font-family: monospace;
    font-size: 13px;
  }

  &__note {
    margin-top: 20px;
    padding: 10px;
    background: var(--el-fill-color-light);
    border-radius: var(--el-border-radius-base);
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__logs {
    flex: 2;
    height: 800px;
  }
}
</style>
