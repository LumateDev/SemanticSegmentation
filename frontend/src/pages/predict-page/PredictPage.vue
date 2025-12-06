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

        <!-- Кнопки в одну строку -->
        <div class="predict-page__buttons-row">
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

          <!-- Кнопка показа статистики -->
          <el-button
            class="predict-page__stats-btn"
            type="success"
            :disabled="!hasResults"
            @click="toggleStatsPanel"
          >
            <el-icon v-if="showStatsInLogs"><Hide /></el-icon>
            <el-icon v-else><View /></el-icon>
            {{ showStatsInLogs ? '📊 Скрыть' : '📊 Показать' }}
          </el-button>
        </div>

        <!-- Примечание -->
        <div class="predict-page__note">
          <small>
            <strong>💡 Новая функция:</strong><br />
            • Вывод точности для каждого датасета<br />
            • Формат: "Датасет: общая точность, класс 1: точность, класс N: точность"<br />
            • Поддержка множественных датасетов<br />
            • Статистика отображается под логами
          </small>
        </div>
      </div>

      <!-- Правая часть: Логи и статистика -->
      <div class="predict-page__right">
        <!-- Логи с динамической высотой -->
        <div class="predict-page__logs" :class="{ 'logs-compact': showStatsInLogs }">
          <WebSocketLogBox
            ref="logBoxRef"
            title="📊 Логи предсказания"
            :logs="logs"
            :status="wsStatus"
            @clear="clearLogs"
          />
        </div>

        <!-- Статистика под логами -->
        <div v-if="showStatsInLogs && hasResults" class="predict-page__stats-panel" ref="statsPanelRef">
          <!-- Вкладки для переключения между общей и детальной статистикой -->
          <div class="stats-tabs">
            <el-radio-group v-model="activeStatsTab" size="small">
              <el-radio-button label="general">📊 Общая статистика</el-radio-button>
              <el-radio-button label="detailed">📈 Детальная статистика</el-radio-button>
            </el-radio-group>
          </div>
          
          <!-- Общая статистика -->
          <div v-if="activeStatsTab === 'general'" class="stats-content">
            <div class="stats-header">
              <h4>📊 Общая статистика предсказания</h4>
              <el-button size="small" type="info" @click="copyStatsToClipboard" title="Копировать статистику">
                📋 Копировать
              </el-button>
            </div>
            
            <!-- Основные метрики -->
            <div class="stats-metrics">
              <div class="metric-card">
                <div class="metric-label">Обработано датасетов</div>
                <div class="metric-value">{{ predictionResult.datasets_processed || 1 }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Всего точек</div>
                <div class="metric-value">{{ formatNumber(predictionResult.total_points || 0) }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Общая точность</div>
                <div class="metric-value" :class="getAccuracyClass(predictionResult.accuracy?.overall)">
                  {{ predictionResult.accuracy?.overall || 'N/A' }}
                </div>
              </div>
            </div>
            
            <!-- Агрегированная статистика -->
            <div v-if="predictionResult.aggregated_statistics" class="aggregated-section">
              <h5>Распределение по классам:</h5>
              <div class="stats-table">
                <div class="table-header">
                  <div class="table-cell class-cell">Класс</div>
                  <div class="table-cell count-cell">Кол-во точек</div>
                  <div class="table-cell percent-cell">%</div>
                </div>
                <div v-for="(stat, className) in predictionResult.aggregated_statistics" :key="className" class="table-row">
                  <div class="table-cell class-cell">{{ className }}</div>
                  <div class="table-cell count-cell">{{ extractCountFromStat(stat) }}</div>
                  <div class="table-cell percent-cell" :class="getAccuracyClass(predictionResult.accuracy?.per_class?.[className])">
                    {{ extractPercentFromStat(stat) }}
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Точность по классам -->
            <div v-if="predictionResult.accuracy?.per_class" class="accuracy-section">
              <h5>Точность по классам:</h5>
              <div class="accuracy-grid">
                <div v-for="(accuracy, className) in predictionResult.accuracy.per_class" :key="className" 
                     class="accuracy-item" :class="getAccuracyClass(accuracy)">
                  <span class="class-name">{{ className }}</span>
                  <span class="class-accuracy">{{ accuracy }}</span>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Детальная статистика -->
          <div v-if="activeStatsTab === 'detailed' && predictionResult.detailed_statistics" class="stats-content detailed-content">
            <div class="stats-header">
              <h4>📈 Детальная статистика по датасетам</h4>
              <div class="dataset-count">
                Всего датасетов: {{ predictionResult.detailed_statistics.length }}
              </div>
            </div>
            
            <div class="datasets-list">
              <div v-for="(dataset, index) in predictionResult.detailed_statistics" :key="index" class="dataset-card">
                <div class="dataset-header">
                  <span class="dataset-index">{{ index + 1 }}.</span>
                  <span class="dataset-name">{{ dataset.dataset }}</span>
                  <span class="dataset-points">{{ formatNumber(dataset.total_points) }} точек</span>
                </div>
                
                <!-- Общая точность датасета -->
                <div v-if="dataset.accuracy?.overall" class="dataset-accuracy">
                  🎯 Общая точность: <strong>{{ dataset.accuracy.overall }}</strong>
                </div>
                
                <!-- Статистика по классам -->
                <div v-if="dataset.class_distribution" class="dataset-classes">
                  <div class="classes-table">
                    <div class="classes-header">
                      <div class="cell class-name-cell">Класс</div>
                      <div class="cell class-count-cell">Кол-во</div>
                      <div class="cell class-percent-cell">%</div>
                      <div class="cell class-meter-cell">Уровень</div>
                      <div class="cell class-acc-cell">Точность</div>
                    </div>
                    <div v-for="(classData, className) in dataset.class_distribution" :key="className" class="classes-row">
                      <div class="cell class-name-cell">{{ className }}</div>
                      <div class="cell class-count-cell">{{ formatNumber(classData.count) }}</div>
                      <div class="cell class-percent-cell">{{ classData.percentage }}%</div>
                      <div class="cell class-meter-cell">
                        <div class="accuracy-meter">
                          <div class="meter-fill" :class="getAccuracyClass(dataset.accuracy?.per_class?.[className])" 
                               :style="{ width: getAccuracyPercent(dataset.accuracy?.per_class?.[className]) + '%' }"></div>
                        </div>
                      </div>
                      <div class="cell class-acc-cell" :class="getAccuracyClass(dataset.accuracy?.per_class?.[className])">
                        {{ getAccuracyValue(dataset.accuracy?.per_class?.[className]) }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue';
import { ElMessage } from 'element-plus';
import { Hide, View } from '@element-plus/icons-vue';
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

interface ClassDistribution {
  count: number;
  percentage: number;
  formatted: string;
}

interface DatasetAccuracy {
  overall: string;
  per_class: Record<string, string>;
}

interface DatasetStat {
  dataset: string;
  total_points: number;
  class_distribution: Record<string, ClassDistribution>;
  accuracy?: DatasetAccuracy;
}

interface PredictionResult {
  success: boolean;
  message: string;
  datasets_processed?: number;
  datasets_failed?: number;
  total_points?: number;
  detailed_statistics?: DatasetStat[];
  aggregated_statistics?: Record<string, string>;
  accuracy?: {
    overall: string;
    per_class: Record<string, string>;
    per_dataset?: Array<{file: string; accuracy: number; total_points: number}>;
  };
  statistics?: Record<string, string>;
  session_id?: string;
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
const showStatsInLogs = ref(false);
const activeStatsTab = ref<'general' | 'detailed'>('general');
const predictionResult = ref<PredictionResult>({} as PredictionResult);
const statsPanelRef = ref<HTMLElement | null>(null);

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

// Расширенный обработчик сообщений для извлечения статистики
const enhancedHandleWebSocketMessage = async (event: MessageEvent) => {
  handleWebSocketMessage(event);
  
  try {
    const data = JSON.parse(event.data);
    
    if (data.type === 'result' && data.data.success) {
      predictionResult.value = data.data;

      showStatsInLogs.value = true;

      await nextTick();
      if (statsPanelRef.value) {
        const offset = 60;
        const elementPosition = statsPanelRef.value.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - offset;
        
        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    }

    if (data.type === 'error') {
      predictionResult.value = {} as PredictionResult;
      showStatsInLogs.value = false;
    }
    
  } catch (error) {

  }
};

// Привязываем обработчик сообщений к WebSocket
watch(ws, newWs => {
  if (newWs) {
    newWs.onmessage = enhancedHandleWebSocketMessage;
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

const hasResults = computed(() => {
  return predictionResult.value.success === true;
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

  // Сбрасываем статистику
  showStatsInLogs.value = false;
  predictionResult.value = {} as PredictionResult;
  
  // Очищаем логи
  clearLogs();
  
  isPredicting.value = true;
  connect();
};

const finishPrediction = () => {
  isPredicting.value = false;
  disconnect();
};

const toggleStatsPanel = () => {
  showStatsInLogs.value = !showStatsInLogs.value;

  if (showStatsInLogs.value && statsPanelRef.value) {
    nextTick(() => {
      if (statsPanelRef.value) {
        const offset = 60;
        const elementPosition = statsPanelRef.value.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - offset;
        
        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  }
};

const formatNumber = (num: number): string => {
  return num.toLocaleString('ru-RU');
};

const getAccuracyValue = (accuracy: string | undefined): string => {
  if (!accuracy) return 'N/A';
  return accuracy;
};

const getAccuracyClass = (accuracy: string | undefined): string => {
  if (!accuracy || accuracy === 'N/A') return '';
  
  const accuracyStr = accuracy.replace('%', '');
  const num = parseFloat(accuracyStr);
  if (isNaN(num)) return '';
  
  if (num >= 80) return 'accuracy-high';
  if (num >= 50) return 'accuracy-medium';
  return 'accuracy-low';
};

// Получение процента точности для ползунка
const getAccuracyPercent = (accuracy: string | undefined): number => {
  if (!accuracy || accuracy === 'N/A') return 0;
  
  const accuracyStr = accuracy.replace('%', '');
  const num = parseFloat(accuracyStr);
  return isNaN(num) ? 0 : Math.min(num, 100);
};

const extractCountFromStat = (stat: string | undefined): string => {
  if (!stat) return '0';
  const match = stat.match(/^([\d,]+)/);
  if (match && match[1]) {
    return match[1];
  }
  return stat;
};

const extractPercentFromStat = (stat: string | undefined): string => {
  if (!stat) return '';
  const match = stat.match(/\(([\d.]+)%\)/);
  if (match && match[1]) {
    return match[1] + '%';
  }
  return '';
};

const copyStatsToClipboard = () => {
  let textToCopy = '📊 Статистика предсказания\n\n';
  
  // Общая информация
  textToCopy += `✅ Обработано датасетов: ${predictionResult.value.datasets_processed || 1}\n`;
  textToCopy += `📊 Всего точек: ${formatNumber(predictionResult.value.total_points || 0)}\n`;
  
  if (predictionResult.value.accuracy?.overall) {
    textToCopy += `🎯 Общая точность: ${predictionResult.value.accuracy.overall}\n\n`;
  }
  
  // Точность по классам
  if (predictionResult.value.accuracy?.per_class) {
    textToCopy += '🎯 Точность по классам:\n';
    Object.entries(predictionResult.value.accuracy.per_class).forEach(([className, accuracy]) => {
      textToCopy += `  ${className}: ${accuracy}\n`;
    });
    textToCopy += '\n';
  }
  
  // Распределение по классам
  if (predictionResult.value.aggregated_statistics) {
    textToCopy += '📊 Распределение по классам:\n';
    Object.entries(predictionResult.value.aggregated_statistics).forEach(([className, stat]) => {
      textToCopy += `  ${className}: ${stat}\n`;
    });
  }
  
  // Копируем в буфер обмена
  navigator.clipboard.writeText(textToCopy)
    .then(() => {
      ElMessage.success('Статистика скопирована в буфер обмена!');
    })
    .catch(() => {
      ElMessage.error('Не удалось скопировать статистику');
    });
};
</script>

<style lang="scss" scoped>
.predict-page {
  padding: 20px;
  font-family: Arial, sans-serif;
  display: flex;
  flex-direction: column;
  overflow: hidden;

  &__container {
    display: flex;
    gap: 20px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  &__config {
    flex: 1;
    padding: 20px;
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    background: var(--el-bg-color-page);
    min-width: 350px;
    max-width: 500px;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;

    &-title {
      margin: 0 0 20px;
      color: var(--el-text-color-primary);
      flex-shrink: 0;
    }
  }

  &__dataset-list {
    display: flex;
    flex-direction: column;
    max-height: 170px;
    overflow-y: auto;
    padding: 8px 0;
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    background: var(--el-fill-color-light);
    flex-shrink: 0;
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
    flex-shrink: 0;
  }

  &__buttons-row {
    display: flex;
    gap: 10px;
    margin: 15px 0;
    flex-shrink: 0;
  }

  &__predict-btn {
    flex: 2;
    height: 40px;
    font-size: 16px;
  }
  
  &__stats-btn {
    flex: 1;
    height: 40px;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    white-space: nowrap;
  }

  &__note {
    margin-top: 2%;
    padding: 10px;
    background: var(--el-fill-color-light);
    border-radius: var(--el-border-radius-base);
    font-size: 12px;
    color: var(--el-text-color-secondary);
    flex-shrink: 0;
  }

  &__right {
    flex: 2;
    display: flex;
    flex-direction: column;
    gap: 15px;
    min-width: 600px;
    height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  &__logs {
    flex-shrink: 0;
    height: 240px;
    min-height: 240px;
    transition: height 0.3s ease;
    
    &.logs-compact {
      height: 135px;
      min-height: 135px;
    }
    
    :deep(.websocket-log-box) {
      height: 100%;
      display: flex;
      flex-direction: column;
      
      .websocket-log-box__logs {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
      }
    }
  }
  
  &__stats-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--el-bg-color);
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    overflow: hidden;
    min-height: 390px;
    max-height: calc(100vh - 320px);
    
    .stats-tabs {
      padding: 18px 18px 0;
      border-bottom: 1px solid var(--el-border-color);
      background: var(--el-fill-color-lighter);
      flex-shrink: 0;
    }
    
    .stats-content {
      flex: 1;
      padding: 18px;
      height: max-content;
      
      &.detailed-content {
        display: flex;
        flex-direction: column;
        min-height: 0;
        overflow-y: auto;
      }
    }
    
    .stats-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 22px;
      padding-bottom: 12px;
      border-bottom: 2px solid var(--el-border-color);
      flex-shrink: 0;
      
      h4 {
        margin: 0;
        color: var(--el-text-color-primary);
        font-size: 18px;
      }
      
      .dataset-count {
        font-size: 13px;
        color: var(--el-text-color-secondary);
      }
    }
    
    .stats-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-bottom: 30px;
      flex-shrink: 0;
      
      .metric-card {
        padding: 18px;
        background: var(--el-fill-color-light);
        border-radius: var(--el-border-radius-base);
        text-align: center;
        border: 1px solid var(--el-border-color);
        
        .metric-label {
          font-size: 13px;
          color: var(--el-text-color-secondary);
          margin-bottom: 9px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .metric-value {
          font-size: 24px;
          font-weight: bold;
          color: var(--el-text-color-primary);
          
          &.accuracy-high {
            color: var(--el-color-success);
          }
          
          &.accuracy-medium {
            color: var(--el-color-warning);
          }
          
          &.accuracy-low {
            color: var(--el-color-danger);
          }
        }
      }
    }
    
    .aggregated-section,
    .accuracy-section {
      margin-bottom: 30px;
      flex-shrink: 0;
      
      h5 {
        margin: 0 0 12px;
        color: var(--el-text-color-regular);
        font-size: 16px;
        font-weight: 600;
      }
    }
    
    .stats-table {
      border: 1px solid var(--el-border-color);
      border-radius: var(--el-border-radius-base);
      overflow: hidden;
      font-size: 14px;
      
      .table-header,
      .table-row {
        display: flex;
        padding: 9px 15px;
        border-bottom: 1px solid var(--el-border-color-light);
        
        &:last-child {
          border-bottom: none;
        }
      }
      
      .table-header {
        background: var(--el-fill-color-lighter);
        font-weight: bold;
        color: var(--el-text-color-primary);
      }
      
      .table-row:hover {
        background: var(--el-fill-color-light);
      }
      
      .table-cell {
        padding: 0 6px;
        
        &.class-cell {
          flex: 2;
          font-weight: 600;
        }
        
        &.count-cell {
          flex: 1;
          text-align: right;
        }
        
        &.percent-cell {
          flex: 1;
          text-align: right;
          color: var(--el-text-color-secondary);
          
          &.accuracy-high {
            color: var(--el-color-success);
            font-weight: bold;
          }
          
          &.accuracy-medium {
            color: var(--el-color-warning);
            font-weight: bold;
          }
          
          &.accuracy-low {
            color: var(--el-color-danger);
            font-weight: bold;
            }
          }
        }
    }
    
    .accuracy-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 9px;
      
      .accuracy-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 15px;
        background: var(--el-fill-color-light);
        border-radius: var(--el-border-radius-base);
        border-left: 3px solid var(--el-border-color);
        font-size: 14px;
        
        &.accuracy-high {
          border-left-color: var(--el-color-success);
          background: var(--el-color-success-light-9);
        }
        
        &.accuracy-medium {
          border-left-color: var(--el-color-warning);
          background: var(--el-color-warning-light-9);
        }
        
        &.accuracy-low {
          border-left-color: var(--el-color-danger);
          background: var(--el-color-danger-light-9);
        }
        
        .class-name {
          font-weight: 600;
          color: var(--el-text-color-primary);
        }
        
        .class-accuracy {
          font-weight: bold;
          font-size: 16px;
        }
      }
    }
    
    .datasets-list {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
    }
    
    .dataset-card {
      margin-bottom: 18px;
      padding: 15px;
      background: var(--el-fill-color-light);
      border-radius: var(--el-border-radius-base);
      border: 1px solid var(--el-border-color);
      
      &:last-child {
        margin-bottom: 5px;
      }
      
      .dataset-header {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--el-border-color-light);
        
        .dataset-index {
          font-weight: bold;
          color: var(--el-color-primary);
          margin-right: 5px;
          font-size: 14px;
        }
        
        .dataset-name {
          flex: 1;
          font-weight: 600;
          color: var(--el-text-color-primary);
          font-size: 16px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        
        .dataset-points {
          font-size: 12px;
          color: var(--el-text-color-secondary);
          background: var(--el-color-info-light-9);
          padding: 2px 5px;
          border-radius: 6px;
          white-space: nowrap;
        }
      }
      
      .dataset-accuracy {
        margin: 6px 0;
        padding: 8px 12px;
        background: var(--el-color-info-light-9);
        border-radius: var(--el-border-radius-small);
        color: var(--el-color-info);
        font-size: 14px;
        text-align: center;
      }
      
      .dataset-classes {
        .classes-table {
          font-size: 14px;
          
          .classes-header,
          .classes-row {
            display: flex;
            padding: 9px 0;
            border-bottom: 1px solid var(--el-border-color-lighter);
            
            &:last-child {
              border-bottom: none;
            }
          }
          
          .classes-header {
            font-weight: bold;
            color: var(--el-text-color-regular);
            border-bottom-width: 2px;
          }
          
          .classes-row:hover {
            background: var(--el-fill-color-lighter);
          }
          
          .cell {
            padding: 0 6px;
            display: flex;
            align-items: center;
            
            &.class-name-cell {
              width: 140px;
              font-weight: 600;
            }
            
            &.class-count-cell {
              width: 80px;
              text-align: right;
            }
            
            &.class-percent-cell {
              width: 60px;
              text-align: right;
              color: var(--el-text-color-secondary);
            }
            
            &.class-meter-cell {
              width: 120px;
              display: flex;
              align-items: center;
              justify-content: center;
              
              .accuracy-meter {
                width: 110px;
                height: 10px;
                background: var(--el-fill-color-light);
                border-radius: 4px;
                overflow: hidden;
                
                .meter-fill {
                  height: 100%;
                  border-radius: 4px;
                  transition: width 0.3s ease;
                  
                  &.accuracy-high {
                    background: linear-gradient(90deg, var(--el-color-success-light-5), var(--el-color-success));
                  }
                  
                  &.accuracy-medium {
                    background: linear-gradient(90deg, var(--el-color-warning-light-5), var(--el-color-warning));
                  }
                  
                  &.accuracy-low {
                    background: linear-gradient(90deg, var(--el-color-danger-light-5), var(--el-color-danger));
                  }
                  
                  &:not(.accuracy-high):not(.accuracy-medium):not(.accuracy-low) {
                    background: var(--el-border-color);
                  }
                }
              }
            }
            
            &.class-acc-cell {
              width: 80px;
              text-align: right;
              font-weight: bold;
              font-size: 16px;
              
              &.accuracy-high {
                color: var(--el-color-success);
              }
              
              &.accuracy-medium {
                color: var(--el-color-warning);
              }
              
              &.accuracy-low {
                color: var(--el-color-danger);
              }
            }
          }
        }
      }
    }
  }
}

.predict-page__dataset-list,
.predict-page__note,
.predict-page__stats-panel .stats-content,
.predict-page__stats-panel .datasets-list,
:deep(.websocket-log-box__logs) {
  &::-webkit-scrollbar {
    width: 5px;
  }
  
  &::-webkit-scrollbar-track {
    background: var(--el-fill-color-light);
    border-radius: 2px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: var(--el-color-primary-light-5);
    border-radius: 2px;
    
    &:hover {
      background: var(--el-color-primary);
    }
  }
}
</style>