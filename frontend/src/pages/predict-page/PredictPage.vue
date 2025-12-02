<template>
  <div class="predict-page">
    <div class="predict-page__header">
      <el-icon size="32" color="var(--el-color-primary)">
        <MagicStick />
      </el-icon>
      <h1 class="predict-page__title">🔮 Предсказание DGCNN модели</h1>
      <div class="predict-page__subtitle">
        Выберите модель и датасеты для предсказания
      </div>
    </div>
    
    <div class="predict-page__container">
      <!-- Конфигурационная панель -->
      <el-card class="predict-page__config-panel" shadow="never">
        <template #header>
          <div class="predict-page__panel-header">
            <el-icon><Setting /></el-icon>
            <h3>⚙️ Конфигурация предсказания</h3>
          </div>
        </template>
        
        <el-form label-width="180px" label-position="left" size="small">
          <!-- Выбор модели -->
          <el-form-item label="Модель:" required>
            <el-select
              v-model="selectedModel"
              placeholder="Выберите модель..."
              filterable
              clearable
              style="width: 100%"
              :loading="loadingModels"
            >
              <el-option
                v-for="(model, index) in trainedModels"
                :key="index"
                :label="`${model.display_name || model.folder} (${model.size})`"
                :value="model.full_path"
              >
                <div class="model-option">
                  <el-icon size="16" color="var(--el-color-success)">
                    <Check />
                  </el-icon>
                  <span class="model-name">{{ model.display_name || model.folder }}</span>
                  <span class="model-size">{{ model.size }}</span>
                </div>
              </el-option>
              
              <el-option v-if="trainedModels.length === 0" disabled value="">
                <div class="no-models">
                  <el-icon><Warning /></el-icon>
                  <span>Нет обученных моделей</span>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
          
          <!-- Выбор датасетов -->
          <el-form-item label="Unlabeled датасеты:" required>
            <div class="datasets-selector">
              <div v-if="loadingDatasets" class="datasets-loading">
                <el-skeleton :rows="3" animated />
              </div>
              
              <div v-else-if="availableDatasets.length === 0" class="datasets-empty">
                <el-empty description="Нет доступных датасетов" :image-size="60" />
              </div>
              
              <div v-else class="datasets-list">
                <el-checkbox-group v-model="selectedDatasets">
                  <el-card
                    v-for="(dataset, index) in availableDatasets"
                    :key="index"
                    class="dataset-card"
                    shadow="never"
                  >
                    <el-checkbox :label="`datasets/raw/${dataset.file}`">
                      <div class="dataset-info">
                        <div class="dataset-header">
                          <span class="dataset-name">{{ dataset.name }}</span>
                          <el-tag size="small" type="info" effect="plain">
                            {{ dataset.points.toLocaleString() }} точек
                          </el-tag>
                        </div>
                        <div class="dataset-meta">
                          <span class="dataset-file">{{ dataset.file }}</span>
                          <span class="dataset-size">{{ dataset.size }}</span>
                        </div>
                      </div>
                    </el-checkbox>
                  </el-card>
                </el-checkbox-group>
                
                <div class="datasets-actions">
                  <el-button type="text" size="small" @click="selectAllDatasets">
                    <el-icon><Select /></el-icon>
                    Выбрать все
                  </el-button>
                  <el-button type="text" size="small" @click="deselectAllDatasets">
                    <el-icon><CircleClose /></el-icon>
                    Снять все
                  </el-button>
                </div>
              </div>
            </div>
          </el-form-item>
          
          <!-- Параметры предсказания -->
          <el-form-item label="Batch Size:" required>
            <el-input-number
              v-model="config.batch_size"
              :min="1"
              :max="32"
              :step="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
          
          <el-form-item label="Выходная папка:">
            <el-input
              v-model="outputDir"
              readonly
              placeholder="datasets/predicted"
            >
              <template #prepend>
                <el-icon><Folder /></el-icon>
              </template>
            </el-input>
          </el-form-item>
        </el-form>
        
        <!-- Кнопка запуска -->
        <div class="predict-page__actions">
          <el-button
            type="primary"
            :loading="isPredicting"
            :disabled="!selectedModel || selectedDatasets.length === 0 || loadingDatasets || loadingModels"
            @click="startPrediction"
            class="predict-page__predict-btn"
            size="large"
          >
            <template #icon>
              <el-icon><MagicStick /></el-icon>
            </template>
            {{ isPredicting ? 'Идет предсказание...' : '🔮 Запустить предсказание' }}
          </el-button>
        </div>
        
        <!-- Статистика -->
        <el-collapse v-model="activeStats" class="predict-page__stats-collapse">
          <el-collapse-item title="📊 Агрегированная статистика" name="stats">
            <div v-html="statsContent" class="stats-content"></div>
          </el-collapse-item>
        </el-collapse>
        
        <!-- Информационная панель -->
        <el-alert
          title="💡 Примечание"
          type="info"
          :closable="false"
          class="predict-page__info-box"
        >
          <template #default>
            <ul class="predict-page__info-list">
              <li>Используется обученная модель</li>
              <li>Обрабатываются unlabeled датасеты</li>
              <li>Результаты сохраняются в <strong>datasets/predicted/</strong></li>
              <li><strong>Новое:</strong> поддержка множественных датасетов!</li>
            </ul>
          </template>
        </el-alert>
      </el-card>
      
      <!-- Панель логов -->
      <el-card class="predict-page__logs-panel" shadow="never">
        <template #header>
          <div class="predict-page__panel-header">
            <el-icon><Monitor /></el-icon>
            <h3>📊 Логи предсказания</h3>
          </div>
        </template>
        
        <!-- Статус подключения -->
        <div class="predict-page__status">
          <el-alert
            :title="statusMessage"
            :type="wsConnected ? 'success' : isPredicting ? 'warning' : 'error'"
            :closable="false"
            :show-icon="true"
            :icon="getStatusIcon()"
            class="predict-page__status-alert"
          />
        </div>
        
        <!-- Логи -->
        <div class="predict-page__logs-container">
          <el-scrollbar height="500px">
            <pre class="predict-page__logs-content">{{ logs }}</pre>
          </el-scrollbar>
          
          <div v-if="!logs" class="predict-page__logs-empty">
            <el-empty description="Логи предсказания появятся здесь" :image-size="80" />
          </div>
        </div>
        
        <!-- Действия с логами -->
        <div class="predict-page__logs-actions">
          <el-button-group>
            <el-button
              type="primary"
              :icon="Download"
              size="small"
              @click="downloadLogs"
              :disabled="!logs"
            >
              Скачать логи
            </el-button>
            <el-button
              type="info"
              :icon="Delete"
              size="small"
              @click="clearLogs"
              :disabled="!logs"
            >
              Очистить
            </el-button>
          </el-button-group>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue';
import {
  MagicStick,
  Setting,
  Check,
  Folder,
  Monitor,
  Download,
  Delete,
  Warning,
  Select,
  CircleClose
} from '@element-plus/icons-vue';
import { ElMessage, ElNotification } from 'element-plus';

const ws = ref(null);
const isPredicting = ref(false);
const wsConnected = ref(false);
const loadingModels = ref(true);
const loadingDatasets = ref(true);
const statusMessage = ref('🔴 Ожидание подключения');
const logs = ref('');
const selectedModel = ref('');
const selectedDatasets = ref([]);
const availableDatasets = ref([]);
const trainedModels = ref([]);
const activeStats = ref([]);
const statsContent = ref('');
const outputDir = ref('datasets/predicted');
const config = ref({
  batch_size: 16
});

// Иконки статуса
const getStatusIcon = () => {
  if (wsConnected.value) return 'success';
  if (isPredicting.value) return 'warning';
  return 'error';
};

// Выбрать все датасеты
const selectAllDatasets = () => {
  selectedDatasets.value = availableDatasets.value.map(ds => `datasets/raw/${ds.file}`);
  ElMessage.success(`Выбрано ${selectedDatasets.value.length} датасетов`);
};

// Снять все датасеты
const deselectAllDatasets = () => {
  selectedDatasets.value = [];
  ElMessage.info('Выбор датасетов сброшен');
};

// Загрузка данных
const loadData = async () => {
  try {
    const [modelsRes, datasetsRes] = await Promise.all([
      fetch('/api/trained-models'),
      fetch('/api/datasets?subdir=raw') // Исправленный запрос
    ]);
    
    const modelsData = await modelsRes.json();
    const datasetsData = await datasetsRes.json();
    
    trainedModels.value = modelsData;
    availableDatasets.value = datasetsData.files || datasetsData.raw || [];
  } catch (error) {
    console.error('Ошибка загрузки данных:', error);
    ElMessage.error('Не удалось загрузить данные');
  } finally {
    loadingModels.value = false;
    loadingDatasets.value = false;
  }
};

// Запуск предсказания
const startPrediction = () => {
  if (isPredicting.value) {
    ElMessage.warning('Предсказание уже запущено!');
    return;
  }

  if (!selectedModel.value) {
    ElMessage.warning('Выберите модель для предсказания!');
    return;
  }

  if (selectedDatasets.value.length === 0) {
    ElMessage.warning('Выберите хотя бы один датасет для предсказания!');
    return;
  }

  const predictionConfig = {
    checkpoint_path: selectedModel.value,
    batch_size: config.value.batch_size,
    output_dir: outputDir.value
  };

  if (selectedDatasets.value.length === 1) {
    predictionConfig.input_file = selectedDatasets.value[0];
  } else {
    predictionConfig.input_files = selectedDatasets.value;
  }

  activeStats.value = [];
  connectWebSocket(predictionConfig);
};

// WebSocket соединение
const connectWebSocket = (config) => {
  statusMessage.value = '🟡 Подключаемся к WebSocket...';
  wsConnected.value = false;
  logs.value = '';

  ws.value = new WebSocket("ws://127.0.0.1:8000/api/ws/predict");

  ws.value.onopen = () => {
    statusMessage.value = '🟢 WebSocket подключён — запуск предсказания...';
    wsConnected.value = true;
    isPredicting.value = true;
    
    ws.value.send(JSON.stringify(config));
  };

  ws.value.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    } catch (error) {
      appendLog(`[${new Date().toLocaleTimeString()}] ${event.data}`, 'info');
    }
  };

  ws.value.onclose = () => {
    statusMessage.value = '🔴 Соединение закрыто';
    wsConnected.value = false;
    isPredicting.value = false;
  };

  ws.value.onerror = (error) => {
    statusMessage.value = '❌ Ошибка соединения';
    wsConnected.value = false;
    console.error('WebSocket error:', error);
    ElMessage.error('Ошибка подключения к серверу');
  };
};

// Обработка сообщений WebSocket
const handleWebSocketMessage = (data) => {
  const timestamp = new Date().toLocaleTimeString();

  switch (data.type) {
    case 'log':
      appendLog(`[${timestamp}] ${data.message}`, data.level);
      break;
    case 'result':
      handleResult(data.data);
      break;
    case 'error':
      appendLog(`[${timestamp}] ❌ ОШИБКА: ${data.data.error || data.data.message}`, 'error');
      ElNotification.error({
        title: 'Ошибка предсказания',
        message: data.data.error || data.data.message,
        duration: 5000
      });
      isPredicting.value = false;
      break;
    case 'connection':
      appendLog(`[${timestamp}] 🔗 ${data.message}`, 'info');
      break;
    default:
      appendLog(`[${timestamp}] ${JSON.stringify(data)}`, 'info');
  }
};

// Обработка результата
const handleResult = (result) => {
  if (result.success) {
    appendLog(`[${new Date().toLocaleTimeString()}] ✅ ${result.message}`, 'info');
    ElNotification.success({
      title: 'Предсказание завершено',
      message: result.message,
      duration: 5000
    });
    
    if (result.aggregated_statistics) {
      showAggregatedStats(result);
      activeStats.value = ['stats'];
    } else if (result.statistics) {
      showSingleStats(result);
      activeStats.value = ['stats'];
    }
  } else {
    appendLog(`[${new Date().toLocaleTimeString()}] ❌ ОШИБКА: ${result.message}`, 'error');
  }
  
  isPredicting.value = false;
};

// Показать статистику для одного датасета
const showSingleStats = (result) => {
  let statsHTML = '<div class="stats-item">📊 Статистика предсказаний:</div>';
  
  if (result.statistics && typeof result.statistics === 'object') {
    for (const [className, stat] of Object.entries(result.statistics)) {
      let statText = '';
      if (typeof stat === 'object' && stat.formatted) {
        statText = stat.formatted;
      } else if (typeof stat === 'string') {
        statText = stat;
      } else {
        statText = JSON.stringify(stat);
      }
      statsHTML += `<div class="stats-item">   ${className}: ${statText}</div>`;
    }
  } else {
    statsHTML += '<div class="stats-item">❌ Нет данных статистики</div>';
  }
  
  if (result.accuracy) {
    statsHTML += `
      <div class="stats-item" style="margin-top: 10px; font-weight: bold; color: var(--el-color-success);">
        🎯 Общая точность: ${result.accuracy.overall}
      </div>
      <div class="stats-item" style="font-weight: bold;">Точность по классам:</div>
    `;
    
    if (result.accuracy.per_class) {
      for (const [className, accuracy] of Object.entries(result.accuracy.per_class)) {
        statsHTML += `<div class="stats-item">   ${className}: ${accuracy}</div>`;
      }
    }
  }
  
  statsContent.value = statsHTML;
};

// Показать агрегированную статистику
const showAggregatedStats = (result) => {
  let statsHTML = `
    <div class="stats-item">✅ Обработано датасетов: ${result.datasets_processed}</div>
    <div class="stats-item">❌ Ошибок: ${result.datasets_failed}</div>
    <div class="stats-item">📊 Всего точек: ${result.total_points?.toLocaleString() || 0}</div>
  `;
  
  if (result.accuracy) {
    statsHTML += `
      <div class="stats-item" style="margin-top: 10px; font-weight: bold; color: var(--el-color-success);">
        🎯 Общая точность: ${result.accuracy.overall}
      </div>
      <div class="stats-item" style="font-weight: bold;">Точность по классам:</div>
    `;
    
    if (result.accuracy.per_class) {
      for (const [className, accuracy] of Object.entries(result.accuracy.per_class)) {
        statsHTML += `<div class="stats-item">   ${className}: ${accuracy}</div>`;
      }
    }
  }
  
  statsHTML += `<div class="stats-item" style="margin-top: 10px; font-weight: bold;">Распределение по классам:</div>`;
  
  if (result.aggregated_statistics && typeof result.aggregated_statistics === 'object') {
    for (const [className, stat] of Object.entries(result.aggregated_statistics)) {
      statsHTML += `<div class="stats-item">   ${className}: ${stat}</div>`;
    }
  }
  
  if (result.individual_results) {
    statsHTML += '<div class="stats-item" style="margin-top: 10px;"><strong>Детали по датасетам:</strong></div>';
    
    result.individual_results.forEach(dataset => {
      if (dataset.success) {
        const filename = dataset.input_file.split('/').pop();
        let accuracyText = '';
        if (dataset.accuracy) {
          accuracyText = ` - Точность: ${dataset.accuracy.overall}`;
        }
        statsHTML += `<div class="stats-item" style="padding-left: 20px; font-size: 12px;">✓ ${filename}: ${dataset.total_points} точек${accuracyText}</div>`;
      } else {
        statsHTML += `<div class="stats-item" style="padding-left: 20px; font-size: 12px; color: var(--el-color-danger);">✗ ${dataset.input_file}: Ошибка</div>`;
      }
    });
  }
  
  statsContent.value = statsHTML;
};

// Добавление логов
const appendLog = (message, level = 'info') => {
  logs.value += message + '\n';
};

// Загрузка логов
const downloadLogs = () => {
  if (!logs.value) return;
  
  const blob = new Blob([logs.value], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `prediction_logs_${new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  ElMessage.success('Логи успешно скачаны');
};

// Очистка логов
const clearLogs = () => {
  logs.value = '';
  ElMessage.info('Логи очищены');
};

// Закрытие WebSocket
const closeWebSocket = () => {
  if (ws.value) {
    ws.value.close();
    ws.value = null;
  }
};

// Жизненный цикл
onMounted(() => {
  loadData();
});

onUnmounted(() => {
  closeWebSocket();
});
</script>

<style lang="scss" scoped>
.predict-page {
  padding: 24px;
  background: var(--el-bg-color-page);
  min-height: 100vh;

  &__header {
    text-align: center;
    margin-bottom: 32px;
    
    .el-icon {
      margin-bottom: 16px;
      filter: drop-shadow(0 0 12px var(--el-color-primary-light-5));
    }
  }

  &__title {
    margin: 0 0 8px;
    font-size: 28px;
    font-weight: 700;
    color: var(--el-text-color-primary);
    background: linear-gradient(135deg, var(--el-color-primary), var(--el-color-primary-light-3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  &__subtitle {
    color: var(--el-text-color-secondary);
    font-size: 14px;
    margin-top: 8px;
  }

  &__container {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;

    @media (max-width: 1200px) {
      grid-template-columns: 1fr;
    }
  }

  &__config-panel,
  &__logs-panel {
    background: var(--el-bg-color);
    border: 1px solid var(--el-border-color-light);
    border-radius: var(--el-border-radius-base);
    
    :deep(.el-card__header) {
      background: var(--el-fill-color-lighter);
      border-bottom: 1px solid var(--el-border-color-light);
      padding: 16px 20px;
    }
  }

  &__panel-header {
    display: flex;
    align-items: center;
    gap: 12px;

    h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .el-icon {
      color: var(--el-color-primary);
      font-size: 20px;
    }
  }

  .model-option {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;

    .model-name {
      flex: 1;
      font-weight: 500;
      color: var(--el-text-color-primary);
    }

    .model-size {
      font-size: 12px;
      color: var(--el-text-color-placeholder);
    }
  }

  .no-models {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--el-color-warning);
  }

  .datasets-selector {
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    padding: 16px;
    background: var(--el-fill-color-light);
  }

  .datasets-loading {
    padding: 20px;
  }

  .datasets-empty {
    padding: 40px 20px;
  }

  .datasets-list {
    max-height: 400px;
    overflow-y: auto;
  }

  .dataset-card {
    margin-bottom: 8px;
    border: 1px solid var(--el-border-color-light);
    
    :deep(.el-card__body) {
      padding: 12px;
    }
    
    &:last-child {
      margin-bottom: 0;
    }
  }

  .dataset-info {
    flex: 1;
    
    .dataset-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      
      .dataset-name {
        font-weight: 600;
        color: var(--el-text-color-primary);
        font-size: 14px;
      }
    }
    
    .dataset-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: var(--el-text-color-secondary);
      
      .dataset-file {
        font-family: 'SF Mono', 'Consolas', monospace;
      }
      
      .dataset-size {
        color: var(--el-text-color-placeholder);
      }
    }
  }

  .datasets-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--el-border-color-light);
  }

  &__actions {
    margin-top: 24px;
  }

  &__predict-btn {
    width: 100%;
    font-size: 16px;
    height: 48px;
    
    :deep(.el-icon) {
      font-size: 18px;
    }
  }

  &__stats-collapse {
    margin-top: 24px;
    
    .stats-content {
      padding: 8px;
      
      .stats-item {
        margin-bottom: 6px;
        padding: 4px 8px;
        background: var(--el-fill-color-light);
        border-radius: var(--el-border-radius-base);
        font-size: 13px;
      }
    }
  }

  &__info-box {
    margin-top: 24px;
    background: var(--el-color-info-light-9);
    border: 1px solid var(--el-color-info-light-5);
    
    :deep(.el-alert__title) {
      color: var(--el-color-info);
    }
  }

  &__info-list {
    margin: 0;
    padding-left: 20px;
    color: var(--el-text-color-regular);

    li {
      margin-bottom: 6px;
      
      &:last-child {
        margin-bottom: 0;
      }
      
      strong {
        color: var(--el-text-color-primary);
        font-weight: 600;
      }
    }
  }

  &__status {
    margin-bottom: 20px;
  }

  &__status-alert {
    :deep(.el-alert__title) {
      font-weight: 500;
    }
  }

  &__logs-container {
    position: relative;
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    background: var(--el-fill-color-light);
    overflow: hidden;
  }

  &__logs-content {
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.5;
    padding: 16px;
    margin: 0;
    white-space: pre-wrap;
    color: var(--el-text-color-primary);
  }

  &__logs-empty {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  &__logs-actions {
    margin-top: 16px;
    display: flex;
    justify-content: flex-end;
  }
}

// Адаптивность
@media (max-width: 768px) {
  .predict-page {
    padding: 16px;
    
    &__title {
      font-size: 24px;
    }
    
    &__container {
      gap: 16px;
    }
    
    &__panel-header {
      h3 {
        font-size: 16px;
      }
    }
    
    :deep(.el-form-item__label) {
      width: 100% !important;
      text-align: left !important;
      margin-bottom: 8px;
    }
    
    :deep(.el-form-item__content) {
      margin-left: 0 !important;
    }
  }
}
</style>