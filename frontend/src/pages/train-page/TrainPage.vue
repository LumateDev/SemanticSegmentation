<template>
  <div class="training-page">
    <div class="training-page__header">
      <el-icon size="32" color="var(--el-color-primary)">
        <MagicStick />
      </el-icon>
      <h1 class="training-page__title">🎯 Обучение DGCNN модели</h1>
      <div class="training-page__subtitle">
        Настройте параметры и запустите процесс обучения модели
      </div>
    </div>
    
    <div class="training-page__container">
      <!-- Конфигурационная панель -->
      <el-card class="training-page__config-panel" shadow="never">
        <template #header>
          <div class="training-page__panel-header">
            <el-icon><Setting /></el-icon>
            <h3>⚙️ Конфигурация обучения</h3>
          </div>
        </template>
        
        <el-form label-width="160px" label-position="left" size="small">
          <!-- Название модели -->
          <el-form-item label="Название модели:" required>
            <el-input
              v-model="config.model_name"
              placeholder="Например: my_model_v1"
              clearable
            />
            <div class="training-page__hint">
              <el-icon size="12"><InfoFilled /></el-icon>
              <span>Оставьте пустым для автоматического названия</span>
            </div>
          </el-form-item>
          
          <!-- Выбор датасетов -->
          <el-form-item label="Датасеты (.xyz):" required>
            <div class="training-page__datasets-container">
              <div v-if="loadingDatasets" class="training-page__loading">
                <el-skeleton :rows="3" animated />
              </div>
              <div v-else-if="datasets.length === 0" class="training-page__empty">
                <el-empty description="Нет доступных датасетов" :image-size="60" />
              </div>
              <div v-else class="training-page__datasets-list">
                <el-checkbox-group v-model="selectedDatasets">
                  <div 
                    v-for="(dataset, index) in datasets" 
                    :key="index" 
                    class="training-page__dataset-item"
                  >
                    <el-checkbox :label="dataset.file" size="large">
                      <div class="dataset-info">
                        <div class="dataset-name">{{ dataset.name }}</div>
                        <div class="dataset-meta">
                          <el-tag size="small" type="info" effect="plain">
                            <el-icon size="12"><Collection /></el-icon>
                            {{ dataset.points }} точек
                          </el-tag>
                        </div>
                      </div>
                    </el-checkbox>
                  </div>
                </el-checkbox-group>
                <div class="training-page__datasets-actions">
                  <el-button type="text" size="small" @click="selectAllDatasets">
                    Выбрать все
                  </el-button>
                  <el-button type="text" size="small" @click="deselectAllDatasets">
                    Снять все
                  </el-button>
                </div>
              </div>
            </div>
          </el-form-item>
          
          <!-- Чекпоинт для дообучения -->
          <el-form-item label="Чекпоинт для дообучения:">
            <el-select
              v-model="selectedCheckpoint"
              placeholder="Не дообучать (новая модель)"
              clearable
              filterable
              style="width: 100%"
            >
              <el-option 
                v-for="(model, index) in trainedModels" 
                :key="index"
                :label="`${model.display_name} (${model.modified})`"
                :value="model.full_path"
              >
                <div class="checkpoint-option">
                  <el-icon size="16" color="var(--el-color-success)">
                    <Check />
                  </el-icon>
                  <span class="checkpoint-name">{{ model.display_name }}</span>
                  <span class="checkpoint-date">{{ model.modified }}</span>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
          
          <!-- Параметры обучения -->
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
          
          <el-form-item label="Количество эпох:" required>
            <el-input-number
              v-model="config.epochs"
              :min="1"
              :max="100"
              :step="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
          
          <el-form-item label="Learning Rate:" required>
            <el-input-number
              v-model="config.learning_rate"
              :min="0.0001"
              :max="0.1"
              :step="0.0001"
              :precision="4"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-form>
        
        <!-- Кнопка запуска обучения -->
        <div class="training-page__actions">
          <el-button
            type="primary"
            :loading="isTraining"
            :disabled="selectedDatasets.length === 0 || loadingDatasets"
            @click="startTraining"
            class="training-page__train-btn"
            size="large"
          >
            <template #icon>
              <el-icon><VideoPlay /></el-icon>
            </template>
            {{ isTraining ? 'Идет обучение...' : '🚀 Начать обучение' }}
          </el-button>
        </div>
        
        <!-- Информационная панель -->
        <el-alert
          title="💡 Примечание"
          type="info"
          :closable="false"
          class="training-page__info-box"
        >
          <template #default>
            <ul class="training-page__info-list">
              <li>Модель сохраняется в <strong>checkpoints/DGCNN/[название_модели]/</strong></li>
              <li>Автоматическое имя: <strong>DGCNN_YYYYMMDD_HHMMSS</strong></li>
              <li>Ручное имя: <strong>ваше_название_YYYYMMDD_HHMMSS</strong></li>
            </ul>
          </template>
        </el-alert>
      </el-card>
      
      <!-- Панель логов -->
      <el-card class="training-page__logs-panel" shadow="never">
        <template #header>
          <div class="training-page__panel-header">
            <el-icon><Monitor /></el-icon>
            <h3>📊 Логи обучения</h3>
          </div>
        </template>
        
        <!-- Статус подключения -->
        <div class="training-page__status">
          <el-alert
            :title="statusMessage"
            :type="wsConnected ? 'success' : isTraining ? 'warning' : 'error'"
            :closable="false"
            :show-icon="true"
            :icon="getStatusIcon()"
            class="training-page__status-alert"
          />
        </div>
        
        <!-- Логи -->
        <div class="training-page__logs-container">
          <el-scrollbar height="500px">
            <pre class="training-page__logs-content">{{ logs }}</pre>
          </el-scrollbar>
          <div v-if="!logs" class="training-page__logs-empty">
            <el-empty description="Логи обучения появятся здесь" :image-size="80" />
          </div>
        </div>
        
        <!-- Действия с логами -->
        <div class="training-page__logs-actions">
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
  InfoFilled,
  Collection,
  Check,
  VideoPlay,
  Monitor,
  Download,
  Delete
} from '@element-plus/icons-vue';
import { ElMessage, ElNotification } from 'element-plus';

const ws = ref(null);
const isTraining = ref(false);
const wsConnected = ref(false);
const loadingDatasets = ref(true);
const statusMessage = ref('🔴 Ожидание подключения');
const logs = ref('');
const datasets = ref([]);
const trainedModels = ref([]);
const selectedDatasets = ref([]);
const selectedCheckpoint = ref('');
const config = ref({
  model_name: '',
  batch_size: 8,
  epochs: 3,
  learning_rate: 0.001
});

// Иконки статуса
const getStatusIcon = () => {
  if (wsConnected.value) return 'success';
  if (isTraining.value) return 'warning';
  return 'error';
};

// Выбрать все датасеты
const selectAllDatasets = () => {
  selectedDatasets.value = datasets.value.map(ds => ds.file);
};

// Снять все датасеты
const deselectAllDatasets = () => {
  selectedDatasets.value = [];
};

// Загрузка данных
const loadData = async () => {
  try {
    const [datasetsRes, modelsRes] = await Promise.all([
      fetch('/api/datasets'),
      fetch('/api/trained-models')
    ]);
    
    const datasetsData = await datasetsRes.json();
    const modelsData = await modelsRes.json();
    
    datasets.value = datasetsData.raw || [];
    trainedModels.value = modelsData;
  } catch (error) {
    console.error('Ошибка загрузки данных:', error);
    ElMessage.error('Не удалось загрузить данные');
  } finally {
    loadingDatasets.value = false;
  }
};

// Запуск обучения
const startTraining = () => {
  if (isTraining.value) {
    ElMessage.warning('Обучение уже запущено!');
    return;
  }

  if (selectedDatasets.value.length === 0) {
    ElMessage.warning('Выберите хотя бы один датасет для обучения!');
    return;
  }

  const trainingConfig = {
    ...config.value,
    xyz_files: selectedDatasets.value.map(file => `datasets/raw/${file}`),
    resume_from: selectedCheckpoint.value || null
  };

  connectWebSocket(trainingConfig);
};

// WebSocket соединение
const connectWebSocket = (config) => {
  statusMessage.value = '🟡 Подключаемся к WebSocket...';
  wsConnected.value = false;
  logs.value = '';

  ws.value = new WebSocket("ws://127.0.0.1:8000/api/ws/train-model");

  ws.value.onopen = () => {
    statusMessage.value = '🟢 WebSocket подключён — запуск обучения...';
    wsConnected.value = true;
    isTraining.value = true;
    
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
    isTraining.value = false;
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
      appendLog(`[${timestamp}] ✅ Результат: ${data.data.message}`, 'info');
      ElNotification.success({
        title: 'Обучение завершено',
        message: data.data.message,
        duration: 5000
      });
      isTraining.value = false;
      break;
    case 'error':
      appendLog(`[${timestamp}] ❌ ОШИБКА: ${data.data.error || data.data.message}`, 'error');
      ElNotification.error({
        title: 'Ошибка обучения',
        message: data.data.error || data.data.message,
        duration: 5000
      });
      isTraining.value = false;
      break;
    case 'connection':
      appendLog(`[${timestamp}] 🔗 ${data.message}`, 'info');
      break;
    default:
      appendLog(`[${timestamp}] ${JSON.stringify(data)}`, 'info');
  }
};

// Добавление логов
const appendLog = (message, level = 'info') => {
  const timestamp = new Date().toLocaleTimeString();
  const coloredMessage = `[${timestamp}] ${message}`;
  logs.value += coloredMessage + '\n';
};

// Загрузка логов
const downloadLogs = () => {
  if (!logs.value) return;
  
  const blob = new Blob([logs.value], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `training_logs_${new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')}.txt`;
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
.training-page {
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

  &__hint {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--el-text-color-placeholder);

    .el-icon {
      font-size: 12px;
    }
  }

  &__datasets-container {
    border: 1px solid var(--el-border-color);
    border-radius: var(--el-border-radius-base);
    padding: 12px;
    background: var(--el-fill-color-light);
  }

  &__loading {
    padding: 20px;
  }

  &__empty {
    padding: 40px 20px;
  }

  &__datasets-list {
    max-height: 300px;
    overflow-y: auto;
  }

  &__dataset-item {
    padding: 8px 4px;
    border-bottom: 1px solid var(--el-border-color-light);

    &:last-child {
      border-bottom: none;
    }

    .dataset-info {
      flex: 1;
      
      .dataset-name {
        font-weight: 500;
        color: var(--el-text-color-primary);
        margin-bottom: 4px;
      }

      .dataset-meta {
        .el-tag {
          background: var(--el-fill-color);
          border-color: var(--el-border-color);
          color: var(--el-text-color-secondary);
        }
      }
    }
  }

  &__datasets-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--el-border-color-light);
  }

  .checkpoint-option {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;

    .checkpoint-name {
      flex: 1;
      font-weight: 500;
      color: var(--el-text-color-primary);
    }

    .checkpoint-date {
      font-size: 12px;
      color: var(--el-text-color-placeholder);
    }
  }

  &__actions {
    margin-top: 24px;
  }

  &__train-btn {
    width: 100%;
    font-size: 16px;
    height: 48px;
    
    :deep(.el-icon) {
      font-size: 18px;
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

// Темная тема специфичные стили
@media (prefers-color-scheme: dark) {
  .training-page {
    &__logs-content {
      background: #1a1a1a;
      color: #e0e0e0;
    }
  }
}

// Адаптивность
@media (max-width: 768px) {
  .training-page {
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
  }
}
</style>