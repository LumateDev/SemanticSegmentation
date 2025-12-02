```vue
<template>
  <div class="training-page">
    <h1>🎯 Обучение DGCNN модели</h1>
    
    <div class="training-container">
      <div class="config-panel">
        <h3>⚙️ Конфигурация обучения</h3>
        
        <div class="form-group">
          <label for="modelName">Название модели:</label>
          <input 
            type="text" 
            id="modelName" 
            v-model="config.model_name"
            placeholder="Например: my_model_v1"
          >
          <div class="name-hint">Оставьте пустым для автоматического названия</div>
        </div>

        <div class="form-group">
          <label>Датасеты (.xyz):</label>
          <div id="datasetsContainer" class="datasets-container">
            <div v-if="loadingDatasets">Загрузка датасетов...</div>
            <div v-else-if="datasets.length === 0">
              <p>Нет доступных датасетов.</p>
            </div>
            <div v-else>
              <div 
                v-for="(dataset, index) in datasets" 
                :key="index" 
                class="dataset-item"
              >
                <label>
                  <input 
                    type="checkbox" 
                    v-model="selectedDatasets"
                    :value="dataset.file"
                    :id="'chk_' + index"
                  >
                  {{ dataset.name }} ({{ dataset.points }} точек)
                </label>
              </div>
            </div>
          </div>
        </div>

        <div class="form-group">
          <label for="checkpointFile">Чекпоинт для дообучения (опционально):</label>
          <select id="checkpointFile" v-model="selectedCheckpoint">
            <option value="">Не дообучать (новая модель)</option>
            <option 
              v-for="(model, index) in trainedModels" 
              :key="index"
              :value="model.full_path"
            >
              {{ model.display_name }} ({{ model.modified }})
            </option>
          </select>
        </div>
        
        <div class="form-group">
          <label for="batchSize">Batch Size:</label>
          <input 
            type="number" 
            id="batchSize" 
            v-model.number="config.batch_size"
            min="1" 
            max="32"
          >
        </div>
        
        <div class="form-group">
          <label for="epochs">Количество эпох:</label>
          <input 
            type="number" 
            id="epochs" 
            v-model.number="config.epochs"
            min="1" 
            max="100"
          >
        </div>
        
        <div class="form-group">
          <label for="learningRate">Learning Rate:</label>
          <input 
            type="number" 
            id="learningRate" 
            v-model.number="config.learning_rate"
            step="0.0001" 
            min="0.0001" 
            max="0.1"
          >
        </div>
        
        <button 
          class="train-btn" 
          @click="startTraining" 
          :disabled="isTraining || loadingDatasets"
          id="trainBtn"
        >
          {{ isTraining ? '🔄 Обучение...' : '🚀 Начать обучение' }}
        </button>
        
        <div class="info-box">
          <small>
            <strong>💡 Примечание:</strong><br>
            • Модель сохраняется в checkpoints/DGCNN/[название_модели]/<br>
            • Автоматическое имя: DGCNN_YYYYMMDD_HHMMSS<br>
            • Ручное имя: ваше_название_YYYYMMDD_HHMMSS
          </small>
        </div>
      </div>
      
      <div class="logs-panel">
        <h3>📊 Логи обучения</h3>
        <div 
          id="status" 
          class="status" 
          :class="{
            'connected': wsConnected,
            'disconnected': !wsConnected
          }"
        >
          {{ statusMessage }}
        </div>
        <pre id="logs">{{ logs }}</pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TrainingPage',
  data() {
    return {
      ws: null,
      isTraining: false,
      wsConnected: false,
      loadingDatasets: true,
      statusMessage: '🔴 Ожидание подключения',
      logs: '',
      datasets: [],
      trainedModels: [],
      selectedDatasets: [],
      selectedCheckpoint: '',
      config: {
        model_name: '',
        batch_size: 8,
        epochs: 3,
        learning_rate: 0.001
      }
    }
  },
  async created() {
    await Promise.all([
      this.loadDatasets(),
      this.loadCheckpoints()
    ]);
  },
  beforeUnmount() {
    this.closeWebSocket();
  },
  methods: {
    async loadDatasets() {
      try {
        const response = await fetch('/api/datasets');
        const data = await response.json();
        this.datasets = data.raw || [];
      } catch (error) {
        console.error('Ошибка загрузки датасетов:', error);
      } finally {
        this.loadingDatasets = false;
      }
    },

    async loadCheckpoints() {
      try {
        const response = await fetch('/api/trained-models');
        this.trainedModels = await response.json();
      } catch (error) {
        console.error('Ошибка загрузки чекпоинтов:', error);
      }
    },

    startTraining() {
      if (this.isTraining) {
        alert('Обучение уже запущено!');
        return;
      }

      if (this.selectedDatasets.length === 0) {
        alert('Выберите хотя бы один датасет для обучения!');
        return;
      }

      const trainingConfig = {
        ...this.config,
        xyz_files: this.selectedDatasets.map(file => `datasets/raw/${file}`),
        resume_from: this.selectedCheckpoint || null
      };

      this.connectWebSocket(trainingConfig);
    },

    connectWebSocket(config) {
      this.statusMessage = '🟡 Подключаемся к WebSocket...';
      this.wsConnected = false;
      this.logs = '';

      this.ws = new WebSocket("ws://127.0.0.1:8000/api/ws/train-model");

      this.ws.onopen = () => {
        this.statusMessage = '🟢 WebSocket подключён — запуск обучения...';
        this.wsConnected = true;
        this.isTraining = true;
        
        this.ws.send(JSON.stringify(config));
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleWebSocketMessage(data);
        } catch (error) {
          this.appendLog(`[${new Date().toLocaleTimeString()}] ${event.data}`, 'info');
        }
      };

      this.ws.onclose = () => {
        this.statusMessage = '🔴 Соединение закрыто';
        this.wsConnected = false;
        this.isTraining = false;
      };

      this.ws.onerror = (error) => {
        this.statusMessage = '❌ Ошибка соединения';
        this.wsConnected = false;
        console.error('WebSocket error:', error);
      };
    },

    handleWebSocketMessage(data) {
      const timestamp = new Date().toLocaleTimeString();

      switch (data.type) {
        case 'log':
          this.appendLog(`[${timestamp}] ${data.message}`, data.level);
          break;
        case 'result':
          this.appendLog(`[${timestamp}] ✅ Результат: ${data.data.message}`, 'info');
          this.isTraining = false;
          break;
        case 'error':
          this.appendLog(`[${timestamp}] ❌ ОШИБКА: ${data.data.error || data.data.message}`, 'error');
          this.isTraining = false;
          break;
        case 'connection':
          this.appendLog(`[${timestamp}] 🔗 ${data.message}`, 'info');
          break;
        default:
          this.appendLog(`[${timestamp}] ${JSON.stringify(data)}`, 'info');
      }
    },

    appendLog(message, level = 'info') {
      this.logs += message + '\n';
      setTimeout(() => {
        const logsElement = document.getElementById('logs');
        if (logsElement) {
          logsElement.scrollTop = logsElement.scrollHeight;
        }
      }, 10);
    },

    closeWebSocket() {
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    }
  }
}
</script>

<style scoped>
.training-page {
  font-family: Arial, sans-serif;
  margin: 20px;
}

h1 {
  color: #333;
  text-align: center;
}

.training-container {
  display: flex;
  gap: 20px;
}

.config-panel {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 5px;
  padding: 20px;
}

.logs-panel {
  flex: 2;
  border: 1px solid #ddd;
  border-radius: 5px;
  padding: 20px;
}

.form-group {
  margin-bottom: 15px;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

input, select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.train-btn {
  background: #28a745;
  color: white;
  padding: 12px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  width: 100%;
  font-size: 16px;
}

.train-btn:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

#logs {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 15px;
  height: 500px;
  overflow-y: scroll;
  border-radius: 5px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.3;
  white-space: pre-wrap;
}

.status {
  padding: 10px;
  margin-bottom: 10px;
  border-radius: 5px;
}

.status.connected {
  background: #d4edda;
  color: #155724;
}

.status.disconnected {
  background: #f8d7da;
  color: #721c24;
}

.name-hint {
  font-size: 12px;
  color: #666;
  margin-top: 5px;
}

.datasets-container {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 10px;
  border-radius: 4px;
}

.dataset-item {
  margin-bottom: 5px;
}

.info-box {
  margin-top: 20px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 5px;
  color: #333; /* Явно задаем цвет текста для светлой темы */
}

.info-box small {
  display: block;
}

/* Только исправление для темной темы */
@media (prefers-color-scheme: dark) {
  .info-box {
    color: #ffffff; /* Белый текст в темной теме */
  }
}

/* Или если есть класс на body */
:global(body.dark-theme) .info-box {
  color: #ffffff;
}
</style>
```