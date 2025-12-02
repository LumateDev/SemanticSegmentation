```vue
<template>
  <div class="predict-page">
    <h1>🔮 Предсказание DGCNN модели</h1>
    
    <div class="predict-container">
      <div class="config-panel">
        <h3>⚙️ Конфигурация предсказания</h3>
        
        <div class="form-group">
          <label for="modelSelect">Модель:</label>
          <select id="modelSelect" v-model="selectedModel">
            <option value="">Загрузка моделей...</option>
            <option 
              v-for="(model, index) in trainedModels" 
              :key="index"
              :value="model.full_path"
            >
              {{ model.display_name || model.folder }} ({{ model.size }})
            </option>
          </select>
        </div>
        
        <div class="form-group">
          <label>Unlabeled датасеты:</label>
          <div id="datasetList" class="dataset-selector">
            <div v-if="loadingDatasets">Загрузка датасетов...</div>
            <div v-else-if="availableDatasets.length === 0">
              <p>Нет доступных датасетов.</p>
            </div>
            <div v-else>
              <div 
                v-for="(dataset, index) in availableDatasets" 
                :key="index" 
                class="dataset-item"
              >
                <input 
                  type="checkbox" 
                  :id="`dataset-${dataset.file}`"
                  :value="`datasets/raw/${dataset.file}`"
                  v-model="selectedDatasets"
                >
                <label :for="`dataset-${dataset.file}`">
                  {{ dataset.name }} ({{ dataset.points }} точек)
                </label>
              </div>
            </div>
          </div>
          <div style="margin-top: 10px;">
            <button type="button" @click="selectAllDatasets">Выбрать все</button>
            <button type="button" @click="deselectAllDatasets">Снять все</button>
          </div>
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
          <label for="outputDir">Выходная папка:</label>
          <input type="text" id="outputDir" value="datasets/predicted" readonly>
        </div>
        
        <button 
          class="predict-btn" 
          @click="startPrediction" 
          :disabled="isPredicting || loadingModels || loadingDatasets"
          id="predictBtn"
        >
          {{ isPredicting ? '🔮 Предсказание...' : '🔮 Запустить предсказание' }}
        </button>
        
        <div class="stats-panel" id="statsPanel" v-show="showStats">
          <div class="stats-header">📊 Агрегированная статистика:</div>
          <div id="aggregatedStats" v-html="statsContent"></div>
        </div>
        
        <div class="info-box">
          <small>
            <strong>💡 Примечание:</strong><br>
            • Используется обученная модель<br>
            • Обрабатываются unlabeled датасеты<br>
            • Результаты сохраняются в datasets/predicted/<br>
            • <strong>Новое:</strong> поддержка множественных датасетов!
          </small>
        </div>
      </div>
      
      <div class="logs-panel">
        <h3>📊 Логи предсказания</h3>
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
  name: 'PredictPage',
  data() {
    return {
      ws: null,
      isPredicting: false,
      wsConnected: false,
      loadingModels: true,
      loadingDatasets: true,
      statusMessage: '🔴 Ожидание подключения',
      logs: '',
      selectedModel: '',
      selectedDatasets: [],
      availableDatasets: [],
      trainedModels: [],
      showStats: false,
      statsContent: '',
      config: {
        batch_size: 16
      }
    }
  },
  async created() {
    await Promise.all([
      this.loadModels(),
      this.loadDatasets()
    ]);
  },
  beforeUnmount() {
    this.closeWebSocket();
  },
  methods: {
    async loadModels() {
      try {
        const response = await fetch('/api/trained-models');
        this.trainedModels = await response.json();
      } catch (error) {
        console.error('Ошибка загрузки моделей:', error);
      } finally {
        this.loadingModels = false;
      }
    },

    async loadDatasets() {
      try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();
        this.availableDatasets = datasets.unlabeled || datasets.raw || [];
      } catch (error) {
        console.error('Ошибка загрузки датасетов:', error);
      } finally {
        this.loadingDatasets = false;
      }
    },

    selectAllDatasets() {
      this.selectedDatasets = this.availableDatasets.map(ds => `datasets/raw/${ds.file}`);
    },

    deselectAllDatasets() {
      this.selectedDatasets = [];
    },

    startPrediction() {
      if (this.isPredicting) {
        alert('Предсказание уже запущено!');
        return;
      }

      if (!this.selectedModel) {
        alert('Выберите модель для предсказания!');
        return;
      }

      if (this.selectedDatasets.length === 0) {
        alert('Выберите хотя бы один датасет для предсказания!');
        return;
      }

      const config = {
        checkpoint_path: this.selectedModel,
        batch_size: this.config.batch_size,
        output_dir: 'datasets/predicted'
      };

      if (this.selectedDatasets.length === 1) {
        config.input_file = this.selectedDatasets[0];
      } else {
        config.input_files = this.selectedDatasets;
      }

      this.showStats = false;
      this.connectWebSocket(config);
    },

    connectWebSocket(config) {
      this.statusMessage = '🟡 Подключаемся к WebSocket...';
      this.wsConnected = false;
      this.logs = '';

      this.ws = new WebSocket("ws://127.0.0.1:8000/api/ws/predict");

      this.ws.onopen = () => {
        this.statusMessage = '🟢 WebSocket подключён — запуск предсказания...';
        this.wsConnected = true;
        this.isPredicting = true;
        
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
        this.isPredicting = false;
      };

      this.ws.onerror = (error) => {
        this.statusMessage = '❌ Ошибка соединения';
        this.wsConnected = false;
        console.error('WebSocket error:', error);
        this.isPredicting = false;
      };
    },

    handleWebSocketMessage(data) {
      const timestamp = new Date().toLocaleTimeString();

      switch (data.type) {
        case 'log':
          this.appendLog(`[${timestamp}] ${data.message}`, data.level);
          break;
        case 'result':
          this.handleResult(data.data);
          break;
        case 'error':
          this.appendLog(`[${timestamp}] ❌ ОШИБКА: ${data.data.error || data.data.message}`, 'error');
          this.isPredicting = false;
          break;
        case 'connection':
          this.appendLog(`[${timestamp}] 🔗 ${data.message}`, 'info');
          break;
        default:
          this.appendLog(`[${timestamp}] ${JSON.stringify(data)}`, 'info');
      }
    },

    handleResult(result) {
      if (result.success) {
        this.appendLog(`[${new Date().toLocaleTimeString()}] ✅ ${result.message}`, 'info');
        
        if (result.aggregated_statistics) {
          this.showAggregatedStats(result);
        } else if (result.statistics) {
          this.showSingleStats(result);
        }
      } else {
        this.appendLog(`[${new Date().toLocaleTimeString()}] ❌ ОШИБКА: ${result.message}`, 'error');
      }
      
      this.isPredicting = false;
    },

    showSingleStats(result) {
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
          <div class="stats-item" style="margin-top: 10px; font-weight: bold; color: #28a745;">
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
      
      this.statsContent = statsHTML;
      this.showStats = true;
    },

    showAggregatedStats(result) {
      let statsHTML = `
        <div class="stats-item">✅ Обработано датасетов: ${result.datasets_processed}</div>
        <div class="stats-item">❌ Ошибок: ${result.datasets_failed}</div>
        <div class="stats-item">📊 Всего точек: ${result.total_points?.toLocaleString() || 0}</div>
      `;
      
      if (result.accuracy) {
        statsHTML += `
          <div class="stats-item" style="margin-top: 10px; font-weight: bold; color: #28a745;">
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
            statsHTML += `<div class="stats-item" style="padding-left: 20px; font-size: 11px;">✓ ${filename}: ${dataset.total_points} точек${accuracyText}</div>`;
          } else {
            statsHTML += `<div class="stats-item" style="padding-left: 20px; font-size: 11px; color: #dc3545;">✗ ${dataset.input_file}: Ошибка</div>`;
          }
        });
      }
      
      this.statsContent = statsHTML;
      this.showStats = true;
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
.predict-page {
  font-family: Arial, sans-serif;
  margin: 20px;
}

h1 {
  color: #333;
  text-align: center;
}

.predict-container {
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

.predict-btn {
  background: #17a2b8;
  color: white;
  padding: 12px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  width: 100%;
  font-size: 16px;
}

.predict-btn:disabled {
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

.dataset-selector {
  margin-bottom: 10px;
}

.dataset-item {
  display: flex;
  align-items: center;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 5px;
  background: #f8f9fa;
}

.dataset-item input {
  width: auto;
  margin-right: 10px;
}

.dataset-item label {
  margin: 0;
  font-weight: normal;
  flex-grow: 1;
}

.stats-panel {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 5px;
}

.stats-header {
  font-weight: bold;
  margin-bottom: 10px;
  color: #333;
}

.stats-item {
  margin-bottom: 5px;
  padding: 5px;
  background: white;
  border-radius: 3px;
}

.info-box {
  margin-top: 20px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 5px;
  color: #333;
}

.info-box small {
  display: block;
}

/* Стили для темной темы */
@media (prefers-color-scheme: dark) {
  .info-box {
    color: #ffffff;
  }
  
  .predict-page {
    background-color: #121212;
    color: #ffffff;
  }
  
  .config-panel,
  .logs-panel {
    background: #1e1e1e;
    border-color: #333;
    color: #ffffff;
  }
  
  .dataset-item {
    background: #2d2d2d;
    border-color: #444;
  }
  
  .stats-panel {
    background: #2d2d2d;
  }
  
  .stats-header {
    color: #ffffff;
  }
  
  .stats-item {
    background: #1e1e1e;
  }
}

/* Для класса на body */
:global(body.dark-theme) .info-box {
  color: #ffffff;
}

:global(body.dark-theme) .predict-page {
  background-color: #121212;
  color: #ffffff;
}

:global(body.dark-theme) .config-panel,
:global(body.dark-theme) .logs-panel {
  background: #1e1e1e;
  border-color: #333;
  color: #ffffff;
}

:global(body.dark-theme) .dataset-item {
  background: #2d2d2d;
  border-color: #444;
}

:global(body.dark-theme) .stats-panel {
  background: #2d2d2d;
}

:global(body.dark-theme) .stats-header {
  color: #ffffff;
}

:global(body.dark-theme) .stats-item {
  background: #1e1e1e;
}
</style>