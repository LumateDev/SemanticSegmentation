<template>
  <div class="test-page">
    <div class="test-page__header">
      <h1 class="test-page__title">
        <el-icon class="test-page__title-icon"><Cpu /></el-icon>
        Тестирование модели DGCNN
      </h1>
      <p class="test-page__description">Запустите тестирование работоспособности модели на устройстве</p>
    </div>

    <div class="test-page__content">
      <!-- Control Panel -->
      <el-card class="test-page__control-panel" shadow="never">
        <template #header>
          <div class="test-page__card-header">
            <span>Панель управления</span>
            <el-tag :type="tagType" effect="dark">
              {{ connectionStatusText }}
            </el-tag>
          </div>
        </template>

        <div class="test-page__controls">
          <el-button
            :type="isConnected ? 'danger' : 'primary'"
            :icon="isConnected ? Close : Connection"
            :loading="isConnecting"
            size="large"
            @click="toggleConnection"
          >
            {{ isConnected ? 'Отключиться' : 'Подключиться' }}
          </el-button>

          <el-button
            type="success"
            :icon="VideoPlay"
            size="large"
            :disabled="!isConnected || isTestRunning"
            :loading="isTestRunning"
            @click="startTest"
          >
            {{ isTestRunning ? 'Тестирование...' : 'Запустить тест' }}
          </el-button>

          <el-button :icon="Delete" size="large" :disabled="logs.length === 0" @click="clearLogs">
            Очистить логи
          </el-button>
        </div>

        <!-- Connection Info -->
        <el-collapse-transition>
          <div v-if="isConnected" class="test-page__connection-info">
            <el-descriptions :column="2" size="small" border>
              <el-descriptions-item label="WebSocket URL">
                {{ wsUrl }}
              </el-descriptions-item>
              <el-descriptions-item label="Время подключения">
                {{ connectionTime }}
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </el-collapse-transition>
      </el-card>

      <!-- Logs Section -->
      <div class="test-page__logs-section">
        <WebSocketLogBox ref="logBoxRef" title="Логи тестирования" :logs="logs" :status="status" @clear="clearLogs" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';
import { Delete, VideoPlay, Connection, Close, Cpu } from '@element-plus/icons-vue';
import { useWebSocket, type WebSocketStatus } from '@/composables/useWebSocket';
import { useWebSocketHandlers, type LogEntry } from '@/composables/useWebSocketHandlers';
import WebSocketLogBox from '@/components/WebSocketLogBox.vue';

const wsBaseUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://127.0.0.1:8000';
const wsUrl = `${wsBaseUrl}/api/ws/test-model`;
const logBoxRef = ref<InstanceType<typeof WebSocketLogBox> | null>(null);
const logs = ref<LogEntry[]>([]);
const isTestRunning = ref(false);
const connectionTime = ref<string>('');
let connectionStartTime: Date | null = null;
let connectionTimer: ReturnType<typeof setInterval> | null = null;
const { ws, isConnected, isConnecting, status, connect, disconnect } = useWebSocket({
  url: wsUrl,
  onOpen: () => {
    ElMessage.success('WebSocket подключён');
    connectionStartTime = new Date();
    startConnectionTimer();
    addLog('🔗 WebSocket соединение установлено', 'info');
  },
  onClose: () => {
    ElMessage.warning('WebSocket отключён');
    stopConnectionTimer();
    connectionTime.value = '';
    isTestRunning.value = false;
    addLog('🔌 WebSocket соединение закрыто', 'warning');
  },
  onError: () => {
    ElMessage.error('Ошибка WebSocket соединения');
    isTestRunning.value = false;
    addLog('❌ Ошибка WebSocket соединения', 'error');
  },
});

const { addLog, handleWebSocketMessage: baseHandleMessage } = useWebSocketHandlers(
  logs,
  computed(() => logBoxRef.value?.logsContainerRef ?? null)
);

const handleWebSocketMessage = (event: MessageEvent) => {
  try {
    const data = JSON.parse(event.data);

    // Проверяем на завершение теста
    if (data.type === 'result' || data.type === 'error') {
      isTestRunning.value = false;
    }
  } catch {
    // ignore parse errors
  }

  baseHandleMessage(event);
};

watch(ws, newWs => {
  if (newWs) {
    newWs.onmessage = handleWebSocketMessage;
  }
});

const tagType = computed<'success' | 'danger' | 'warning' | 'info'>(() => {
  const typeMap: Record<WebSocketStatus, 'success' | 'danger' | 'warning' | 'info'> = {
    connected: 'success',
    connecting: 'warning',
    disconnected: 'danger',
    error: 'danger',
  };
  return typeMap[status.value];
});

const connectionStatusText = computed(() => {
  const textMap: Record<WebSocketStatus, string> = {
    connected: '🟢 Подключено',
    connecting: '🟡 Подключение...',
    disconnected: '🔴 Отключено',
    error: '❌ Ошибка',
  };
  return textMap[status.value];
});

const startConnectionTimer = () => {
  stopConnectionTimer();
  updateConnectionTime();
  connectionTimer = setInterval(updateConnectionTime, 1000);
};

const stopConnectionTimer = () => {
  if (connectionTimer) {
    clearInterval(connectionTimer);
    connectionTimer = null;
  }
};

const updateConnectionTime = () => {
  if (!connectionStartTime) return;

  const diff = Math.floor((Date.now() - connectionStartTime.getTime()) / 1000);
  const hours = Math.floor(diff / 3600);
  const minutes = Math.floor((diff % 3600) / 60);
  const seconds = diff % 60;

  const parts: string[] = [];
  if (hours > 0) parts.push(`${hours}ч`);
  if (minutes > 0) parts.push(`${minutes}м`);
  parts.push(`${seconds}с`);

  connectionTime.value = parts.join(' ');
};

const toggleConnection = () => {
  if (isConnected.value) {
    disconnect();
  } else {
    connect();
  }
};

const startTest = () => {
  if (!isConnected.value) {
    ElMessage.warning('Сначала подключитесь к WebSocket');
    return;
  }

  if (!ws.value) {
    ElMessage.error('WebSocket не инициализирован');
    return;
  }

  try {
    isTestRunning.value = true;
    ws.value.send(JSON.stringify({}));
    addLog('🚀 Запущено тестирование модели...', 'info');
  } catch (error) {
    isTestRunning.value = false;
    addLog(`❌ Ошибка запуска теста: ${error}`, 'error');
    ElMessage.error('Не удалось запустить тест');
  }
};

const clearLogs = () => {
  logs.value = [];
};

onUnmounted(() => {
  disconnect();
  stopConnectionTimer();
});
</script>

<style lang="scss" scoped>
.test-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 24px 24px 0 24px;
  background: var(--el-bg-color-page);
  overflow: hidden;

  &__header {
    margin-bottom: 24px;
    flex-shrink: 0;
  }

  &__title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 8px 0;
    font-size: 28px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  &__title-icon {
    color: var(--el-color-primary);
  }

  &__description {
    margin: 0;
    color: var(--el-text-color-secondary);
    font-size: 14px;
  }

  &__content {
    display: flex;
    flex-direction: column;
    flex: 1;
    gap: 20px;
    min-height: 0;
    overflow: hidden;
    padding-bottom: 24px;
  }

  &__control-panel {
    flex-shrink: 0;

    :deep(.el-card__header) {
      padding: 16px 20px;
      background: var(--el-fill-color-light);
    }

    :deep(.el-card__body) {
      padding: 20px;
    }
  }

  &__card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 500;
  }

  &__controls {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }

  &__connection-info {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--el-border-color-lighter);
  }

  &__logs-section {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
}

@media (max-width: 768px) {
  .test-page {
    padding: 16px;

    &__title {
      font-size: 22px;
    }

    &__controls {
      flex-direction: column;

      .el-button {
        width: 100%;
      }
    }
  }
}
</style>
