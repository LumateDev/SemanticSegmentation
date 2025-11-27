import { nextTick, type Ref, type ComputedRef } from 'vue';

export interface LogEntry {
  timestamp: string;
  message: string;
  level: 'info' | 'warning' | 'error' | 'debug';
}

export function useWebSocketHandlers(
  logs: Ref<LogEntry[]>,
  logsContainerRef: Ref<HTMLDivElement | null> | ComputedRef<HTMLDivElement | null>
) {
  const addLog = (message: string, level: LogEntry['level'] = 'info') => {
    const timestamp = new Date().toLocaleTimeString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });

    logs.value.push({ timestamp, message, level });

    // Автопрокрутка вниз
    nextTick(() => {
      const container = logsContainerRef.value;
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    });
  };

  const handleWebSocketMessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      const time = data.timestamp
        ? new Date(data.timestamp).toLocaleTimeString('ru-RU', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          })
        : new Date().toLocaleTimeString('ru-RU', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          });

      if (data.type === 'log') {
        addLog(data.message, data.level || 'info');
      } else if (data.type === 'result') {
        addLog(`✅ Результат: ${data.data?.message || JSON.stringify(data.data)}`, 'info');
      } else if (data.type === 'error') {
        addLog(`❌ ОШИБКА: ${data.data?.error || data.error || 'Неизвестная ошибка'}`, 'error');
      } else if (data.message) {
        addLog(data.message, data.level || 'info');
      } else {
        addLog(event.data, 'info');
      }
    } catch {
      // Если не JSON, просто выводим как текст
      addLog(event.data, 'info');
    }
  };

  const clearLogs = () => {
    logs.value = [];
  };

  return {
    addLog,
    handleWebSocketMessage,
    clearLogs,
  };
}
