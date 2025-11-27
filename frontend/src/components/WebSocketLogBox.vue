<template>
  <div class="ws-log-box">
    <div class="ws-log-box__header">
      <div class="ws-log-box__title">
        <el-icon class="ws-log-box__icon">
          <Monitor />
        </el-icon>
        <span>{{ title }}</span>
      </div>
      <div class="ws-log-box__actions">
        <el-tag :type="tagType" size="small" class="ws-log-box__status">
          <span class="ws-log-box__status-dot" :class="`ws-log-box__status-dot--${status}`" />
          {{ statusText }}
        </el-tag>
        <el-button :icon="Delete" size="small" text :disabled="logs.length === 0" @click="$emit('clear')">
          Очистить
        </el-button>
      </div>
    </div>

    <div ref="logsContainerRef" class="ws-log-box__content">
      <div v-if="logs.length === 0" class="ws-log-box__empty">
        <el-icon :size="48" color="#5c6370">
          <Document />
        </el-icon>
        <p>Логи появятся здесь после подключения</p>
      </div>

      <div v-for="(log, index) in logs" :key="index" class="ws-log-box__line" :class="`ws-log-box__line--${log.level}`">
        <span class="ws-log-box__timestamp">[{{ log.timestamp }}]</span>
        <span class="ws-log-box__message">{{ log.message }}</span>
      </div>
    </div>

    <div v-if="showFooter" class="ws-log-box__footer">
      <span class="ws-log-box__count">
        {{ logs.length }} {{ pluralize(logs.length, 'запись', 'записи', 'записей') }}
      </span>
      <el-button size="small" text @click="scrollToBottom">
        <el-icon><Bottom /></el-icon>
        В конец
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue';
import { Delete, Monitor, Document, Bottom } from '@element-plus/icons-vue';
import type { LogEntry } from '@/composables/useWebSocketHandlers';
import type { WebSocketStatus } from '@/composables/useWebSocket';

interface Props {
  title?: string;
  logs: LogEntry[];
  status: WebSocketStatus;
  showFooter?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Логи',
  showFooter: true,
});

defineEmits<{
  clear: [];
}>();

const logsContainerRef = ref<HTMLDivElement | null>(null);

const tagType = computed(() => {
  const typeMap: Record<WebSocketStatus, 'success' | 'danger' | 'warning' | 'info'> = {
    connected: 'success',
    connecting: 'warning',
    disconnected: 'danger',
    error: 'danger',
  };
  return typeMap[props.status];
});

const statusText = computed(() => {
  const textMap: Record<WebSocketStatus, string> = {
    connected: 'Подключено',
    connecting: 'Подключение...',
    disconnected: 'Отключено',
    error: 'Ошибка',
  };
  return textMap[props.status];
});

const pluralize = (count: number, one: string, few: string, many: string): string => {
  const mod10 = count % 10;
  const mod100 = count % 100;

  if (mod100 >= 11 && mod100 <= 19) return many;
  if (mod10 === 1) return one;
  if (mod10 >= 2 && mod10 <= 4) return few;
  return many;
};

const scrollToBottom = () => {
  nextTick(() => {
    if (logsContainerRef.value) {
      logsContainerRef.value.scrollTop = logsContainerRef.value.scrollHeight;
    }
  });
};

watch(
  () => props.logs.length,
  () => {
    scrollToBottom();
  }
);

defineExpose({
  logsContainerRef,
  scrollToBottom,
});
</script>

<style lang="scss" scoped>
.ws-log-box {
  display: flex;
  flex-direction: column;
  height: 100%;
  border: 1px solid var(--el-border-color);
  border-radius: var(--el-border-radius-base);
  overflow: hidden;
  background: var(--el-bg-color);

  &__header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--el-fill-color-light);
    border-bottom: 1px solid var(--el-border-color);
    flex-shrink: 0;
  }

  &__title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--el-text-color-primary);
    font-weight: 500;
  }

  &__icon {
    color: var(--el-color-primary);
  }

  &__actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  &__status {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  &__status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;

    &--connected {
      background: var(--el-color-success);
      box-shadow: 0 0 8px var(--el-color-success-light-5);
    }

    &--connecting {
      background: var(--el-color-warning);
      animation: pulse 1s infinite;
    }

    &--disconnected {
      background: var(--el-color-info);
    }

    &--error {
      background: var(--el-color-error);
    }
  }

  &__content {
    flex: 1;
    padding: 12px 16px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.6;
    background: var(--el-bg-color);
    color: var(--el-text-color-primary);
    min-height: 0;

    &::-webkit-scrollbar {
      width: 8px;
    }

    &::-webkit-scrollbar-track {
      background: var(--el-fill-color-light);
    }

    &::-webkit-scrollbar-thumb {
      background: var(--el-border-color-darker);
      border-radius: 4px;

      &:hover {
        background: var(--el-border-color);
      }
    }
  }

  &__empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--el-text-color-placeholder);
    gap: 12px;

    p {
      margin: 0;
    }
  }

  &__line {
    display: flex;
    gap: 8px;
    padding: 2px 0;
    word-break: break-word;

    &--info {
      color: var(--el-text-color-primary);
    }

    &--warning {
      color: var(--el-color-warning);
    }

    &--error {
      color: var(--el-color-error);
    }

    &--debug {
      color: var(--el-color-info);
    }
  }

  &__timestamp {
    color: var(--el-text-color-placeholder);
    flex-shrink: 0;
  }

  &__message {
    flex: 1;
  }

  &__footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--el-fill-color-light);
    border-top: 1px solid var(--el-border-color);
    flex-shrink: 0;
  }

  &__count {
    color: var(--el-text-color-placeholder);
    font-size: 12px;
  }
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
