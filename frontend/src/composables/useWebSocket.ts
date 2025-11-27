import { ref, reactive } from 'vue';

export type WebSocketStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface WebSocketOptions {
  url: string;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: () => void;
  onMessage?: (event: MessageEvent) => void;
}

export function useWebSocket(options: WebSocketOptions) {
  const ws = ref<WebSocket | null>(null);
  const isConnected = ref(false);
  const isConnecting = ref(false);
  const status = ref<WebSocketStatus>('disconnected');

  const connect = () => {
    if (isConnected.value || isConnecting.value) return;

    isConnecting.value = true;
    status.value = 'connecting';

    try {
      ws.value = new WebSocket(options.url);

      ws.value.onopen = () => {
        isConnected.value = true;
        isConnecting.value = false;
        status.value = 'connected';
        options.onOpen?.();
      };

      ws.value.onclose = () => {
        isConnected.value = false;
        isConnecting.value = false;
        status.value = 'disconnected';
        options.onClose?.();
      };

      ws.value.onerror = () => {
        isConnected.value = false;
        isConnecting.value = false;
        status.value = 'error';
        options.onError?.();
      };

      ws.value.onmessage = event => {
        options.onMessage?.(event);
      };
    } catch (error) {
      isConnecting.value = false;
      status.value = 'error';
      options.onError?.();
    }
  };

  const disconnect = () => {
    if (ws.value) {
      ws.value.close();
    }
  };

  return {
    ws,
    isConnected,
    isConnecting,
    status,
    connect,
    disconnect,
  };
}
