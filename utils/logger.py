import logging
import json
from datetime import datetime
from typing import Optional
from pathlib import Path
import asyncio
from threading import Lock
from configs.settings import settings

# Глобальный словарь для хранения WebSocket сессий
websocket_sessions = {}
session_lock = Lock()

class WebSocketHandler(logging.Handler):
    """Обработчик для отправки логов через WebSocket"""
    
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(message)s'))
    
    def emit(self, record):
        """Отправка лога через WebSocket если есть подключение"""
        global websocket_sessions
        
        session_id = getattr(record, 'session_id', None)
        if session_id and session_id in websocket_sessions:
            try:
                log_data = {
                    'type': 'log',
                    'level': record.levelname.lower(),
                    'message': self.format(record),
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id
                }
                
                # Получаем WebSocket из сессии
                websocket = websocket_sessions[session_id]
                
                # Для отправки через WebSocket в синхронном обработчике
                # мы используем отдельный поток с новым event loop
                import threading
                thread = threading.Thread(target=self._send_sync, args=(websocket, log_data))
                thread.daemon = True
                thread.start()
                
            except Exception:
                pass  # Игнорируем ошибки WebSocket(Коостыль, да)
    
    def _send_sync(self, websocket, log_data):
        """Синхронная отправка в отдельном потоке"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(websocket.send_text(json.dumps(log_data, ensure_ascii=False)))
            loop.close()
        except Exception:
            pass

class ColoredFormatter(logging.Formatter):
    """Форматтер с цветами для консоли"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Для консоли добавляем цвета
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

class NoColorFormatter(logging.Formatter):
    """Форматтер без цветов для файла"""
    
    def format(self, record):
        # Убираем цвета из уровня лога для файла
        record.levelname = record.levelname.replace('\033[32m', '').replace('\033[36m', '').replace('\033[33m', '').replace('\033[31m', '').replace('\033[35m', '').replace('\033[0m', '')
        return super().format(record)

class UniversalLogger:
    """Универсальный логгер для консоли, файла и WebSocket"""
    
    def __init__(self, name: str, session_id: str, websocket=None):
        self.name = name
        self.session_id = session_id
        self.logger = logging.getLogger(f"{name}_{session_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Очищаем предыдущие обработчики
        self.logger.handlers.clear()
        
        # Console handler с цветами (если не отключен)
        if name not in settings.DISABLE_CONSOLE_LOGGING:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
            self.logger.addHandler(console_handler)
        
        # File handler БЕЗ цветов (если не отключен)
        if name not in settings.DISABLE_FILE_LOGGING:
            # Определяем папку для логов
            log_dir_name = settings.LOG_DIRS.get(name, settings.LOG_DIRS['default'])
            log_dir = Path(log_dir_name)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{name}_{session_id}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(NoColorFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
            self.logger.addHandler(file_handler)
        
        # WebSocket handler БЕЗ цветов (если не отключен и предоставлен)
        if (websocket and 
            name not in settings.DISABLE_WEBSOCKET_LOGGING):
            ws_handler = WebSocketHandler()
            ws_handler.setFormatter(NoColorFormatter('%(message)s'))
            self.logger.addHandler(ws_handler)
            # Сохраняем WebSocket для сессии
            websocket_sessions[session_id] = websocket
    
    def _log_with_session(self, level: str, msg: str, **kwargs):
        """Внутренний метод для логирования с добавлением session_id"""
        # Отделяем зарезервированные параметры от дополнительных
        reserved_keys = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
        reserved_kwargs = {}
        extra_kwargs = {}
        
        for key, value in kwargs.items():
            if key in reserved_keys:
                reserved_kwargs[key] = value
            else:
                extra_kwargs[key] = value
        
        # Добавляем session_id в extra
        extra = {'session_id': self.session_id}
        extra.update(extra_kwargs)
        reserved_kwargs['extra'] = extra
        
        getattr(self.logger, level.lower())(msg, **reserved_kwargs)
    
    def debug(self, msg: str, **kwargs):
        self._log_with_session('DEBUG', msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log_with_session('INFO', msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log_with_session('WARNING', msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log_with_session('ERROR', msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log_with_session('CRITICAL', msg, **kwargs)


async def send_result_to_websocket(ws, result: dict):
    """Асинхронная отправка результата через WebSocket"""
    try:
        await ws.send_text(json.dumps({
            "type": "result",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False))
    except Exception:
        # Логируем ошибку, если нужно, но не прерываем основной процесс
        # logger.error(f"Failed to send result via WebSocket: {e}") # Если у нас есть доступ к логгеру
        pass # Или используем print/debug логирование

def get_logger(name: str, session_id: str = None, websocket=None) -> UniversalLogger:
    """Получить универсальный логгер"""
    if session_id is None:
        session_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    return UniversalLogger(name, session_id, websocket)

def cleanup_websocket(session_id: str):
    """Очистить WebSocket сессию"""
    websocket_sessions.pop(session_id, None)

def disable_file_logging_for_module(module_name: str):
    """Отключить логирование в файл для модуля"""
    settings.DISABLE_FILE_LOGGING.add(module_name)

def enable_file_logging_for_module(module_name: str):
    """Включить логирование в файл для модуля"""
    settings.DISABLE_FILE_LOGGING.discard(module_name)

def disable_console_logging_for_module(module_name: str):
    """Отключить логирование в консоль для модуля"""
    settings.DISABLE_CONSOLE_LOGGING.add(module_name)

def enable_console_logging_for_module(module_name: str):
    """Включить логирование в консоль для модуля"""
    settings.DISABLE_CONSOLE_LOGGING.discard(module_name)

def disable_websocket_logging_for_module(module_name: str):
    """Отключить WebSocket логирование для модуля"""
    settings.DISABLE_WEBSOCKET_LOGGING.add(module_name)

def enable_websocket_logging_for_module(module_name: str):
    """Включить WebSocket логирование для модуля"""
    settings.DISABLE_WEBSOCKET_LOGGING.discard(module_name)
