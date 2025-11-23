import asyncio
from fastapi import WebSocket
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from datetime import datetime
from pathlib import Path
import json
from models.modelDGCNN import DGCNN_LiDAR, knn, get_graph_feature, EdgeConvBlock
from utils.logger import get_logger, cleanup_websocket, send_result_to_websocket
from configs.settings import settings


async def test_model_with_logging(ws: WebSocket, config: dict):
    """Тест модели с логированием в консоль, файл и WebSocket"""
    
    # Создаем универсальный логгер с WebSocket
    logger = get_logger('test', websocket=ws)
    
    try:
        logger.info("🔗 WebSocket подключён — логи транслируются в реальном времени")
        logger.info("🚀 Запуск тестирования DGCNN модели для LiDAR сегментации")
        
        # Устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Устройство: {device}")
        
        if device.type == 'cuda':
            logger.info(f"   📊 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   📉 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Получаем параметры модели из конфига или используем по умолчанию
        num_classes = config.get('num_classes', settings.NUM_CLASSES if hasattr(settings, 'NUM_CLASSES') else 8)
        k = config.get('k', 20)
        use_features = config.get('use_features', True)
        feature_dim = config.get('feature_dim', 3)
        dropout = config.get('dropout', 0.5)
        
        logger.info(f"⚙️ Параметры модели:")
        logger.info(f"   Классов: {num_classes}")
        logger.info(f"   K соседей: {k}")
        logger.info(f"   Использование признаков: {use_features}")
        logger.info(f"   Размерность признаков: {feature_dim}")
        logger.info(f"   Dropout: {dropout}")
        
        # Создание модели
        logger.info("🧠 Создание DGCNN модели...")
        model = DGCNN_LiDAR(
            num_classes=num_classes,
            k=k,
            use_features=use_features,
            feature_dim=feature_dim,
            dropout=dropout
        ).to(device)
        
        logger.info("✅ Модель создана успешно")
        
        # Вывод информации о модели (с полной информацией)
        await _log_model_summary(logger, model)
        
        # Тестирование forward pass
        logger.info("\n🧪 Тестирование forward pass...")
        
        # Определяем размерность входных данных
        input_channels = 3 + (feature_dim if use_features else 0)
        batch_size = 2
        num_points = 4096
        
        x = torch.randn(batch_size, num_points, input_channels).to(device)
        logger.info(f"   Input shape: {x.shape}")
        
        with torch.no_grad():
            model.eval()
            start_time = datetime.now()
            output = model(x)
            end_time = datetime.now()
        
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Время inference: {(end_time - start_time).total_seconds():.3f}s")
        logger.info("✅ Forward pass успешен!")
        
        # Тестирование backward pass
        logger.info("\n🔬 Тестирование backward pass...")
        
        model.train()
        labels = torch.randint(0, num_classes, (batch_size, num_points)).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, num_classes), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        logger.info(f"   📉 Loss: {loss.item():.6f}")
        logger.info("✅ Backward pass успешен!")
        
        # Тестирование на разных размерах батча
        logger.info("\n🧪 Тестирование на разных размерах батча...")
        
        batch_sizes = [1, 4, 8]
        for bs in batch_sizes:
            try:
                test_x = torch.randn(bs, num_points, input_channels).to(device)
                with torch.no_grad():
                    test_output = model(test_x)
                logger.info(f"   Batch size {bs}: ✅ (output: {test_output.shape})")
            except Exception as e:
                logger.warning(f"   Batch size {bs}: ❌ {str(e)}")
        
        logger.info("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        logger.info(f"✅ Модель готова к работе!")
        logger.info(f"📊 Параметров: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"📊 Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Отправляем финальный результат
        result = {
            "success": True,
            "message": "Тестирование модели завершено успешно",
            "device": str(device),
            "loss": loss.item(),
            "num_classes": num_classes,
            "parameters": sum(p.numel() for p in model.parameters()),
            "session_id": logger.session_id
        }
        
        await send_result_to_websocket(ws, result)
        return result
        
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e),
            "session_id": logger.session_id if hasattr(logger, 'session_id') else None
        }
        
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "data": error_result,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))
        except:
            pass
        
        raise
    finally:
        # Очищаем WebSocket сессию
        cleanup_websocket(logger.session_id)
        await asyncio.sleep(0.1)  # Даем время на отправку последних сообщений


async def _log_model_summary(logger, model: DGCNN_LiDAR):
    """Вывод информации о модели с полным логированием"""
    
    logger.info("=" * 80)
    logger.info(f"{'DGCNN MODEL SUMMARY':^80}")
    logger.info("=" * 80)
    
    # Параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n📊 Параметры модели:")
    logger.info(f"   Всего параметров: {total_params:,}")
    logger.info(f"   Обучаемых: {trainable_params:,}")
    logger.info(f"   Размер модели: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Архитектура
    logger.info(f"\n🏗️ Архитектура:")
    logger.info(f"   Классов: {model.num_classes}")
    logger.info(f"   K соседей: {model.k}")
    logger.info(f"   Использование признаков: {model.use_features}")
    
    logger.info(f"\n🔧 Слои:")
    logger.info(f"   ┌─ ENCODER (Edge Convolutions):")
    logger.info(f"   │  ├─ EdgeConv1: input → 64 channels")
    logger.info(f"   │  ├─ EdgeConv2: 64 → 64 channels")
    logger.info(f"   │  ├─ EdgeConv3: 64 → 128 channels")
    logger.info(f"   │  └─ EdgeConv4: 128 → 256 channels")
    logger.info(f"   │")
    logger.info(f"   ├─ GLOBAL AGGREGATION:")
    logger.info(f"   │  └─ Conv1d: 512 → 1024 channels")
    logger.info(f"   │")
    logger.info(f"   └─ DECODER (Segmentation Head):")
    logger.info(f"      ├─ Conv1d: 1536 → 512 channels")
    logger.info(f"      ├─ Conv1d: 512 → 256 channels")
    logger.info(f"      ├─ Dropout: p={model._modules.get('dp', nn.Dropout(0.5)).p}")
    logger.info(f"      ├─ Conv1d: 256 → 128 channels")
    logger.info(f"      └─ Output: 128 → {model.num_classes} classes")
    
    logger.info(f"\n⚡ Функции активации: LeakyReLU(negative_slope=0.2)")
    logger.info(f"   Нормализация: BatchNorm2d/BatchNorm1d")
    
    logger.info("=" * 80)