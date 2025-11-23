import asyncio
import torch
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from fastapi import WebSocket
from configs.settings import settings
from utils.xyz_loader import MultiXYZDataset
from utils.logger import get_logger, cleanup_websocket, send_result_to_websocket
import numpy as np


async def train_dgcnn_with_logging(ws: WebSocket, config: dict):
    """Запуск обучения DGCNN с логированием через WebSocket"""

    logger = get_logger('train', websocket=ws)
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from models.modelDGCNN import DGCNN_LiDAR
        from torch.utils.data import DataLoader, random_split
        import torch.nn as nn
        import torch.optim as optim

        logger.info("📊 Подготовка конфигурации...")

        train_config = {
            'xyz_files': config.get('xyz_files', [str(settings.DATASETS_DIR / 'raw' / 'Смешанный_лес2.xyz')]),
            'num_points': config.get('num_points', settings.DEFAULT_NUM_POINTS),
            'batch_size': config.get('batch_size', settings.DEFAULT_BATCH_SIZE),
            'epochs': config.get('epochs', settings.DEFAULT_EPOCHS),
            'learning_rate': config.get('learning_rate', settings.DEFAULT_LEARNING_RATE),
            'num_classes': settings.NUM_CLASSES,
            'k_neighbors': 20,
            'use_features': False,
            'feature_dim': 0,
            'dropout': 0.5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_name': config.get('model_name', ''),
            'resume_from': config.get('resume_from', None)
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_folder_name = f"{train_config['model_name']}_{timestamp}" if train_config['model_name'] else f"DGCNN_{timestamp}"

        logger.info(f"🖥️ Устройство: {train_config['device']}")
        logger.info(f"📊 Файлы для обучения: {train_config['xyz_files']}")
        logger.info(f"📊 Эпох: {train_config['epochs']}")
        logger.info(f"📊 Batch Size: {train_config['batch_size']}")
        logger.info(f"📊 Learning Rate: {train_config['learning_rate']}")
        logger.info(f"📊 Классов: {train_config['num_classes']}")
        logger.info(f"📊 Название модели: {model_folder_name}")

        # Создаём папку для чекпоинтов 
        checkpoint_dir = settings.CHECKPOINTS_DIR / "DGCNN" / model_folder_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Проверка файлов
        for f in train_config['xyz_files']:
            if not Path(f).exists():
                error_msg = f"❌ Файл данных не найден: {f}"
                logger.error(error_msg)
                error_result = {
                    "success": False, 
                    "message": error_msg,
                    "session_id": logger.session_id
                }
                await send_result_to_websocket(ws, error_result)
                return error_result

        # Модель
        model = DGCNN_LiDAR(
            num_classes=train_config['num_classes'],
            k=train_config['k_neighbors'],
            use_features=train_config['use_features'],
            feature_dim=train_config['feature_dim'],
            dropout=train_config['dropout']
        ).to(train_config['device'])

        # Дообучение
        if train_config['resume_from']:
            logger.info(f"📥 Дообучение из {train_config['resume_from']}")
            checkpoint = torch.load(train_config['resume_from'], map_location=train_config['device'])
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
        if train_config['resume_from'] and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        criterion = nn.CrossEntropyLoss()

        # Данные
        logger.info("📂 Загрузка датасета...")
        full_dataset = MultiXYZDataset(
            file_paths=[Path(f) for f in train_config['xyz_files']],
            num_points=train_config['num_points'],
            normalize=True
        )

        # Проверка меток в датасете
        logger.info(f"📊 Уникальные метки в исходном датасете (до преобразования): {np.unique(full_dataset.labels)}")
        logger.info(f"📊 Распределение классов:")
        unique, counts = np.unique(full_dataset.labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = settings.CLASS_MAPPING.get(int(label), f'Unknown {label}')
            percent = 100.0 * count / len(full_dataset.labels)
            logger.info(f"   {class_name} ({label}): {count:,} точек ({percent:.2f}%)")

        logger.info("📊 Метод нормализации: ЛОКАЛЬНАЯ (per-batch центрирование + масштабирование)")
        logger.info("   ⚠️ Глобальные параметры НЕ используются при обучении!")

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=0)

        logger.info(f"📊 Данные готовы: {len(train_loader)} train, {len(val_loader)} val батчей")
        logger.info("🎯 Начало обучения...")

        # Цикл обучения
        model.train()
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        
        for epoch in range(1, train_config['epochs'] + 1):
            logger.info(f"\n📅 Эпоха {epoch}/{train_config['epochs']}")
            total_loss = 0.0
            
            for batch_idx, (points, labels) in enumerate(train_loader):
                points = points.to(train_config['device'])
                labels = labels.to(train_config['device'])

                # Проверка меток в первом батче первой эпохи
                if epoch == 1 and batch_idx == 0:
                    logger.info(f"🔍 Проверка первого батча:")
                    logger.info(f"   Форма points: {points.shape}")
                    logger.info(f"   Форма labels: {labels.shape}")
                    logger.info(f"   Уникальные метки (после преобразования датасетом): {torch.unique(labels).cpu().numpy()}")
                    logger.info(f"   Ожидаемый диапазон: 0-{train_config['num_classes']-1}")

                optimizer.zero_grad()
                outputs = model(points)
                loss = criterion(outputs.view(-1, train_config['num_classes']), labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    logger.info(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"📊 Эпоха {epoch} | Средний Loss: {avg_loss:.4f}")

            # Валидация
            model.eval()
            val_loss = 0.0
            correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for points, labels in val_loader:
                    points = points.to(train_config['device'])
                    labels = labels.to(train_config['device'])
                    outputs = model(points)
                    
                    loss = criterion(outputs.view(-1, train_config['num_classes']), labels.view(-1))
                    val_loss += loss.item()
                    
                    predictions = outputs.argmax(dim=-1)
                    correct += (predictions == labels).sum().item()
                    total_samples += labels.numel()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100.0 * correct / total_samples
            
            logger.info(f"📉 Валидация | Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")
            
            # Сохранение лучшего чекпоинта
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_accuracy = accuracy
                logger.info(f"💫 Новый лучший результат! Сохранение best_model.pth...")
                
                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'num_classes': train_config['num_classes'],
                        'k_neighbors': train_config['k_neighbors'],
                        'use_features': train_config['use_features'],
                        'feature_dim': train_config['feature_dim']
                    },
                    'training_info': {
                        'epoch': epoch,
                        'val_loss': avg_val_loss,
                        'val_accuracy': accuracy,
                        'train_loss': avg_loss
                    }
                }, best_checkpoint_path)
            
            model.train()

        # Сохранение финального чекпоинта
        success_msg = f"✅ Обучение завершено! Final Loss: {avg_loss:.4f}, Best Val Accuracy: {best_val_accuracy:.2f}%"
        logger.info(success_msg)

        checkpoint_path = checkpoint_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'num_classes': train_config['num_classes'],
                'k_neighbors': train_config['k_neighbors'],
                'use_features': train_config['use_features'],
                'feature_dim': train_config['feature_dim']
            },
            'normalization_method': 'local_per_batch',
            'training_info': {
                'model_name': model_folder_name,
                'epochs': train_config['epochs'],
                'final_loss': avg_loss,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'dataset': train_config['xyz_files'],
                'timestamp': datetime.now().isoformat()
            }
        }, checkpoint_path)

        config_path = checkpoint_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': model_folder_name,
                'training_config': train_config,
                'normalization_method': 'local_per_batch',
                'final_metrics': {
                    'train_loss': avg_loss,
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': best_val_accuracy
                }
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Финальная модель: {checkpoint_path}")
        logger.info(f"💾 Лучшая модель: {checkpoint_dir / 'best_model.pth'}")

        result = {
            "success": True,
            "message": success_msg,
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(checkpoint_dir / "best_model.pth"),
            "model_folder": str(checkpoint_dir),
            "model_name": model_folder_name,
            "final_loss": avg_loss,
            "best_val_accuracy": best_val_accuracy,
            "session_id": logger.session_id
        }
        
        await send_result_to_websocket(ws, result)
        return result

    except Exception as e:
        error_msg = f"❌ Ошибка обучения: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        error_result = {
            "success": False, 
            "message": error_msg,
            "session_id": logger.session_id
        }
        
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "data": error_result,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))
        except:
            pass
        
        return error_result

    finally:
        cleanup_websocket(logger.session_id)
        await asyncio.sleep(0.1)