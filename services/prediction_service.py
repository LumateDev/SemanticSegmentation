import asyncio
import numpy as np
import torch
import json
import os
from pathlib import Path
from fastapi import WebSocket
from configs.settings import settings
from models.modelDGCNN import DGCNN_LiDAR
from utils.logger import get_logger, cleanup_websocket, send_result_to_websocket
from utils.xyz_loader import indices_to_labels, load_xyz_no_labels, save_xyz_with_labels
from datetime import datetime


async def predict_with_logging(ws: WebSocket, config: dict):
    """Запуск предсказания с логированием через WebSocket (поддержка одного или нескольких датасетов)"""
    
    logger = get_logger('predict', websocket=ws)
    
    try:
        logger.info("🔮 Запуск предсказания...")
        
        # Определяем режим работы с файлами
        input_files = config.get('input_files', [])
        input_file = config.get('input_file')
        
        # Поддержка обратной совместимости
        if input_file and not input_files:
            input_files = [input_file]
            single_file_mode = True
        elif input_files:
            single_file_mode = False
        else:
            error_msg = "❌ Не указаны файлы для предсказания"
            logger.error(error_msg)
            error_result = {"success": False, "message": error_msg, "session_id": logger.session_id}
            await send_result_to_websocket(ws, error_result)
            return error_result
        
        if single_file_mode:
            logger.info("📁 Режим: один датасет")
        else:
            logger.info(f"📁 Режим: несколько датасетов ({len(input_files)} файлов)")
        
        predict_config = {
            'checkpoint_path': config.get('checkpoint_path'),
            'input_files': input_files,
            'output_dir': config.get('output_dir', 'datasets/predicted'),
            'batch_size': config.get('batch_size', settings.DEFAULT_BATCH_SIZE),
            'num_points': config.get('num_points', settings.DEFAULT_NUM_POINTS),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'single_file_mode': single_file_mode
        }
        
        logger.info(f"🖥️ Устройство: {predict_config['device']}")
        logger.info(f"📊 Конфигурация:")
        logger.info(f"   - Модель: {predict_config['checkpoint_path']}")
        logger.info(f"   - Количество датасетов: {len(predict_config['input_files'])}")
        logger.info(f"   - Выходная папка: {predict_config['output_dir']}")
        logger.info(f"   - Точек в батче: {predict_config['num_points']}")
        
        # Проверяем файлы
        if not os.path.exists(predict_config['checkpoint_path']):
            error_msg = f"❌ Файл модели не найден: {predict_config['checkpoint_path']}"
            logger.error(error_msg)
            error_result = {"success": False, "message": error_msg, "session_id": logger.session_id}
            await send_result_to_websocket(ws, error_result)
            return error_result
        
        if not predict_config['input_files']:
            error_msg = "❌ Не выбраны датасеты для предсказания"
            logger.error(error_msg)
            error_result = {"success": False, "message": error_msg, "session_id": logger.session_id}
            await send_result_to_websocket(ws, error_result)
            return error_result

        missing_files = []
        for file_path in predict_config['input_files']:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            error_msg = f"❌ Файлы не найдены: {', '.join(missing_files)}"
            logger.error(error_msg)
            error_result = {"success": False, "message": error_msg, "session_id": logger.session_id}
            await send_result_to_websocket(ws, error_result)
            return error_result
        
        logger.info("🧠 Загрузка модели...")
        
        # Загружаем модель
        device = torch.device(predict_config['device'])
        model = await load_model(predict_config['checkpoint_path'], device, logger)
        
        if model is None:
            error_msg = "❌ Не удалось загрузить модель"
            logger.error(error_msg)
            error_result = {"success": False, "message": error_msg, "session_id": logger.session_id}
            await send_result_to_websocket(ws, error_result)
            return error_result
        
        logger.info(f"✅ Модель загружена: {sum(p.numel() for p in model.parameters()):,} параметров")
        
        # Создаём выходную директорию
        output_dir = Path(predict_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Статистика по всем датасетам
        all_results = []
        aggregated_stats = {}
        total_points_all = 0
        
        # Обрабатываем каждый датасет
        for i, input_file in enumerate(predict_config['input_files']):
            logger.info(f"📂 Обработка датасета {i+1}/{len(predict_config['input_files'])}: {Path(input_file).name}")
            
            try:
                # Загружаем XYZ файл БЕЗ меток
                xyz = load_xyz_no_labels(Path(input_file))
                
                logger.info(f"📊 Загружено {len(xyz):,} точек")
                if single_file_mode or i == 0:
                    logger.info(f"📊 Диапазон координат:")
                    logger.info(f"   X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
                    logger.info(f"   Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
                    logger.info(f"   Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
                
                # Создаем имя выходного файла
                input_filename = Path(input_file).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"{input_filename}_{timestamp}.xyz"
                
                logger.info("🎯 Начало предсказания...")
                
                predictions = await run_prediction(logger, model, xyz, predict_config, device)
                
                logger.info("💾 Сохранение результатов...")
                
                # Сохраняем результаты
                save_xyz_with_labels(output_file, xyz, predictions)
                
                # Статистика для текущего датасета
                unique, counts = np.unique(predictions, return_counts=True)
                total_points = len(predictions)
                total_points_all += total_points
                
                class_names = {i: name for i, name in settings.CLASS_MAPPING.items()}
                stats = {}
                
                logger.info(f"📊 Статистика для {Path(input_file).name}:")
                for cls, count in zip(unique, counts):
                    percent = 100.0 * count / total_points
                    class_name = class_names.get(int(cls), f'Class {cls}')
                    logger.info(f"   {class_name}: {count:,} точек ({percent:.2f}%)")
                    stats[class_name] = f"{count:,} ({percent:.2f}%)"
                    
                    # Агрегируем статистику
                    if class_name not in aggregated_stats:
                        aggregated_stats[class_name] = 0
                    aggregated_stats[class_name] += count
                
                # Результат для текущего датасета
                dataset_result = {
                    "input_file": input_file,
                    "output_file": str(output_file),
                    "statistics": stats,
                    "total_points": total_points,
                    "success": True
                }
                all_results.append(dataset_result)
                
                logger.info(f"✅ Датсет {Path(input_file).name} обработан успешно")
                
            except Exception as e:
                error_msg = f"❌ Ошибка при обработке {Path(input_file).name}: {str(e)}"
                logger.error(error_msg)
                
                dataset_result = {
                    "input_file": input_file,
                    "success": False,
                    "error": error_msg
                }
                all_results.append(dataset_result)
        
        # Формируем финальный результат в зависимости от режима
        if single_file_mode:
            # Режим одного файла - возвращаем простой результат для обратной совместимости
            if all_results and all_results[0]['success']:
                result = all_results[0]
                result.update({
                    "success": True,
                    "message": f"✅ Предсказание завершено! Результат сохранен: {result['output_file']}",
                    "session_id": logger.session_id
                })
            else:
                result = {
                    "success": False,
                    "message": "❌ Ошибка при обработке файла",
                    "session_id": logger.session_id
                }
        else:
            # Режим нескольких файлов - возвращаем агрегированный результат
            aggregated_formatted = {}
            for class_name, count in aggregated_stats.items():
                percent = 100.0 * count / total_points_all if total_points_all > 0 else 0
                aggregated_formatted[class_name] = f"{count:,} ({percent:.2f}%)"
            
            success_count = len([r for r in all_results if r['success']])
            success_msg = f"✅ Предсказание завершено! Обработано {success_count} датасетов, всего {total_points_all:,} точек"
            logger.info(success_msg)
            
            result = {
                "success": True,
                "message": success_msg,
                "datasets_processed": success_count,
                "datasets_failed": len([r for r in all_results if not r['success']]),
                "total_points": total_points_all,
                "individual_results": all_results,
                "aggregated_statistics": aggregated_formatted,
                "session_id": logger.session_id
            }
        
        await send_result_to_websocket(ws, result)
        return result
        
    except Exception as e:
        error_msg = f"❌ Ошибка предсказания: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        error_result = {"success": False, "message": error_msg, "session_id": logger.session_id}
        
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


async def load_model(checkpoint_path, device, logger):
    """
    Загрузка модели
    """
    try:
        logger.info("🔄 Загрузка чекпоинта...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' not in checkpoint:
            logger.error("❌ Неверный формат чекпоинта")
            return None
        
        # Загружаем конфигурацию модели
        train_config = checkpoint.get('config', {})
        num_classes = train_config.get('num_classes', settings.NUM_CLASSES)
        k_neighbors = train_config.get('k_neighbors', 20)
        use_features = train_config.get('use_features', False)
        feature_dim = train_config.get('feature_dim', 0)
        
        logger.info(f"📋 Конфигурация модели: num_classes={num_classes}, k={k_neighbors}")
        
        # Проверка метода нормализации
        normalization_method = checkpoint.get('normalization_method', 'unknown')
        logger.info(f"📊 Метод нормализации при обучении: {normalization_method}")
        
        if normalization_method != 'local_per_batch':
            logger.warning(f"⚠️ Ожидался метод 'local_per_batch', найден '{normalization_method}'")
            logger.warning("⚠️ Результаты могут быть неточными!")
        
        model = DGCNN_LiDAR(
            num_classes=num_classes,
            k=k_neighbors,
            use_features=use_features,
            feature_dim=feature_dim,
            dropout=0.0
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info("✅ Модель успешно загружена")
        return model
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {str(e)}", exc_info=True)
        return None


async def run_prediction(logger, model, xyz, config, device):
    """
    Предсказание с локальной нормализацией и правильным преобразованием меток
    """
    logger.info("🔨 Подготовка к предсказанию...")
    logger.info(f"📊 Метод нормализации: ЛОКАЛЬНАЯ (per-batch)")
    logger.info(f"🔍 Исходный диапазон координат:")
    logger.info(f"   X: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    logger.info(f"   Y: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    logger.info(f"   Z: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")

    num_points = config['num_points']
    total_points = len(xyz)
    num_batches = (total_points + num_points - 1) // num_points

    predictions = np.zeros(total_points, dtype=np.int32)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * num_points
        end_idx   = min(start_idx + num_points, total_points)
        actual_size = end_idx - start_idx

        # Создаём КОПИЮ исходных данных
        batch_xyz = xyz[start_idx:end_idx].copy()

        # ЛОКАЛЬНАЯ нормализация (идентична датасету)
        centroid = batch_xyz.mean(axis=0)
        batch_xyz -= centroid
        max_dist = np.max(np.sqrt(np.sum(batch_xyz**2, axis=1)))
        if max_dist > 0:
            batch_xyz /= max_dist

        # Дополняем до полного батча
        if actual_size < num_points:
            pad_size = num_points - actual_size
            batch_xyz = np.vstack([batch_xyz, np.tile(batch_xyz[-1:], (pad_size, 1))])

        batch_tensor = torch.FloatTensor(batch_xyz).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)  # (1, N, num_classes)
            batch_pred = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()  # (N,) индексы 0-7

        predictions[start_idx:end_idx] = batch_pred[:actual_size]

        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            progress = (batch_idx + 1) / num_batches * 100
            logger.info(f"🔮 Прогресс: {batch_idx + 1}/{num_batches} батчей ({progress:.1f}%)")

    logger.info("✅ Предсказание завершено")
    
    # Проверяем, что исходные координаты не изменились, удалить
    logger.info(f"🔍 Финальный диапазон координат")
    logger.info(f"   X: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    logger.info(f"   Y: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    logger.info(f"   Z: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")

    logger.info(f"🔍 Уникальные индексы модели (0-{settings.NUM_CLASSES-1}): {sorted(np.unique(predictions))}")

    # преобразование классов 0-7 → 1-8
    predictions = indices_to_labels(predictions)

    logger.info(f"🔍 Уникальные метки для сохранения (1-8): {sorted(np.unique(predictions))}")
    
    return predictions