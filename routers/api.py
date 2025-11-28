from datetime import datetime
import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from configs.settings import settings 
from services.comparison_service import compare_xyz_files
from services.dataset_stats_service import analyze_xyz_dataset
from services.model_service import test_model_with_logging
from services.prediction_service import predict_with_logging
from services.training_service import train_dgcnn_with_logging
from services.management_service import (
    get_model_architectures, 
    get_trained_models, 
    get_datasets_list, 
)

router = APIRouter()

class TestConfig(BaseModel):
    model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    # Добавь другие поля конфига по необходимости

@router.get("/model-architectures", 
           summary="Получить архитектуры моделей",
           description="Возвращает список доступных архитектур моделей (.py файлы)")
async def list_model_architectures():
    """Получить список архитектур моделей (.py файлы)"""
    return get_model_architectures()

@router.get("/trained-models",
           summary="Получить обученные модели",
           description="Возвращает список обученных моделей (.pth файлы)")
async def list_trained_models():
    """Получить список обученных моделей (.pth файлы)"""
    return get_trained_models()

@router.get("/datasets",
           summary="Получить датасеты",
           description="Возвращает список датасетов")
async def list_datasets(subdir: Optional[str] = Query(None, description="Поддиректория")):
    """Получить список датасетов"""
    return get_datasets_list(subdir=subdir)

@router.websocket("/ws/test-model")
async def websocket_test_model(websocket: WebSocket):
    """Тестирование модели через WebSocket"""
    await websocket_manager(websocket, test_model_with_logging, "test")

@router.websocket("/ws/train-model")
async def websocket_train_model(websocket: WebSocket):
    """Обучение модели через WebSocket"""
    await websocket_manager(websocket, train_dgcnn_with_logging, "train")

@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Предсказание через WebSocket"""
    await websocket_manager(websocket, predict_with_logging, "predict")

@router.get("/compare-xyz",
           summary="Сравнить XYZ файлы",
           description="Сравнивает два XYZ файла: исходный и предсказанный")
async def compare_xyz(
    original_file: str = Query(..., description="Имя файла в DATASETS_DIR/raw/"),
    predicted_file: str = Query(..., description="Имя файла в DATASETS_DIR/predicted/")
):
    """
    Сравнить два XYZ файла: исходный и предсказанный
    """
    orig_path = settings.DATASETS_DIR / "raw" / original_file
    pred_path = settings.DATASETS_DIR / "predicted" / predicted_file
    
    if not orig_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {orig_path}")
    if not pred_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {pred_path}")
    
    try:
        result = compare_xyz_files(orig_path, pred_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset-stats",
           summary="Статистика датасета",
           description="Получить статистику по классам в .xyz файле")
async def get_dataset_stats(
    file_path: str = Query(..., description="Имя файла (например, 'Univer2019_short.xyz')"),
    subdir: str = Query("raw", description="Поддиректория (например, 'raw', 'predicted')")
):
    """
    Получить статистику по классам в .xyz файле
    """
    full_path = settings.DATASETS_DIR / subdir / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {full_path}")
    
    try:
        result = analyze_xyz_dataset(full_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Роуты для HTML страниц
@router.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/test_ws.html")

@router.get("/models.html")
async def models_page():
    return FileResponse("static/models.html")

@router.get("/datasets.html")
async def datasets_page():
    return FileResponse("static/datasets.html")

@router.get("/train.html")
async def train_page():
    return FileResponse("static/train.html")

@router.get("/test_ws.html")
async def test_ws_page():
    return FileResponse("static/test_ws.html")

@router.get("/predict.html")
async def predict_page():
    return FileResponse("static/predict.html")

@router.get("/compare.html")
async def compare_page():
    return FileResponse("static/compare.html")

async def websocket_manager(websocket: WebSocket, service_func, session_type: str):
    """Универсальный менеджер WebSocket сессий"""
    await websocket.accept()
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": f"🔗 WebSocket подключён. Ожидаю конфигурацию {session_type}...",
            "timestamp": datetime.now().isoformat()
        }))
        
        data = await websocket.receive_text()
        config = json.loads(data)
        
        await service_func(websocket, config)
        
    except WebSocketDisconnect:
        print(f"Клиент отключился от {session_type}")
    except Exception as e:
        error_msg = json.dumps({
            "type": "error",
            "message": f"❌ Критическая ошибка: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        try:
            await websocket.send_text(error_msg)
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass