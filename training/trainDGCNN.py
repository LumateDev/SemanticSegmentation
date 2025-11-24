"""
Скрипт обучения DGCNN для семантической сегментации LiDAR данных
С поддержкой нескольких датасетов и дообучения
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import traceback
import glob

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modelDGCNN import DGCNN_LiDAR
from utils.dataset import LidarDataset
from utils.losses import FocalLoss, compute_class_weights
from utils.metrics import SegmentationMetrics


# ==================== КОНФИГУРАЦИЯ ====================

def get_config():
    """Конфигурация обучения"""
    config = {
        # ========== ДАННЫЕ ==========
        'data_files': [],  # Список файлов для обучения
        'dataset_config': 'configs/datasets/neon_sample.yaml',
        'num_points': 4096,
        'block_size': 50.0,
        'stride': 25.0,
        'use_features': True,
        'feature_dim': 3,
        'normalize': True,
        'train_ratio': 0.8,
        
        # ========== МОДЕЛЬ ==========
        'model_name': 'DGCNN',
        'num_classes': 4,
        'k_neighbors': 20,
        'dropout': 0.5,
        
        # ========== ОБУЧЕНИЕ ==========
        'batch_size': 8,
        'epochs': 3,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # ========== LOSS ==========
        'loss_type': 'focal',
        'use_class_weights': True,
        'focal_gamma': 2.0,
        'weight_mode': 'effective',
        
        # ========== OPTIMIZER & SCHEDULER ==========
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'min_lr': 1e-6,
        'patience': 10,
        
        # ========== ДООБУЧЕНИЕ ==========
        'resume_training': False,
        'checkpoint_path': None,
        'finetune': False,
        'freeze_backbone': False,
        
        # ========== EARLY STOPPING ==========
        'early_stopping': True,
        'early_stopping_patience': 15,
        'early_stopping_delta': 0.001,
        
        # ========== AUGMENTATION ==========
        'augment_train': True,
        'augment_val': False,
        
        # ========== ДРУГОЕ ==========
        'num_workers': 4,
        'pin_memory': True if torch.cuda.is_available() else False,
        'save_freq': 5,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    return config


class MultiDatasetTrainer:
    """Тренер для работы с несколькими датасетами"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Создание директорий
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"DGCNN_{self.timestamp}"
        
        self.log_dir = Path('logs') / 'DGCNN' / self.run_name
        self.checkpoint_dir = Path('checkpoints') / 'DGCNN' / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # История обучения
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_miou': [],
            'val_loss': [], 'val_acc': [], 'val_miou': [], 'lr': []
        }
        
        # Лучшие метрики
        self.best_val_acc = 0.0
        self.best_val_miou = 0.0
        self.best_epoch = 0
        
        # Метрики
        self.train_metrics = SegmentationMetrics(num_classes=config['num_classes'])
        self.val_metrics = SegmentationMetrics(num_classes=config['num_classes'])
        
        # Early stopping
        if config['early_stopping']:
            self.early_stopping = EarlyStopping(
                patience=config['early_stopping_patience'],
                delta=config['early_stopping_delta']
            )
        else:
            self.early_stopping = None
        
        self._print_header()
        self._set_seed()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._save_config()
    
    def _print_header(self):
        """Печать заголовка"""
        print("\n" + "="*80)
        print(f"{'🚀 DGCNN LIDAR SEMANTIC SEGMENTATION (MULTI-DATASET)':^80}")
        print("="*80)
        print(f"\n📅 Timestamp: {self.timestamp}")
        print(f"📁 Logs: {self.log_dir}")
        print(f"💾 Checkpoints: {self.checkpoint_dir}")
        print(f"🖥️  Device: {self.device}")
        print(f"📊 Датасеты: {len(self.config['data_files'])} файлов")
        
        for i, file in enumerate(self.config['data_files'], 1):
            print(f"   {i}. {Path(file).name}")
    
    def _set_seed(self):
        """Установка seed для воспроизводимости"""
        seed = self.config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_data(self):
        """Подготовка данных из нескольких файлов"""
        print("\n" + "="*80)
        print("📦 ПОДГОТОВКА ДАННЫХ ИЗ НЕСКОЛЬКИХ ФАЙЛОВ")
        print("="*80)
        
        if not self.config['data_files']:
            raise ValueError("Не указаны файлы для обучения!")
        
        # Загрузка конфигурации датасета
        from utils.dataset_config import DatasetConfig
        
        dataset_config_path = self.config.get('dataset_config', None)
        if dataset_config_path and os.path.exists(dataset_config_path):
            dataset_config = DatasetConfig(dataset_config_path)
            print(f"📋 Загружена конфигурация: {dataset_config_path}")
        else:
            # Используем первый файл для автоопределения
            dataset_config = None
            for data_file in self.config['data_files']:
                from utils.dataset_config import auto_detect_config
                dataset_config = auto_detect_config(data_file)
                if dataset_config is not None:
                    break
            
            if dataset_config is None:
                # Fallback
                neon_config = Path('configs/datasets/neon_sample.yaml')
                if neon_config.exists():
                    dataset_config = DatasetConfig(neon_config)
                    print("⚠️  Используется конфигурация NEON по умолчанию")
                else:
                    raise FileNotFoundError("Dataset config not found!")
        
        dataset_config.print_info()
        self.config['num_classes'] = dataset_config.num_classes
        self.dataset_config = dataset_config
        
        # Создание датасетов для каждого файла
        train_datasets = []
        val_datasets = []
        
        for data_file in self.config['data_files']:
            if not os.path.exists(data_file):
                print(f"⚠️  Файл не найден: {data_file}, пропускаем")
                continue
            
            print(f"\n📂 Загрузка: {Path(data_file).name}")
            
            # Полный датасет
            full_dataset = LidarDataset(
                data_file=data_file,
                num_points=self.config['num_points'],
                block_size=self.config['block_size'],
                stride=self.config['stride'],
                use_features=self.config['use_features'],
                normalize=self.config['normalize'],
                augment=False,
                dataset_config=dataset_config
            )
            
            # Разделение на train/val
            train_size = int(self.config['train_ratio'] * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            generator = torch.Generator().manual_seed(self.config['seed'])
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size], generator=generator
            )
            
            # Включаем аугментации для train
            if self.config['augment_train']:
                full_dataset.augment = True
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            
            print(f"   • Блоков: {len(full_dataset)}")
            print(f"   • Train: {len(train_dataset)}")
            print(f"   • Val: {len(val_dataset)}")
        
        if not train_datasets:
            raise ValueError("Не удалось загрузить ни один датасет!")
        
        # Объединяем датасеты
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        print(f"\n✅ DataLoaders созданы:")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches:   {len(self.val_loader)}")
        print(f"   Всего блоков:  {len(self.train_dataset) + len(self.val_dataset)}")
    
    def _setup_model(self):
        """Создание или загрузка модели"""
        print("\n" + "="*80)
        print("🧠 СОЗДАНИЕ МОДЕЛИ")
        print("="*80)
        
        # Автоматическое определение feature_dim
        # Проверяем первый элемент датасета для определения размерности
        if len(self.train_dataset) > 0:
            sample_points, _ = self.train_dataset[0]
            actual_feature_dim = sample_points.shape[1] - 3  # минус 3 координаты
            self.config['feature_dim'] = actual_feature_dim
            print(f"🔍 Автоопределен feature_dim: {actual_feature_dim}")
        
        # Загрузка чекпоинта для дообучения
        if self.config['resume_training'] and self.config['checkpoint_path']:
            print(f"📥 Загрузка модели для дообучения: {self.config['checkpoint_path']}")
            self.model, start_epoch, self.history = self._load_checkpoint()
        else:
            # Создание новой модели с правильным feature_dim
            self.model = DGCNN_LiDAR(
                num_classes=self.config['num_classes'],
                k=self.config['k_neighbors'],
                use_features=self.config['use_features'],
                feature_dim=self.config['feature_dim'],  # теперь правильное значение
                dropout=self.config['dropout']
            ).to(self.device)
            start_epoch = 0
        
        # Заморозка backbone для тонкой настройки
        if self.config['finetune'] and self.config['freeze_backbone']:
            print("🔒 Заморозка backbone слоев")
            for name, param in self.model.named_parameters():
                if 'edgeconv' in name or 'conv_global' in name:
                    param.requires_grad = False
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Модель: DGCNN")
        print(f"   Классов: {self.config['num_classes']}")
        print(f"   Параметров: {total_params:,}")
        print(f"   Обучаемых: {trainable_params:,}")
        print(f"   Заморожено: {total_params - trainable_params:,}")
        
        # Loss function
        print(f"\n📉 Loss function: {self.config['loss_type'].upper()}")
        if self.config['loss_type'] == 'focal':
            self.criterion = FocalLoss(gamma=self.config['focal_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def _load_checkpoint(self):
        """Загрузка чекпоинта для дообучения"""
        checkpoint_path = self.config['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Создание модели с параметрами из чекпоинта
        model = DGCNN_LiDAR(
            num_classes=self.config['num_classes'],
            k=checkpoint['config'].get('k_neighbors', 20),
            use_features=self.config['use_features'],
            feature_dim=self.config['feature_dim'],
            dropout=checkpoint['config'].get('dropout', 0.5)
        ).to(self.device)
        
        # Загрузка весов
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # История обучения
        history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [], 'train_miou': [],
            'val_loss': [], 'val_acc': [], 'val_miou': [], 'lr': []
        })
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"   ✅ Модель загружена из эпохи {start_epoch - 1}")
        print(f"   📊 Лучшая accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        
        return model, start_epoch, history
    
    def _setup_optimizer(self):
        """Настройка оптимизатора"""
        print("\n" + "="*80)
        print("🎯 OPTIMIZER & SCHEDULER")
        print("="*80)
        
        lr = self.config['learning_rate']
        wd = self.config['weight_decay']
        
        # Разные learning rates для разных частей модели при fine-tuning
        if self.config['finetune'] and self.config['freeze_backbone']:
            print("🎯 Использование разных learning rates для fine-tuning")
            
            # Параметры backbone (заморожены или маленький lr)
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'edgeconv' in name or 'conv_global' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            
            # Для fine-tuning используем меньший lr для backbone
            param_groups = [
                {'params': backbone_params, 'lr': lr * 0.1 if not self.config['freeze_backbone'] else 0.0},
                {'params': head_params, 'lr': lr}
            ]
        else:
            # Все параметры с одинаковым lr
            param_groups = self.model.parameters()
        
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=wd,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optim.Adam(param_groups, lr=lr, weight_decay=wd)
        
        print(f"✅ Optimizer: {self.config['optimizer'].upper()}")
        print(f"   Learning Rate: {lr}")
        
        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['min_lr']
            )
        else:
            self.scheduler = None
    
    def train(self):
        """Основной цикл обучения"""
        print("\n" + "="*80)
        print("🚀 НАЧАЛО ОБУЧЕНИЯ")
        print("="*80)
        
        start_epoch = len(self.history['train_loss']) + 1
        
        for epoch in range(start_epoch, self.config['epochs'] + 1):
            print(f"\n{'='*80}")
            print(f"📅 Эпоха {epoch}/{self.config['epochs']}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate(epoch)
            
            # Вывод результатов
            print(f"\n📊 Результаты эпохи {epoch}:")
            print(f"   {'─'*76}")
            print(f"   📈 TRAIN | Loss: {train_loss:.4f} | Acc: {train_metrics['overall_acc']:6.2f}% | mIoU: {train_metrics['mean_iou']:6.2f}%")
            print(f"   📉 VAL   | Loss: {val_loss:.4f} | Acc: {val_metrics['overall_acc']:6.2f}% | mIoU: {val_metrics['mean_iou']:6.2f}%")
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['overall_acc'])
            self.history['train_miou'].append(train_metrics['mean_iou'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['overall_acc'])
            self.history['val_miou'].append(val_metrics['mean_iou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # TensorBoard
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_metrics['overall_acc'], 'val': val_metrics['overall_acc']}, epoch)
            self.writer.add_scalars('mIoU', {'train': train_metrics['mean_iou'], 'val': val_metrics['mean_iou']}, epoch)
            
            # Проверка на лучшую модель
            is_best = val_metrics['overall_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['overall_acc']
                self.best_val_miou = val_metrics['mean_iou']
                self.best_epoch = epoch
            
            # Сохранение чекпоинта
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_metrics['overall_acc'], epoch)
                if self.early_stopping.early_stop:
                    print(f"\n⚠️  Early stopping сработал на эпохе {epoch}")
                    break
        
        self._finish_training()
    
    # Остальные методы (train_epoch, validate, save_checkpoint, etc.) остаются аналогичными
    # из предыдущей реализации, но используют self.train_loader и self.val_loader
    
    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config["epochs"]} [TRAIN]',
            ncols=100,
            ascii=True
        )
        
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(points)
            
            outputs_flat = outputs.reshape(-1, self.config['num_classes'])
            labels_flat = labels.reshape(-1)
            
            loss = self.criterion(outputs_flat, labels_flat)
            loss.backward()
            
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            with torch.no_grad():
                preds = outputs_flat.argmax(dim=1)
                self.train_metrics.update(preds, labels_flat)
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                current_metrics = self.train_metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_metrics["overall_acc"]:.1f}%'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.train_metrics.get_metrics()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Валидация"""
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch}/{self.config["epochs"]} [VAL]  ',
            ncols=100,
            ascii=True,
            leave=False
        )
        
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(points)
            
            outputs_flat = outputs.reshape(-1, self.config['num_classes'])
            labels_flat = labels.reshape(-1)
            
            loss = self.criterion(outputs_flat, labels_flat)
            total_loss += loss.item()
            
            preds = outputs_flat.argmax(dim=1)
            self.val_metrics.update(preds, labels_flat)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.get_metrics()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'history': self.history,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'best_val_miou': self.best_val_miou,
            'best_epoch': self.best_epoch
        }
        
        last_path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"      💾 Best model saved! (Acc: {self.best_val_acc:.2f}%)")
        
        if epoch % self.config['save_freq'] == 0:
            epoch_path = self.checkpoint_dir / f'model_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def _save_config(self):
        """Сохранение конфигурации"""
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"\n💾 Конфигурация сохранена: {config_path}")
    
    def _finish_training(self):
        """Завершение обучения"""
        print("\n" + "="*80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("="*80)
        
        print(f"\n🏆 Лучший результат:")
        print(f"   Эпоха: {self.best_epoch}")
        print(f"   Accuracy: {self.best_val_acc:.2f}%")
        print(f"   mIoU: {self.best_val_miou:.2f}%")
        
        self.writer.close()


# ==================== УТИЛИТЫ ====================

class EarlyStopping:
    """Early stopping для предотвращения переобучения"""
    
    def __init__(self, patience=15, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"   ⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0


def find_data_files(data_input):
    """
    Поиск файлов данных на основе ввода пользователя
    """
    if isinstance(data_input, list):
        return data_input
    
    data_input = str(data_input)
    
    # Если это папка - ищем все поддерживаемые файлы
    if os.path.isdir(data_input):
        extensions = ['*.las', '*.laz', '*.xyz', '*.txt']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(data_input, ext)))
        return files
    
    # Если это шаблон (например, "datasets/raw/*.las")
    if '*' in data_input:
        return glob.glob(data_input)
    
    # Одиночный файл
    return [data_input]


# ==================== MAIN ====================

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Train DGCNN for LiDAR Segmentation')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to data file, folder, or pattern (e.g., "datasets/raw/*.las")')
    parser.add_argument('--dataset_config', type=str, default=None, 
                       help='Path to dataset config YAML')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, 
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--finetune', action='store_true',
                       help='Fine-tuning mode')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers during fine-tuning')
    
    args = parser.parse_args()
    
    # Конфигурация
    config = get_config()
    
    # Поиск файлов данных
    data_files = find_data_files(args.data)
    if not data_files:
        print(f"❌ Не найдены файлы данных: {args.data}")
        return
    
    config['data_files'] = data_files
    
    # Переопределение из аргументов
    if args.dataset_config is not None:
        config['dataset_config'] = args.dataset_config
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    # Настройки дообучения
    if args.resume is not None:
        config['resume_training'] = True
        config['checkpoint_path'] = args.resume
        config['finetune'] = args.finetune
        config['freeze_backbone'] = args.freeze_backbone
    
    print(f"\n📁 Найдено файлов: {len(data_files)}")
    for file in data_files:
        print(f"   • {file}")
    
    # Создание тренера
    try:
        trainer = MultiDatasetTrainer(config)
        trainer.train()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
    
    except Exception as e:
        print(f"\n\n❌ Ошибка во время обучения:")
        print(f"   {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()