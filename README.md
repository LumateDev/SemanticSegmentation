# 🚀 LiDAR Semantic Segmentation with DGCNN

## ⚡ Quick Start

### System Requirements
- **Python**: 3.11+ (required)
- **CUDA**: 12.6+ (optional, for GPU acceleration)

### 1. Setup Environment

```bash
# Create virtual environment with Python 3.11
python -3.11 -m venv .venv

# Activate on Windows
.\.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install PyTorch with CUDA Support (Optional)

**If you want to use GPU acceleration, install PyTorch with CUDA first:**

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> ⚠️ **Important**: 
> - Python 3.11+ is required for compatibility with dependencies
> - If CUDA installation fails, try using VPN or install CPU-only version
> - This step is needed because requirements.txt may install PyTorch without CUDA

### 3. Install Other Dependencies

```bash
# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Prepare Data Structure

Ensure you have the following folder structure:

```
datasets/
├── raw/           # 📦 Labeled LAS datasets for training
├── unlabeled/     # 🔮 Clean datasets for prediction
```

### 5. Data Preparation

**Create unlabeled datasets from labeled ones:**

```bash
python .\utils\create_unlabeled.py
```

## 🛠️ Usage Commands

### 🧪 Test Model Architecture

```bash
python .\models\modelDGCNN.py
```

### 🎓 Train Model

```bash
python .\training\trainDGCNN.py --las_file datasets/raw/NEONDSSampleLiDARPointCloud.las
```

### 🔮 Run Prediction

**First, check available trained models:**

```bash
dir .\checkpoints\DGCNN\
```

**Then run prediction with the latest model:**

```bash
python .\inference\predictDGCNN.py --checkpoint .\checkpoints\DGCNN\DGCNN_20251113_192702\best_model.pth --input .\datasets\unlabeled\NEONDSSampleLiDARPointCloud.las
```

---

## 📋 Command Summary

| Action | Command |
|--------|---------|
| **Create unlabeled data** | `python .\utils\create_unlabeled.py` |
| **Test model** | `python .\models\modelDGCNN.py` |
| **Train model** | `python .\training\trainDGCNN.py --las_file datasets/raw/NEONDSSampleLiDARPointCloud.las` |
| **Run prediction** | `python .\inference\predictDGCNN.py --checkpoint .\checkpoints\DGCNN\DGCNN_20251113_192702\best_model.pth --input .\datasets\unlabeled\NEONDSSampleLiDARPointCloud.las` |

> ⚠️ **Note**: Always check the actual model folder name in `checkpoints/DGCNN/` before running prediction!