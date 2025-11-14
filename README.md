# 🚀 LiDAR Semantic Segmentation with DGCNN

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data Structure

Ensure you have the following folder structure:

```
datasets/
├── raw/           # 📦 Labeled LAS datasets for training
├── unlabeled/     # 🔮 Clean datasets for prediction
```

### 3. Data Preparation

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

| Action                    | Command                                                                                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Create unlabeled data** | `python .\utils\create_unlabeled.py`                                                                                                                                    |
| **Test model**            | `python .\models\modelDGCNN.py`                                                                                                                                         |
| **Train model**           | `python .\training\trainDGCNN.py --las_file datasets/raw/NEONDSSampleLiDARPointCloud.las`                                                                               |
| **Run prediction**        | `python .\inference\predictDGCNN.py --checkpoint .\checkpoints\DGCNN\DGCNN_20251113_192702\best_model.pth --input .\datasets\unlabeled\NEONDSSampleLiDARPointCloud.las` |

> ⚠️ **Note**: Always check the actual model folder name in `checkpoints/DGCNN/` before running prediction!
