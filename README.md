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
>
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

### 5. Launch Application

```bash
uvicorn main:app --reload
```

### 6. Access Web Interface

- **📚 Swagger API Documentation**: http://127.0.0.1:8000/docs/
- **🌐 Web Application**: http://127.0.0.1:8000/api/

---

## 📋 Application Features

### 1. **Models**

- View available architectural models (DGCNN)
- Browse trained models stored locally in `checkpoints/` folder

### 2. **Datasets**

- Explore datasets available on your local machine in `datasets/` folder
- View overall list and detailed information for each dataset

### 3. **Training**

- Train new models or fine-tune existing ones
- **3.1 New Model Training**: Enter model name, select one or multiple datasets, configure parameters, and start training
- **3.2 Fine-tuning**: Select pre-trained model and continue training with new datasets and parameters

### 4. **Prediction**

- Select trained model and dataset
- Configure prediction parameters
- Generate predicted datasets

### 5. **Model Testing**

- Test model functionality and performance on your device
- Verify model compatibility and operation

### 6. **Comparison**

- Compare one raw dataset with one predicted dataset
- Analyze differences between classes within the datasets
- Visualize and evaluate segmentation results
