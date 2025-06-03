<p align="center">
  <img src="https://your-image-url.com/banner.png" width="80%" alt="Project Banner">
</p>

<h1 align="center">🔍 CREDIT PVAMU - CADOT Challenge Submission</h1>

<p align="center">
  <b>Ultralytics YOLO-based Ensemble for Aerial Object Detection</b><br>
  <i>Final submission for the CADOT Challenge</i>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.0-yellow">
</p>

---

## 🖥️ Hardware and Software Requirements

- **Operating System**: Ubuntu 20.04 (Linux kernel 4.15.0-213-generic)  
- **CPU**: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (40 cores)  
- **RAM**: 251 GB total (204 GB used, 29 GB cached)  
- **GPU**: 4x NVIDIA Tesla V100-DGXS (32 GB VRAM each)  
- **CUDA Version**: 12.0  
- **Driver Version**: 525.147.05  
- **Python Version**: 3.10.14  
- **Pip Version**: 25.1.1

---
## 🚀 Clone the Repository
```bash
git clone https://github.com/your-username/CREDIT_PVAMU_CADOT_Challenge.git
cd CREDIT_PVAMU_CADOT_Challenge
```

---

## 📦 Dependency Installation

> 💡 It is recommended to use a Python virtual environment.

```bash
# Create and activate virtual environment
cd ./CREDIT_PVAMU_CADOT_Challenge
python3 -m venv .
source venv/bin/activate  # For Linux/macOS
```

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Change current working directory to ./scripts
cd ./CREDIT_PVAMU_CADOT_Challenge/scripts
```

---

## 📂 Get CADOT Dataset

### Option A
- Download CADOT Dataset and augment train data:  
  - Train data: `./CREDIT_PVAMU_CADOT_Challenge/data/data_split`  
  - Validation and test data: `./CREDIT_PVAMU_CADOT_Challenge/data`

```bash
python dataset.py
```

### Option B
- Download Train and Augmented Dataset used in model training from Google Drive:

```bash
# Install gdown package to download on Google Drive
pip install gdown
pip install --upgrade gdown

gdown --fuzzy 'https://drive.google.com/file/d/1BxwQGUsE7ZYH6COhOPPtOLhdeVg062dD/view?usp=sharing'
```

```bash
unzip data_split.zip && cp -r data_split ./CREDIT_PVAMU_CADOT_Challenge/data
```

---

## 🏋️‍♂️ Train Models

### Option A: Train Models

- This solution implements an ensemble technique for object detection in the CADOT Dataset.  
- Three (3) models were trained: `YOLOFocus8l`, `YOLOFocus11x`, and pretrained `YOLOFocus11x` weights  
- These models are located at: `./Challenge-Object-Detection/models`  
- Model train parameters: `./CREDIT_PVAMU_CADOT_Challenge/models/train_config.yaml`

```bash
# The pretrained weight 'pretrained_yolofocus11x.pt' can be downloaded using:
gdown --folder '1dlOGU5FrLSj6yycoJ_FKPTlm2HdCu9nz?usp=sharing'
gdown --folder '1NqD1eEdOdaVTzKRdBulGGRNLjJFCn1Cz?usp=sharing'

# Move pretrained model to target directory
cp ./pretrained_yolofocus11x.pt ./CREDIT_PVAMU_CADOT_Challenge/models
```

```bash
# Train models
python train.py --pretrained_path models/yolofocus8l.yaml --datapath data/data.yaml
python train.py --pretrained_path models/yolofocus11x.yaml --datapath data/data.yaml
python train.py --pretrained_path models/pretrained_yolofocus11x.pt --datapath data/data.yaml

# Trained model weights are saved at:
# ./CREDIT_PVAMU_CADOT_Challenge/models/CADOT_Trained_Models
```

### Option B: Download Trained Model Weights

```bash
# Download 'model_weights'
gdown --folder '1dlOGU5FrLSj6yycoJ_FKPTlm2HdCu9nz'
gdown --folder '1dlOGU5FrLSj6yycoJ_FKPTlm2HdCu9nz?usp=sharing'

cp ./model_weights ./CREDIT_PVAMU_CADOT_Challenge/models
```

---

## 🔍 Run Inference

- This script runs predictions for the three (3) models.  
- It then ensembles the predictions and saves the result to:  
  `../results/predictions.json`

```bash
python infer.py
```

> ℹ️ **Note**: To run inference with provided weights:
> - Ensure `model_weights` is in `./CREDIT_PVAMU_CADOT_Challenge/models`
> - Open `infer.py` and change:  
> `gen_path = f'{project_root}/models/CADOT_Trained_Models'`  
> to  
> `gen_path = f'{project_root}/models/model_weights'`

```bash
cd ./CREDIT_PVAMU_CADOT_Challenge/scripts
vim infer.py
```

---

## 📊 Model Evaluation

```bash
# Script evaluates model performance and saves results to ../results/metrics.csv
# GT file must be in COCO format.

python evaluation_metrics.py
```

---

## 📎 License

MIT License © 2025 CREDIT PVAMU Team

---

## 🙏 Acknowledgments

This project builds upon the open-source work provided by [Ultralytics](https://github.com/ultralytics).  
We used their YOLO models and training framework as a foundation for model development and evaluation.

If you use this work in your research or applications, please also consider citing the original Ultralytics YOLO paper:

```
@article{glenn_jocher_2023_10366482,
  author       = {Glenn Jocher and Ayush Chaurasia and Jiacong Fang and Abhiram V},
  title        = {{YOLO by Ultralytics}},
  journal      = {Zenodo},
  year         = 2023,
  doi          = {10.5281/zenodo.10366482},
  url          = {https://zenodo.org/record/10366482}
}
```
