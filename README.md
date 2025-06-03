# Challenge Object Detection

This project involves training, inference, and evaluation of object detection models using the CADOT Dataset in COCO format and YOLO/Ultralytics-based pipelines. It includes tools for prediction, COCO-style evaluation, and metrics analysis.

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

## 📦 Dependency Installation

> 💡 It is recommended to use a Python virtual environment.


# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS

# Install dependencies
pip install -r requirements.txt


# Change current working directory to ./scripts
cd ./Challenge-Object-Detection/scripts


## Get CADOT Dataset

# Downlaod CADOT Dataset, Augument train data: train data is located @ ./CREDIT_PVAMU_CADOT_Challenge/data/data_split.
# validation and test data @ ./CREDIT_PVAMU_CADOT_Challenge/data.

python dataset.py


## Train models
# This solution implements an emsemble technique for object detection in the CADOT Dataset.

# Three (3) models were trained: 
# ------ YOLOFocus8l
# ------ YOLOFocus11x
# ------ pretrained YOLOFocus11x weight

# These models are located @ ./Challenge-Object-Detection/models. Model train parameters are located @ ./CREDIT_PVAMU_CADOT_Challenge/models/train_config.yaml

# Train YOLOFocus8l, YOLOFocus11x and pretrained YOLOFocus11x weight  by running this cmd:

python train.py --pretrained_path models/yolofocus8l.yaml --datapath data/data.yaml 

python train.py --pretrained_path models/yolofocus11x.yaml --datapath data/data.yaml

python train.py --pretrained_path models/pretrained_yolofocus11x.pt --datapath data/data.yaml


# Trained model weights are saved @ CREDIT_PVAMU_CADOT_Challenge/models/CADOT_Trained_Models

## Run Inference
# This script runs predictions for the three (3) models. It then emsembles the three predictions. The emsemble prediction is saved @../results/predictions.json

python infer.py


## Note: you can directly run inference with the models trained on CADOT dataset provided for this challenge. To run inference with the trained weights; 

cd .CREDIT_PVAMU_CADOT_Challenge/scripts && vim infer.py

change the variable 'gen_path = f'{project_root}/models/CADOT_Trained_Models' to 'gen_path = f'{project_root}/models/model_weights'


## Model Evaluation
# Script evaluate models performance and saves results to ../results/metrics.csv
# GT file for evaluation is in coco format. 
python evaluation_metrics.py


