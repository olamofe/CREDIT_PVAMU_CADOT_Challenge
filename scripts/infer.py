#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, 'CREDIT_PVAMU_CADOT_Challenge/ultralytics')
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import shutil
from pathlib import Path
from ultralytics import YOLO
import json
import csv
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion


project_root = Path(__file__).resolve().parents[1]


gen_path = f'{project_root}/models/CADOT_Trained_Models'




# === CONFIGURATION ===
MODEL_PATH = 'weights/best.pt'

TEST_DIR = f'{project_root}/data/test'

output_json_path = Path(f'{project_root}/results')

output_json_path.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = "predictions.json"


CLASS_MAP = {
    0: "small_objects",
    1: "basketball field",
    2: "building",
    3: "crosswalk",
    4: "large vehicle",
    5: "ship",
    6: "medium vehicle",
    7: "small vehicle",
    8: "tennis court",
    9: "football field",
    10: "train",
    11: "small car",
    12: "playground",
    13: "roundabout",
    14: "basketball field (dup)"
}


model_list = os.listdir(gen_path)

#print(f' ========= {model_list} =======')

# === Load COCO-style JSON ===
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

image_ids = load_json(f"{project_root}/data/image_ids_cadot.json")
filename_to_id = {img["file_name"]: img["id"] for img in image_ids["images"]}


def get_categories(class_map):
    """Generate category list in COCO format."""
    return [{"id": cid, "name": name} for cid, name in class_map.items()]


def load_model(model_path):
    """Load YOLOv8 model."""
    return YOLO(model_path)


def run_prediction(model, image_dir, conf=0.01, dir=None):
    """Run inference on a folder of images."""
    return model.predict(source=image_dir, save=True, conf=conf, name=dir)


def extract_image_metadata(result):
    """Extract COCO-style image info from prediction result."""
    img_path = Path(result.path)
    #img = Image.open(img_path)
    #width, height = img.size
    file_name =  img_path.name
    image_id = filename_to_id[file_name]

    #print(filename_to_id)
    #print(f'file_name = {file_name} | image_id = {image_id}')
    

    return image_id


def extract_annotations(result, image_id, ann_id_start, score_threshold=0.25):
    """Extract COCO-style annotations from result.boxes"""
    annotations = []
    ann_id = ann_id_start

    for box in result.boxes.data.tolist():
        
        x1, y1, x2, y2, score, class_id = box[:6]

        if score < score_threshold:
            continue  # Skip low-confidence boxes

        w = x2 - x1
        h = y2 - y1

        annotations.append({
            #"id": ann_id,
            "image_id": image_id,
            "category_id": int(class_id),
            #"class_label": CLASS_MAP[class_id],
            "bbox": [x1, y1, w, h],
            #"area": w * h,
            "score": float(score)
            #"iscrowd": 0
        })
        ann_id += 1

    return annotations, ann_id


def build_coco_output(results, class_map):
    """Construct the full COCO output structure from predictions."""
    
    images = []
    annotations = []
    image_id = 0
    ann_id = 1

    #image_inference_path = results.path
    '''
    print(f'==============================================\n')
    print(f'==============================================\n')
    print(f'==============================================\n')
    print(f'==============================================\n')
    print(image_inference_path)
    llllll
    '''

    for result in results:
        
        image_id = extract_image_metadata(result)
        #images.append(img_info)

        ann_list, ann_id = extract_annotations(result, image_id, ann_id)
        annotations.extend(ann_list)
        #image_id += 1

    return annotations


def save_coco_json(output_dict, output_path):
    """Save the final COCO output dictionary to a JSON file."""
    #print(f' ************** output_path = {output_path} ************************')
    output_dict = json.dumps(output_dict, indent=2)
    with open(output_path, "w") as f:
        f.write(output_dict)
    print(f"✅ COCO-style prediction results saved to: {output_path}")


def run_detection(model_path, model_name=''):
    model = load_model(model_path)
    results = model.predict(TEST_DIR, save=False, 
                            imgsz=500, conf=0.85)
    results = results[0].to_json()
    save_coco_json(results, f"{output_json_path}/{model_name}_{OUTPUT_JSON}")

def run_pipeline(model_path, model_name=''):
    """Main runner to handle the full pipeline."""
    model = load_model(model_path)
    results = run_prediction(model, TEST_DIR, dir=model_name)
    return results




def group_results_by_image(result_lists):
    """
    Groups prediction results from multiple models by image file name.
    Input: result_lists = [[model1 results], [model2 results], ...]
    Output: {filename: [result1, result2, ...]}
    """
    grouped = defaultdict(list)
    for model_results in result_lists:
        for res in model_results:
            filename = Path(res.path).name
            grouped[filename].append(res)
    return grouped



def apply_wbf_on_grouped_results(grouped_results, filename_to_id, image_size=(500, 500), iou_thr=0.25, skip_box_thr=0.25):
    annotations = []
    

    for filename, result_list in grouped_results.items():
        all_boxes, all_scores, all_labels = [], [], []

        for res in result_list:
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
            labels = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []

            # Normalize boxes for WBF
            norm_boxes = [[x1/image_size[0], y1/image_size[1], x2/image_size[0], y2/image_size[1]]
                          for x1, y1, x2, y2 in boxes]
            
            all_boxes.append(norm_boxes)
            all_scores.append(list(scores))
            all_labels.append(list(labels))

        if not all_boxes:
            continue

        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='avg'
        )

        image_id = filename_to_id[filename]
        for box, score, label in zip(boxes_fused, scores_fused, labels_fused):
            x1 = box[0] * image_size[0]
            y1 = box[1] * image_size[1]
            x2 = box[2] * image_size[0]
            y2 = box[3] * image_size[1]
            w = x2 - x1
            h = y2 - y1

            annotations.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })
            

    return annotations




    

def run_pipeline_all_models():
    all_results_by_models = []
    
    # Step 1: Run all models
    for index, model_ in enumerate(model_list):
        print(f' \n\n {"+"*150}')
        print(f' {"+"*150}')
        
        if 'model_weights' in gen_path:
            model_name = model_
            print(f' \n Running Prediction for model --> {model_name}\n')
            model_path = f"{gen_path}/{model_}"

        else:
            model_name = model_.split('_')
            model_name = f"{model_name[3]}_Scale_{model_name[5]}"
            print(f' \n Running Prediction for model --> {model_name}\n')

            model_path = f"{gen_path}/{model_}/{MODEL_PATH}"
            print(f'model_path === > {model_path}\n')
        
        model_name = f"{output_json_path}/{model_name}"
        result = run_pipeline(model_path, model_name=model_name)
        all_results_by_models.append(result)

        print(f' {"+"*150}')
        print(f' {"+"*150}\n')
    
    
    # Step 2: Group results by image
    grouped_results = group_results_by_image(all_results_by_models)

    # Step 3: Apply WBF
    annotations = apply_wbf_on_grouped_results(grouped_results, filename_to_id)

    # Step 4: Construct final COCO-style JSON
    coco_dict = annotations
    # when iou_thr=0.5, skip_box_thr=0.25
    #save_coco_json(coco_dict, f"{output_json_path}/ensemble_predictions_0.5_0.25.json")
    save_coco_json(coco_dict, f"{output_json_path}/{OUTPUT_JSON}")
    
    #save_coco_json(image_info, f"{output_json_path}/image_ids.json")


# === RUN
if __name__ == "__main__":
    #run_pipeline()
    run_pipeline_all_models()

# %%
