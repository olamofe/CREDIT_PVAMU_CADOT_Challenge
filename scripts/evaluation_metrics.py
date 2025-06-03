import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]

def box_iou(box1, box2):
    """Compute IoU between two boxes: [x1, y1, x2, y2]"""
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def load_ground_truth(gt_dir):
    gt = defaultdict(list)
    for label_file in glob.glob(os.path.join(gt_dir, "*.txt")):
        image_id = int(os.path.basename(label_file).split(".")[0])
        with open(label_file) as f:
            for line in f:
                cls, x1, y1, x2, y2 = map(float, line.strip().split())
                gt[image_id].append([int(cls), x1, y1, x2, y2])
    return gt

def evaluate(pred_json_path, gt_dir, output_csv='metrics.csv', iou_thresholds=np.arange(0.5, 1.0, 0.05), conf_threshold=0.001):
    with open(pred_json_path) as f:
        preds = json.load(f)

    gt_data = load_ground_truth(gt_dir)
    classwise_gt_count = defaultdict(int)
    for boxes in gt_data.values():
        for box in boxes:
            classwise_gt_count[box[0]] += 1

    results = defaultdict(lambda: {
        'tp': defaultdict(list),
        'fp': defaultdict(list),
        'conf': [],
        'n_gt': 0
    })

    for pred in sorted(preds, key=lambda x: -x['score']):
        if pred['score'] < conf_threshold:
            continue
        image_id = pred['image_id']
        cls = pred['category_id']
        x, y, w, h = pred['bbox']
        pred_box = [x, y, x + w, y + h]

        gt_boxes = gt_data.get(image_id, [])
        ious = [box_iou(pred_box, gt_box[1:]) if cls == gt_box[0] else 0 for gt_box in gt_boxes]
        max_iou = max(ious) if ious else 0
        max_index = np.argmax(ious) if ious else -1

        for iou_thr in iou_thresholds:
            match_key = (image_id, cls, max_index, iou_thr)
            if max_iou >= iou_thr and match_key not in results[cls]['tp'][iou_thr]:
                results[cls]['tp'][iou_thr].append(1)
                results[cls]['fp'][iou_thr].append(0)
            else:
                results[cls]['tp'][iou_thr].append(0)
                results[cls]['fp'][iou_thr].append(1)
        results[cls]['conf'].append(pred['score'])

    # Aggregate per-class metrics
    rows = []
    for cls, data in results.items():
        ap_list = []
        ar_list = []
        for iou_thr in iou_thresholds:
            tp = np.array(data['tp'][iou_thr])
            fp = np.array(data['fp'][iou_thr])
            n_gt = classwise_gt_count[cls]
            if n_gt == 0:
                continue
            sorted_idx = np.argsort(-np.array(data['conf']))
            tp = tp[sorted_idx]
            fp = fp[sorted_idx]
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recall = tp_cumsum / (n_gt + 1e-6)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            ap = compute_ap(recall, precision)
            ar = recall[-1] if len(recall) > 0 else 0
            ap_list.append(ap)
            ar_list.append(ar)

        ap50 = ap_list[0] if ap_list else 0
        ap_avg = np.mean(ap_list) if ap_list else 0
        ar50 = ar_list[0] if ar_list else 0
        rows.append({
            "class": cls,
            "AP@50": round(ap50, 4),
            "AP@[.5:.95]": round(ap_avg, 4),
            "AR@50": round(ar50, 4),
            "GT_Count": classwise_gt_count[cls]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved per-class metrics to: {output_csv}")
    print(df)

if __name__ == "__main__":
    evaluate(
        pred_json_path=f"{project_root}/results/predictions.json",
        gt_dir=f"{project_root}/data/test/_annotations.coco.json",   # Replace with your GT .txt directory
        output_csv=f"{project_root}/results/metrics.csv",
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        conf_threshold=0.001
    )

