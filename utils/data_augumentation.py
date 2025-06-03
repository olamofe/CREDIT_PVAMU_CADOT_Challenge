import os
import cv2
import json
import random
import numpy as np
import albumentations as A
from pathlib import Path

# === CONFIG ===
image_dir = "../data/train/images"
label_dir = "../data/train/labels"

output_image_dir = Path("../data/augmented_dir/images")
output_label_dir = Path("../data/augmented_dir/labels")

output_image_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

rare_class_ids = set([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # e.g., large/medium vehicles


# === BBOX SANITIZATION FUNCTION ===
def clamp_bbox(box):
    return [
        max(0.0, min(1.0, box[0])),  # x_center
        max(0.0, min(1.0, box[1])),  # y_center
        max(0.0, min(1.0, box[2])),  # width
        max(0.0, min(1.0, box[3]))   # height
    ]


# === LOAD AUGMENTATION WEIGHTS ===
augmentation_weight = {
    1: 7,   # basketball field (very rare)
    14: 3,  # basketball field (duplicate ID)
    4: 3,   # large vehicle
    5: 3,   # ship
    8: 2,   # tennis court
    9: 2,   # football field
    12: 2,  # playground
    13: 2   # roundabout
}



# === AUGMENTATION PIPELINE ===

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomSizedBBoxSafeCrop(height=500, width=500, p=0.5),
    A.Cutout(num_holes=4, max_h_size=20, max_w_size=20, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    


# === COPY-PASTE AUGMENTATION ===
def copy_paste_objects(base_img, base_bboxes, base_labels, src_img, src_bboxes, src_labels, max_objects=3):
    h, w = base_img.shape[:2]
    for i in range(min(len(src_bboxes), max_objects)):
        box = src_bboxes[i]
        cls = src_labels[i]
        x_center, y_center, bw, bh = box
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)
        obj_crop = src_img[y1:y2, x1:x2]
        paste_x = random.randint(0, w - (x2 - x1))
        paste_y = random.randint(0, h - (y2 - y1))
        base_img[paste_y:paste_y + (y2 - y1), paste_x:paste_x + (x2 - x1)] = obj_crop
        new_x_center = (paste_x + (x2 - x1) / 2) / w
        new_y_center = (paste_y + (y2 - y1) / 2) / h
        new_bw = (x2 - x1) / w
        new_bh = (y2 - y1) / h
        base_bboxes.append(clamp_bbox([new_x_center, new_y_center, new_bw, new_bh]))
        base_labels.append(cls)
    return base_img, base_bboxes, base_labels

# === MAIN AUGMENTATION FUNCTION ===
def augment_yolo_image_weighted(image_path, label_path, all_data):
    img = cv2.imread(image_path)
    if img is None:
        return
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 🔄 MODIFIED: Only use labels with class IDs in augmentation_weight
    rare_lines = [line for line in lines if int(line.split()[0]) in augmentation_weight]
    if not rare_lines:
        return

    max_multiplier = max([augmentation_weight[int(line.split()[0])] for line in rare_lines])

    for i in range(max_multiplier):
        bboxes, class_labels = [], []
        for line in rare_lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
            class_labels.append(class_id)

        # Copy-paste from a random rare-class image
        src_path, src_lbl = random.choice(all_data)
        src_img = cv2.imread(src_path)
        with open(src_lbl, 'r') as f:
            src_lines = f.readlines()
        src_bboxes, src_labels = [], []
        for sline in src_lines:
            sp = sline.strip().split()
            sid = int(sp[0])
            if sid in augmentation_weight:  # ✅ ADDED: filter only rare classes
                src_bboxes.append(list(map(float, sp[1:])))
                src_labels.append(sid)

        if src_img is not None and src_bboxes:
            img_cp, bboxes_cp, class_labels_cp = copy_paste_objects(
                img.copy(), bboxes.copy(), class_labels.copy(), src_img, src_bboxes, src_labels
            )
        else:
            img_cp, bboxes_cp, class_labels_cp = img.copy(), bboxes.copy(), class_labels.copy()
        
        '''
        print("CLASS LABELS TYPE:", type(class_labels_cp))
        print("CLASS LABELS CONTENT:", class_labels_cp)
        print("ELEMENT TYPE:", [type(c) for c in class_labels_cp])
        print("BBOXES:", bboxes_cp)
        print("LABELS:", class_labels_cp)
        '''

        try:
            
            augmented = augment(image=img_cp, bboxes=bboxes_cp, class_labels=class_labels_cp)
        
        except Exception as e:
            print(f"Augmentation failed for {image_path}: {e}")
            continue

        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        base = os.path.splitext(os.path.basename(image_path))[0]
        aug_img_name = os.path.join(output_image_dir, f"{base}_aug_{i}.jpg")
        aug_lbl_name = os.path.join(output_label_dir, f"{base}_aug_{i}.txt")

        cv2.imwrite(aug_img_name, aug_img)
        with open(aug_lbl_name, 'w') as f:
            for label, box in zip(aug_labels, aug_bboxes):
                f.write(f"{label} {' '.join(map(str, box))}\n")

# === GATHER ALL RARE-LABEL IMAGES ===  # 🔄 MODIFIED (reuse original structure)

def run_augmentation(data_path='', image_dir='', label_dir=''):
    

    output_image_dir = Path(f"{data_path}/augmented_dir/images")
    output_label_dir = Path(f"{data_path}/augmented_dir/labels")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    all_data = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            label_file = os.path.splitext(file)[0] + ".txt"
            if os.path.exists(os.path.join(label_dir, label_file)):
                all_data.append((os.path.join(image_dir, file), os.path.join(label_dir, label_file)))

    # === RUN AUGMENTATION ===
    for image_path, label_path in all_data:
        augment_yolo_image_weighted(image_path, label_path, all_data)

    print("✅ Augmentation Task Completed.")


