import os
import json
from pathlib import Path
from collections import defaultdict


def run_coco_to_yolo_format(path):
    # Define dataset folders
    base_path = Path(path)
    
    
    splits = ["train", "valid"]

    for split in splits:
        print(f"\n🔄 Processing: {split}")

        sub_dir = Path(f"{base_path}/{split}")
        image_dir = Path(os.path.join(sub_dir, "images"))
        label_dir = Path(os.path.join(sub_dir, "labels"))

        print(f'image_dir here = {image_dir} | label_dir here = {label_dir}')
        
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        coco_json = Path(os.path.join(sub_dir, "_annotations.coco.json"))
        
        if not coco_json.exists():
            print(f"❌ Missing: {coco_json}")
            continue

        with open(coco_json) as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]

        # Group annotations by image_id
        ann_by_image = defaultdict(list)
        for ann in annotations:
            if ann["iscrowd"] == 0:
                ann_by_image[ann["image_id"]].append(ann)

        #image_id_map = {img["id"]: img for img in images}

        # Write YOLO format labels
        for img in images:
            file_name = img["file_name"]
            img_id = img["id"]
            width, height = img["width"], img["height"]

            os.system(f'mv {sub_dir}/{file_name} {image_dir}')

            label_path = label_dir / f"{Path(file_name).stem}.txt"
            with open(label_path, 'w') as f:
                for ann in ann_by_image.get(img_id, []):
                    class_id = ann["category_id"]
                    x, y, w, h = ann["bbox"]

                    if w <= 0 or h <= 0:
                        continue  # skip invalid boxes

                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        os.system(f"rm {sub_dir}/_annotations.coco.json")
        print(f"✅ Labels written to: {label_dir}")


