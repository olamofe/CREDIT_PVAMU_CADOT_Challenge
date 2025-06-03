import os
import cv2
import matplotlib.pyplot as plt


class_names = {
    0: "small-object",
    1: "basketball field",
    2: "building",
    3: "crosswalk",
    4: "football field",
    5: "graveyard",
    6: "large vehicle",
    7: "medium vehicle",
    8: "playground",
    9: "roundabout",
    10: "ship",
    11: "small vehicle",
    12: "swimming pool",
    13: "tennis court",
    14: "train"
}


# === DRAW ANNOTATIONS FUNCTION ===
def draw_yolo_boxes(image_path, label_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not load image: {image_path}")
        return

    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
            x1 = int((x_center - bbox_width / 2) * w)
            y1 = int((y_center - bbox_height / 2) * h)
            x2 = int((x_center + bbox_width / 2) * w)
            y2 = int((y_center + bbox_height / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names.get(class_id, str(class_id))
            cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

def run_annotation(image_dir='', label_dir='', 
                    output_dir=''):
    '''
    image_dir = "../data/dataset_split/train/images"         # Change to your image path
    label_dir = "../data/dataset_split/train/labels"         # Change to your ima         # Change to your label path
    output_dir = "../data/annotation"
    '''
    os.makedirs(output_dir, exist_ok=True)
    # === LOOP THROUGH ALL TRAIN IMAGES ===
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(label_path):
            draw_yolo_boxes(image_path, label_path, output_path)

    print(f"✅ All annotated images saved to {output_dir}")



#run_annotation()

