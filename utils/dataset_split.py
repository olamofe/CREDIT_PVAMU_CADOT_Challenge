import os
import shutil
import random
from collections import defaultdict, Counter
from pathlib import Path

# === CONFIG ===



# === STEP 1: Parse label files and build reverse index by class ===
class_to_images = defaultdict(set)
image_to_classes = defaultdict(set)
image_exts = [".jpg", ".jpeg", ".png"]

def process_dataset(data_path='', train_size=0.95, test_size=0.05):

    combined_images_dir = Path(f"{data_path}/images")
    combined_labels_dir = Path(f"{data_path}/labels")

    print(f'image dir = {combined_images_dir}')

    for label_file in combined_labels_dir.glob("*.txt"):
        image_id = label_file.stem
        image_file = None
        for ext in image_exts:
            candidate = combined_images_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        if not image_file:
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        classes = {int(line.split()[0]) for line in lines}
        for cls in classes:
            class_to_images[cls].add(image_file)
        image_to_classes[image_file] = classes

    # === STEP 2: Sort classes by rarity (ascending count) ===
    class_counts = {cls: len(images) for cls, images in class_to_images.items()}
    sorted_classes = sorted(class_counts, key=lambda c: class_counts[c])  # rarest first

    # === STEP 3: Assign images to train/test prioritizing rare classes ===
    used_images = set()
    train_set, test_set = set(), set()
    image_pool = set(image_to_classes.keys())

    for cls in sorted_classes:
        cls_images = list(class_to_images[cls])
        random.shuffle(cls_images)
        n_total = len(cls_images)
        n_test = max(1, int(n_total * test_size))  # ensure at least 1 test sample
        test_candidates = [img for img in cls_images if img not in used_images]
        test_selected = test_candidates[:n_test]
        train_selected = [img for img in cls_images if img not in used_images and img not in test_selected]

        # Assign test
        for img in test_selected:
            test_set.add(img)
            used_images.add(img)

        # Assign train
        for img in train_selected:
            train_set.add(img)
            used_images.add(img)

    # === STEP 4: Assign remaining unassigned images randomly (to fill gaps) ===
    remaining_images = list(image_pool - used_images)
    random.shuffle(remaining_images)
    split_idx = int(len(remaining_images) * train_size)
    train_set.update(remaining_images[:split_idx])
    test_set.update(remaining_images[split_idx:])

    return train_set, test_set

# === STEP 5: Copy files to respective folders ===
def copy_to_split(images, target_image_dir, target_label_dir, data_path):

    combined_labels_dir = Path(f"{data_path}/labels")
    print(f'combined labels_dir = {combined_labels_dir}')
    for img_path in images:
        lbl_path = combined_labels_dir / f"{img_path.stem}.txt"
        shutil.copy(img_path, target_image_dir / img_path.name)
        shutil.copy(lbl_path, target_label_dir / lbl_path.name)

def run_split(project_path='', data_path='', 
                train_size=0.95, 
                test_size=0.05):

    train_images_dir = Path(f"{project_path}/data_split/train/images")
    train_labels_dir = Path(f"{project_path}/data_split/train/labels")
    test_images_dir = Path(f"{project_path}/data_split/test/images")
    test_labels_dir = Path(f"{project_path}/data_split/test/labels")

    # Create output directories
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)


    train_set, test_set = process_dataset(data_path=data_path, train_size=train_size, 
                                            test_size=test_size)
    print(f'len of set = {len(train_set)}')

    copy_to_split(train_set, train_images_dir, train_labels_dir, data_path)
    copy_to_split(test_set, test_images_dir, test_labels_dir, data_path)

    print(f"✅ Train/Test split complete.")
    print(f"Train images: {len(train_set)}")
    print(f"Test images: {len(test_set)}")

'''
if __name__ == "__main__":
    run_split()
'''

