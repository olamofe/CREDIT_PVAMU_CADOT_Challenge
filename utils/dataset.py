import os
import shutil
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import data_augumentation
import class_distribution
import dataset_split
import annotation_for_visualization
import coco_yolo_format

LINK = "https://www-l2ti.univ-paris13.fr/iriser/dashboard/pages/download_CADOT_Dataset.php"




class YOLODataset(Dataset):
    def __init__(self, image_paths, label_dir, transform=None):
        self.image_paths = image_paths
        self.label_dir = Path(label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # normalize to [0, 1]

        # Load label
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append([cls, x, y, w, h])
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Apply transform (Albumentations or custom)
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes[:, 1:], class_labels=boxes[:, 0])
            img = transformed["image"]
            boxes = torch.tensor([
                [cls] + list(bbox) for cls, bbox in zip(transformed["class_labels"], transformed["bboxes"])
            ], dtype=torch.float32)

        img = torch.tensor(img).permute(2, 0, 1).float()  # HWC → CHW

        return img, boxes


def move_aug(aug_dir='../data/augmented_dir', destination_dir='./data/train'):

    # === SOURCE DIRECTORIES ===
    aug_images_dir = Path(f"{aug_dir}/images")
    aug_labels_dir = Path(f"{aug_dir}/labels")
    print(f'aug_images_dir = {aug_images_dir}')

    # === DESTINATION DIRECTORIES ===
    train_images_dir = Path(f"{destination_dir}/images")
    train_labels_dir = Path(f"{destination_dir}/labels")
    print(f'train_images_dir = {train_images_dir}')

    # === COPY AUGMENTED IMAGES ===
    for file in aug_images_dir.glob("*.*"):
        shutil.copy(file, train_images_dir / file.name)

    # === COPY AUGMENTED LABELS ===
    for file in aug_labels_dir.glob("*.txt"):
        shutil.copy(file, train_labels_dir / file.name)

    print("✅ Augmented images and labels successfully copied into train directory.")

def preprocess_dataset(data_path='', image_dir='', label_dir='',
                        aug_dir ='', destination_dir='',
                        view_annotation=False, split_dataset=True, 
                        show_class_distribution=True,
                        annotate_image='',
                        annotate_label='',
                        annotat_output_dir='',
                        run_augmentation=True,
                        move_aug_data=True,
                        remove_aug_dir=False,
                        train_size=0.95, 
                        test_size=0.05,
                        train=True, 
                        valid=False, 
                        test=False
                        ):

    if run_augmentation:
        print(f'\n\n Processing Data Augmentation =====')
        data_augumentation.run_augmentation(data_path=data_path, 
                                        image_dir=image_dir, 
                                        label_dir=label_dir)

    if move_aug_data:
        print(f'\n\nMerging augmentation directory with train directories ===')
        dest_dir = Path(str(annotate_image).replace('/images', ''))
        move_aug(aug_dir=aug_dir, destination_dir=destination_dir)

    
    if split_dataset:
        print(f'\n\n Processing dataset split =====')
        dataset_split.run_split(project_path=data_path, data_path=destination_dir, 
                                train_size=train_size, 
                                test_size=test_size)

    if view_annotation:
        print(f'\n\n Runing Annotation')
        print(f'annotation image = {annotate_image}')
        annotation_for_visualization.run_annotation(image_dir=annotate_image,
                                                    label_dir=annotate_label,
                                                    output_dir=annotat_output_dir)

    

    if show_class_distribution:
        print(f' \n\n Get class distribution')
        base_dir = Path(str(annotate_image).replace('train/images', ''))
        print(f'base dir = {base_dir}')
        class_distribution.run_distribution(base_dir=base_dir, valid=valid)

    if remove_aug_dir:
        os.system(f"rm -rf {aug_dir}")

def download_cadot(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(data_dir)
    
    os.system(f'wget -O CADOT_dataset.zip {LINK}')
    os.system(f'unzip CADOT_dataset.zip')


def load_cadot_data(base_path="Challenge-Object-Detection", dir=''):
    data_dir = Path(os.path.join(base_path, 'data'))
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
        print(f"✅ Removed directory: {data_dir}")

    download_cadot(data_dir)
    



if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]
    #print(f'project root = {project_root}')

    load_cadot_data(base_path=project_root)
    os.chdir(f"{project_root}/utils")
    print(f'CURRENT DIR = {os.getcwd()}')
    coco_yolo_format.run_coco_to_yolo_format(f'{project_root}/data')
    preprocess_dataset(data_path=f'{project_root}/data', 
                        image_dir=f'{project_root}/data/train/images',
                        label_dir=f'{project_root}/data/train/labels',
                        aug_dir=f'{project_root}/data/augmented_dir', 
                        destination_dir=f'{project_root}/data/train',
                        annotate_image=f'{project_root}/data/data_split/train/images',
                        annotate_label=f'{project_root}/data/data_split/train/labels',
                        annotat_output_dir=f'{project_root}/data/train_annotation_dir',
                        move_aug_data=True,
                        run_augmentation=True,
                        view_annotation=False,
                        valid=True
                        
                        )


    

