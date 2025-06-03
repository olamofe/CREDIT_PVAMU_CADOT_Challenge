import os
from glob import glob
from collections import defaultdict

def count_class_instances(label_dir, num_classes=15):
    class_counts = defaultdict(int)

    label_files = glob(os.path.join(label_dir, '**', '*.txt'), recursive=True)

    for file_path in label_files:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

    # Ensure all classes are present
    return {i: class_counts[i] for i in range(num_classes)}

def print_distribution(title, class_counts):
    print(f"\n📁 {title}")
    for i in sorted(class_counts.keys()):
        print(f"Class {i}: {class_counts[i]} instances")

def run_distribution(base_dir='', train=True, valid=False, test=False):

    # === Set your paths ===
    base_dir = base_dir
    folders = {
        "Train": os.path.join(base_dir, "train"),
        "Validation": os.path.join(base_dir, "valid"),           # or "valid"
        "Test ": os.path.join(base_dir, "test")
    }

    status_map = {
        "Train": train,
        "Validation": valid,           # or "valid"
        "Test ": test
    }

    
    # === Count per folder ===
    all_counts = defaultdict(int)
    for folder_name, path in folders.items():
        print(f'folder name = {folder_name}')
        if status_map.get(folder_name):
            counts = count_class_instances(path, num_classes=15)
            print_distribution(folder_name, counts)
            # Aggregate totals
            for k, v in counts.items():
                all_counts[k] += v

    # === Print overall class distribution ===
    print_distribution("Overall Distribution", all_counts)
    

#run_distribution()
