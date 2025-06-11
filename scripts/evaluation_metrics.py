import sys
sys.path.insert(0, 'CREDIT_PVAMU_CADOT_Challenge/ultralytics')
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json

from collections import defaultdict
import numpy as np
import torch

from ultralytics.utils import SimpleClass, TryExcept, checks, plt_settings


project_root = Path(__file__).resolve().parents[1]

def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU



def compute_ap(recall: List[float], precision: List[float]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: Dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> Tuple:
    """
    Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (Dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for Ultralytics YOLO models.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initialize a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self) -> Union[np.ndarray, List]:
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> Union[np.ndarray, List]:
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """
        Return the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """
        Return the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """
        Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> List[float]:
        """Return mean of results, mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> Tuple[float, float, float, float]:
        """Return class-aware result, p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        """Return mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        """Return model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.nan_to_num(np.array(self.mean_results())) * w).sum()

    def update(self, results: tuple):
        """
        Update the evaluation metrics with a new set of results.

        Args:
            results (tuple): A tuple containing evaluation metrics:
                - p (list): Precision for each class.
                - r (list): Recall for each class.
                - f1 (list): F1 score for each class.
                - all_ap (list): AP scores for all classes and all IoU thresholds.
                - ap_class_index (list): Index of class for each AP score.
                - p_curve (list): Precision curve for each class.
                - r_curve (list): Recall curve for each class.
                - f1_curve (list): F1 curve for each class.
                - px (list): X values for the curves.
                - prec_values (list): Precision values for each class.
        
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results
        """

        (
        self.tp,             # results[0]
        self.fp,             # results[1]
        self.p,              # results[2]
        self.r,              # results[3]
        self.f1,             # results[4]
        self.all_ap,         # results[5]
        self.ap_class_index, # results[6]
        self.p_curve,        # results[7]
        self.r_curve,        # results[8]
        self.f1_curve,       # results[9]
        self.px,             # results[10]
        self.prec_values,    # results[11]
        ) = results

    @property
    def curves(self) -> List:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> List[List]:
        """Return a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]



def build_names_instance_count(gt_path):
    

    # === Load COCO groundtruth.json ===
    with open(gt_path, "r") as f:
        coco_gt = json.load(f)

    # === Build names = {class_id: class_name} ===
    names = {cat["id"]: cat["name"] for cat in coco_gt["categories"]}

    # === Build instance_count = {class_id: total number of annotations (instances)} ===
    instance_count = defaultdict(int)
    for ann in coco_gt["annotations"]:
        class_id = ann["category_id"]
        instance_count[class_id] += 1

    # === Build image_to_class_count = {class_id: number of unique images that contain this class} ===
    # This requires collecting image_ids per class, then counting the unique ones
    class_to_image_ids = defaultdict(set)
    for ann in coco_gt["annotations"]:
        class_id = ann["category_id"]
        image_id = ann["image_id"]
        class_to_image_ids[class_id].add(image_id)

    # Now count unique images per class
    image_to_class_count = {class_id: len(image_ids) for class_id, image_ids in class_to_image_ids.items()}

    return names, instance_count, image_to_class_count




def eval(groundtruth_path, predictions_path):
    with open(groundtruth_path) as f:
        gt_data = json.load(f)

    
    # Load predictions
    with open(predictions_path) as f:
        preds = json.load(f)
    
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []

    # Create a mapping from image_id → list of (bbox, category_id)
    image_to_gt = {}
    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        image_to_gt.setdefault(img_id, []).append((ann['bbox'], ann['category_id']))
    
    

    # Build image_id -> gt_boxes/class mapping
    '''
    gt_data_ = {}
    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_data:
            gt_data[img_id] = []
        gt_data_[img_id].append((ann['bbox'], ann['category_id']))  # bbox is [x, y, w, h]
    '''

    gt_data_ = defaultdict(list)

    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        gt_data_[img_id].append((ann['bbox'], ann['category_id']))  # bbox is [x, y, w, h]

    for pred in preds:
        img_id = pred["image_id"]
        pred_box = torch.tensor(pred["bbox"])  # [x, y, w, h]
        pred_class = pred["category_id"]
        conf = pred["score"]

        # Compare to all GT boxes in the same image
        gts = gt_data_.get(img_id, [])
        ious = []
        matched = False
        for gt_box, gt_class in gts:
            if gt_class != pred_class:
                continue
            iou = bbox_iou(pred_box.unsqueeze(0), torch.tensor(gt_box).unsqueeze(0), xywh=True)
            ious.append(iou.item())
        
        ious = np.array(ious)
        #tp = [(iou >= t).astype(float).max() if len(ious) else 0.0 for t in iou_thresholds]
        tp = [(iou >= t).float().max().item() if iou.numel() > 0 else 0.0 for t in iou_thresholds]

        tp_list.append(tp)
        conf_list.append(conf)
        pred_cls_list.append(pred_class)

        # Record target class for recall calculation
        for gt_box, gt_class in gts:
            target_cls_list.append(gt_class)

    # Convert lists to arrays
    tp = np.array(tp_list)
    conf = np.array(conf_list)
    pred_cls = np.array(pred_cls_list)
    target_cls = np.array(target_cls_list)

    return tp, conf, pred_cls, target_cls 


def print_metrics_table(metric, names, image_to_class_count, instance_count):
    headers = ["Class", "Images", "Instances", "Box(P)", "R", "mAP50", "mAP50-95"]
    row_format = "{:<18} {:>6} {:>10} {:>8} {:>6} {:>8} {:>10}"

    print(row_format.format(*headers))

    # All metrics row
    print(row_format.format(
        "all",
        sum(image_to_class_count.values()),
        sum(instance_count.values()),
        f"{metric.mp:.3f}",
        f"{metric.mr:.3f}",
        f"{metric.map50:.3f}",
        f"{metric.map:.3f}",
    ))

    for i, cls_id in enumerate(metric.ap_class_index):
        cls_name = names.get(cls_id, str(cls_id))
        images = image_to_class_count.get(cls_id, 0)
        instances = instance_count.get(cls_id, 0)
        precision = metric.p[i]
        recall = metric.r[i]
        ap50 = metric.ap50[i]
        ap5095 = metric.ap[i]

        print(row_format.format(
            cls_name,
            images,
            instances,
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{ap50:.3f}",
            f"{ap5095:.3f}",
        ))

    


def save_metrics_csv(metric, names, image_to_class_count, instance_count, save_path="metrics.csv"):
    rows = []

    # Add "all" row (summary)
    rows.append({
        "Class": "all",
        "Images": sum(image_to_class_count.values()),
        "Instances": sum(instance_count.values()),
        "Box(P)": round(metric.mp, 3),
        "R": round(metric.mr, 3),
        "mAP50": round(metric.map50, 3),
        "mAP50-95": round(metric.map, 3),
    })

    # Add per-class rows
    for i, cls_id in enumerate(metric.ap_class_index):
        cls_name = names.get(cls_id, str(cls_id))
        images = image_to_class_count.get(cls_id, 0)
        instances = instance_count.get(cls_id, 0)
        precision = metric.p[i]
        recall = metric.r[i]
        ap50 = metric.ap50[i]
        ap5095 = metric.ap[i]

        rows.append({
            "Class": cls_name,
            "Images": images,
            "Instances": instances,
            "Box(P)": round(precision, 3),
            "R": round(recall, 3),
            "mAP50": round(ap50, 3),
            "mAP50-95": round(ap5095, 3),
        })

    # Convert to DataFrame and print
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\n✅ Metrics saved to: {save_path}")





if __name__ == "__main__":
    # Your code here
    groundtruth_path = f"{project_root}/data/test/_annotations.coco.json"
    predictions_path = f"{project_root}/results/predictions.json"
    tp, conf, pred_cls, target_cls = eval(groundtruth_path, predictions_path)
    metric = Metric()
    results = ap_per_class(tp, conf, pred_cls, target_cls)
    
    print(f"Length of results: {len(results)}")
    
    print(f"[DEBUG] ap.shape (should be 2D): {results[5].shape}")  # results[5] = ap
    print(f"[DEBUG] tp.shape: {tp.shape}")  # from eval()
    
    metric.update(results)

    names, image_to_class_count, instance_count = build_names_instance_count(groundtruth_path)
    print_metrics_table(metric, names, image_to_class_count, instance_count)
    save_metrics_csv(metric, names, image_to_class_count, instance_count, save_path=f"{project_root}/results/metrics.csv")


