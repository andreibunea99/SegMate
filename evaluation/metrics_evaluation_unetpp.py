# evaluation/metrics_evaluation_segmate-25D.py
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from medpy.metric.binary import dc, hd95, precision, recall
from sklearn.metrics import jaccard_score, accuracy_score, roc_auc_score
import argparse
import sys
sys.path.append(".")

from dataloaders.balanced_dataloader import get_dataloaders_balanced as get_dataloaders

ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta","trachea"
]

MODEL_PATH = "checkpoints/segthorpp_best.pth"
ROOT_DIR = "processed_dataset"
CSV_PATH = "evaluation/metrics_summary.csv"
os.makedirs("evaluation", exist_ok=True)

def evaluate_model(split="test", threshold=0.5, model_path=None, model_class=None, deep_supervision=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_class is not None, "model_class must be provided"
    model = model_class(num_classes=9, deep_supervision=deep_supervision).to(device)

    model_path = model_path or MODEL_PATH
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    train_loader, val_loader, test_loader = get_dataloaders(ROOT_DIR, batch_size=1, num_workers=2)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataloader = loader_map.get(split)

    if dataloader is None:
        raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', or 'test'.")

    print(f"\n🔍 Evaluating on {split.upper()} set with threshold={threshold}")

    metrics = {organ: {"dice": [], "hd95": [], "iou": [], "precision": [], "recall": []} for organ in ORGANS}
    presence_metrics = {organ: {"y_true": [], "y_pred": [], "y_prob": []} for organ in ORGANS}

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                pred_masks = outputs[0]  # CustomUNetPlusPlus returns multiple outputs
            else:
                pred_masks = outputs  # CustomUNet returns just the mask

            pred = torch.sigmoid(pred_masks).squeeze().cpu().numpy()  # [C, H, W]
            label = labels.squeeze().cpu().numpy()  # [C, H, W]

            for i, organ in enumerate(ORGANS):
                p = (pred[i] > threshold).astype(np.uint8)
                g = label[i].astype(np.uint8)
                
                # Presence detection metrics (for all slices)
                gt_present = int(np.sum(g) > 0)
                pred_present = int(np.sum(p) > 0)
                pred_prob_presence = np.max(pred[i])  # Max probability as presence confidence
                
                presence_metrics[organ]["y_true"].append(gt_present)
                presence_metrics[organ]["y_pred"].append(pred_present)
                presence_metrics[organ]["y_prob"].append(pred_prob_presence)
                
                # Segmentation metrics (only when organ is present)
                if np.sum(g) == 0:
                    continue
                try:
                    metrics[organ]["dice"].append(dc(p, g))
                    metrics[organ]["hd95"].append(hd95(p, g))
                    metrics[organ]["iou"].append(jaccard_score(g.flatten(), p.flatten()))
                    metrics[organ]["precision"].append(precision(p, g))
                    metrics[organ]["recall"].append(recall(p, g))
                except Exception as e:
                    print(f"⚠️ {organ.upper()} failed: {e}")
                    continue

    print("\n📊 Full Evaluation Report:")
    rows = []
    total_metrics = {"dice": [], "hd95": [], "iou": [], "precision": [], "recall": []}

    for organ in ORGANS:
        row = {"organ": organ}
        print(f"\n🤁 {organ.upper()}")
        
        # Segmentation metrics
        for metric in metrics[organ]:
            values = metrics[organ][metric]
            score = np.mean(values) if values else None
            row[metric] = round(score, 4) if score is not None else "n/a"
            print(f"  {metric.capitalize()}: {row[metric]}")
            if values:
                total_metrics[metric].extend(values)
        
        # Presence detection metrics
        y_true = np.array(presence_metrics[organ]["y_true"])
        y_pred = np.array(presence_metrics[organ]["y_pred"])
        y_prob = np.array(presence_metrics[organ]["y_prob"])
        
        if len(np.unique(y_true)) > 1:  # Only if we have both present/absent cases
            presence_acc = accuracy_score(y_true, y_pred)
            presence_auc = roc_auc_score(y_true, y_prob)
            
            # Calculate precision/recall for presence detection
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            presence_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            presence_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            row["presence_acc"] = round(presence_acc, 4)
            row["presence_precision"] = round(presence_precision, 4)
            row["presence_recall"] = round(presence_recall, 4)
            row["presence_auc"] = round(presence_auc, 4)
            
            print(f"  Presence Accuracy: {row['presence_acc']}")
            print(f"  Presence Precision: {row['presence_precision']}")
            print(f"  Presence Recall: {row['presence_recall']}")
            print(f"  Presence AUC: {row['presence_auc']}")
        else:
            row["presence_acc"] = "n/a"
            row["presence_precision"] = "n/a"
            row["presence_recall"] = "n/a"
            row["presence_auc"] = "n/a"
            print("  Presence metrics: n/a (organ always present/absent)")
            
        rows.append(row)

    # Calculate total averages
    total_row = {"organ": "TOTAL"}
    print("\n📊 AVERAGE OVER ALL ORGANS:")
    for metric in total_metrics:
        if total_metrics[metric]:
            score = np.mean(total_metrics[metric])
            total_row[metric] = round(score, 4)
            print(f"  {metric.capitalize()}: {total_row[metric]}")
        else:
            total_row[metric] = "n/a"
    rows.append(total_row)

    df = pd.DataFrame(rows)
    csv_final = CSV_PATH.replace(".csv", f"_{split}.csv")
    df.to_csv(csv_final, index=False)
    print(f"\n📁 Saved summary to {csv_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable if model uses deep supervision")
    args = parser.parse_args()

    from models.custom_unet_plus_plus import CustomUNetPlusPlus
    evaluate_model(
        split=args.split,
        threshold=args.threshold,
        model_class=CustomUNetPlusPlus,
        deep_supervision=args.deep_supervision
    )
