# evaluation/metrics_evaluation.py
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from medpy.metric.binary import dc, hd95, precision, recall
from sklearn.metrics import jaccard_score
import argparse
import sys
sys.path.append(".")

from models.custom_unet import CustomUNet
from utils.segthor_data_loader import get_dataloaders

ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta", "trachea"
]

MODEL_PATH = "checkpoints/segthor_best.pth"
ROOT_DIR = "processed_dataset"
CSV_PATH = "evaluation/metrics_summary.csv"
os.makedirs("evaluation", exist_ok=True)

def evaluate_model(split="test", threshold=0.5, model_path=None, model_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_class is not None, "model_class must be provided"
    model = model_class(num_classes=9).to(device)

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

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            output, _ = model(images)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()  # [C, H, W]
            label = labels.squeeze().cpu().numpy()  # [C, H, W]

            for i, organ in enumerate(ORGANS):
                p = (pred[i] > threshold).astype(np.uint8)
                g = label[i].astype(np.uint8)
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
        print(f"\n🫁 {organ.upper()}")
        for metric in metrics[organ]:
            values = metrics[organ][metric]
            score = np.mean(values) if values else None
            row[metric] = round(score, 4) if score is not None else "n/a"
            print(f"  {metric.capitalize()}: {row[metric]}")
            if values:
                total_metrics[metric].extend(values)
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
    df.to_csv(CSV_PATH.replace(".csv", f"_{split}.csv"), index=False)
    print(f"\n📁 Saved summary to {CSV_PATH.replace('.csv', f'_{split}.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    args = parser.parse_args()

    evaluate_model(split=args.split, threshold=args.threshold, model_class=CustomUNet)

