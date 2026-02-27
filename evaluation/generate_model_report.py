# evaluation/generate_model_report.py
import os
import time
import torch
import argparse
import logging
import pandas as pd
from datetime import datetime
import sys
sys.path.append(".")

from evaluation.metrics_evaluation_unetpp import evaluate_model
from models.custom_unet import CustomUNet
from models.custom_unet_plus_plus import CustomUNetPlusPlus
from models.SegMateNoSE import SegMateNoSE
from models.vanilla_segmate import VanillaSegMate

MODEL_CLASSES = {
    "CustomUNetPlusPlus": CustomUNetPlusPlus,
    "SegMateNoSE": SegMateNoSE,
    "VanillaSegMate": VanillaSegMate,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta", "trachea"
]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpus = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        return f"{num_gpus} GPU(s): {', '.join(gpus)}"
    return "CPU only"


def measure_inference_time(model, dataloader, device):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= 10:
                break
            images = images.to(device)
            start = time.time()
            _ = model(images)
            end = time.time()
            times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time, avg_time * len(dataloader)


def generate_report(model_path, deep_supervision=False, model_class=CustomUNetPlusPlus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Model class: {model_class.__name__}")

    # Update num_classes from 10 to 9 (removed heart)
    model = model_class(num_classes=9, deep_supervision=deep_supervision).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    logging.info("Counting model parameters...")
    num_params = count_params(model)

    logging.info("Getting hardware info...")
    hardware_info = get_device_info()

    from dataloaders.balanced_dataloader import get_dataloaders_balanced as get_dataloaders
    _, val_loader, test_loader = get_dataloaders("processed_dataset", batch_size=1, num_workers=2)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logging.info("Running evaluation on validation set...")
    evaluate_model(split="val", model_path=model_path, model_class=model_class, deep_supervision=deep_supervision)
    val_metrics = pd.read_csv("evaluation/metrics_summary_val.csv")

    logging.info("Running evaluation on test set...")
    evaluate_model(split="test", model_path=model_path, model_class=model_class, deep_supervision=deep_supervision)
    test_metrics = pd.read_csv("evaluation/metrics_summary_test.csv")

    logging.info("Measuring inference time (approx.)...")
    inf_per_slice, inf_total = measure_inference_time(model, test_loader, device)

    logging.info("Generating final report...\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluation/model_report_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("MODEL REPORT\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Device Used: {hardware_info}\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"Input Shape: [1, 1, 512, 512]\n")
        # Updated output shape to 10 channels
        f.write(f"Output Shape: [1, 10, 512, 512]\n")
        f.write(f"Validation Set Size: {val_size} slices\n")
        f.write(f"Test Set Size: {test_size} slices\n")
        f.write(f"Inference Time (1 slice): {inf_per_slice:.4f} sec\n")
        f.write(f"Estimated Full CT (300 slices): {inf_total:.2f} sec\n")
        f.write("\n--- VALIDATION METRICS ---\n")
        f.write(val_metrics.to_string(index=False))
        f.write("\n\n--- TEST METRICS ---\n")
        f.write(test_metrics.to_string(index=False))

    logging.info(f"✅ Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable deep supervision during evaluation")
    parser.add_argument("--model_type", type=str, default="CustomUNetPlusPlus",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model class to use (default: CustomUNetPlusPlus)")
    args = parser.parse_args()

    model_class = MODEL_CLASSES[args.model_type]
    generate_report(args.model, deep_supervision=args.deep_supervision, model_class=model_class)

