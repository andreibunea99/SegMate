# evaluation/generate_model_report_25D.py
import os
import time
import torch
import argparse
import logging
import pandas as pd
from datetime import datetime
import sys
sys.path.append(".")

from evaluation.metrics_evaluation_25D import evaluate_model
from models.custom_unet import CustomUNet
from models.custom_unet_plus_plus import CustomUNetPlusPlus
from models.SegMateNoSE import SegMateNoSE
from models.vanilla_segmate import VanillaSegMate
from models.vanilla_segmate_fastvit import SegMateFastViT
from models.vanilla_segmate_mambaout import SegMateMambaOut
from models.vanilla_segmate_v2 import SegMate
from models.segmate_v2_film import SegMateFiLM

import torch.nn as nn

MODEL_CLASSES = {
    "CustomUNetPlusPlus": CustomUNetPlusPlus,
    "SegMateNoSE": SegMateNoSE,
    "VanillaSegMate": VanillaSegMate,
    "SegMateFastViT": SegMateFastViT,
    "SegMateMambaOut": SegMateMambaOut,
    "SegMate": SegMate,
    "SegMateFiLM": SegMateFiLM
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta", "trachea"
]


# ----------------------------
# 2.5D wrapper: 3 slices -> 1 pseudo-slice -> backbone (expects 1ch)
# Keep it local, like you did in metrics_evaluation_25D.py
# ----------------------------
class SliceFusion(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 16, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class TwoPointFiveDWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.fusion = SliceFusion(in_ch=3, mid_ch=16, out_ch=1)
        self.model = backbone

    def forward(self, x25d, z_norm=None):
        x1 = self.fusion(x25d)  # [B,1,H,W]
        if z_norm is not None and hasattr(self.model, "film"):
            return self.model(x1, z_norm=z_norm)
        return self.model(x1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpus = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        return f"{num_gpus} GPU(s): {', '.join(gpus)}"
    return "CPU only"


def measure_inference_time_and_peak_vram(model, dataloader, device):
    """
    Same idea as before (10 batches), but now also returns peak VRAM.
    Peak VRAM measured across those 10 forwards (after resetting peak stats).
    """
    model.eval()
    times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            if len(batch) == 3:
                images, _, z_norm = batch
                z_norm = z_norm.to(device)
            else:
                images, _ = batch
                z_norm = None
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            if images.size(1) != 3:
                raise RuntimeError(
                    f"[Timing] Expected 2.5D input [B,3,H,W], got {tuple(images.shape)}. "
                    f"Use balanced_dataloader_25D in generate_model_report-25D.py."
                )

            _ = model(images, z_norm=z_norm)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / max(len(times), 1)

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = None

    return avg_time, avg_time * len(dataloader), peak_vram_mb


def generate_report(model_path, deep_supervision=False, model_class=CustomUNetPlusPlus, encoder_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Model class: {model_class.__name__}")

    # Backbone is still your usual model_class (expects 1-channel)
    # Pass encoder_name only if model supports it (e.g., SegMateFastViT, SegMateMambaOut)
    models_with_encoder = ["SegMateFastViT", "SegMateMambaOut", "SegMate", "SegMateFiLM"]
    if encoder_name is not None and model_class.__name__ in models_with_encoder:
        logging.info(f"Using encoder: {encoder_name}")
        base_model = model_class(num_classes=9, deep_supervision=deep_supervision, encoder_name=encoder_name).to(device)
    else:
        if encoder_name is not None:
            logging.warning(f"encoder_name '{encoder_name}' ignored for model class {model_class.__name__}")
        base_model = model_class(num_classes=9, deep_supervision=deep_supervision).to(device)

    # Wrap into 2.5D model (expects 3-channel input)
    model = TwoPointFiveDWrapper(base_model).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # IMPORTANT: load into WRAPPER (contains fusion + backbone)
    model.load_state_dict(state_dict, strict=True)

    logging.info("Counting model parameters...")
    num_params = count_params(model)

    logging.info("Getting hardware info...")
    hardware_info = get_device_info()

    if model_class.__name__ == "SegMateFiLM":
        from dataloaders.balanced_dataloader_25D_film import get_dataloaders_balanced_film as get_dataloaders
    else:
        from dataloaders.balanced_dataloader_25D import get_dataloaders_balanced as get_dataloaders
    _, val_loader, test_loader = get_dataloaders("processed_dataset", batch_size=1, num_workers=2)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logging.info("Running evaluation on validation set (2.5D)...")
    evaluate_model(split="val", model_path=model_path, model_class=model_class, deep_supervision=deep_supervision, encoder_name=encoder_name)
    val_metrics = pd.read_csv("evaluation/metrics_summary_25D_val.csv")

    logging.info("Running evaluation on test set (2.5D)...")
    evaluate_model(split="test", model_path=model_path, model_class=model_class, deep_supervision=deep_supervision, encoder_name=encoder_name)
    test_metrics = pd.read_csv("evaluation/metrics_summary_25D_test.csv")

    logging.info("Measuring inference time + peak VRAM (approx.)...")
    inf_per_slice, inf_total, peak_vram_mb = measure_inference_time_and_peak_vram(model, test_loader, device)

    logging.info("Generating final report...\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluation/model_report_25D_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("MODEL REPORT (2.5D)\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Device Used: {hardware_info}\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"Input Shape: [1, 3, 512, 512]\n")
        f.write(f"Output Shape: [1, 9, 512, 512]\n")
        f.write(f"Validation Set Size: {val_size} slices\n")
        f.write(f"Test Set Size: {test_size} slices\n")
        f.write(f"Inference Time (1 slice): {inf_per_slice:.4f} sec\n")
        f.write(f"Estimated Full CT (300 slices): {inf_total:.2f} sec\n")

        if peak_vram_mb is not None:
            f.write(f"Peak VRAM (10-slice benchmark): {peak_vram_mb:.2f} MB\n")
        else:
            f.write("Peak VRAM (10-slice benchmark): n/a (CPU)\n")

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
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Encoder name for SegMateFastViT (e.g., fastvit_t12, fastvit_sa24). Only used with SegMateFastViT.")
    args = parser.parse_args()

    model_class = MODEL_CLASSES[args.model_type]
    generate_report(args.model, deep_supervision=args.deep_supervision, model_class=model_class, encoder_name=args.encoder_name)
