# evaluation/generate_segthor_report_25D.py
"""
Generate comprehensive model evaluation report for SegTHOR dataset (2.5D).
Adapted from generate_model_report-25D.py for SegTHOR-specific evaluation.

SegTHOR Note:
-------------
- Dataset contains 3 overlapping organs: esophagus, trachea, aorta
- Other 6 channels (lungs, kidneys, liver, cord) are zero-filled
- Metrics will be computed only for present organs
- Uses processed_dataset_segthor/ as default dataset path

Usage:
------
# Evaluate pre-trained TotalSegmentator model on SegTHOR
python evaluation/generate_segthor_report_25D.py \
  --model checkpoints/segmate_film_tf_efficientnetv2_s_25D_best.pth \
  --model_type SegMateFiLM \
  --deep_supervision

# Evaluate fine-tuned SegTHOR model
python evaluation/generate_segthor_report_25D.py \
  --model checkpoints/segmate_segthor_finetune_best.pth \
  --model_type SegMateFiLM \
  --deep_supervision
"""

import os
import time
import torch
import argparse
import logging
import pandas as pd
from datetime import datetime
import sys
sys.path.append(".")

from evaluation.metrics_evaluation_segthor_25D import evaluate_model
from models.custom_unet import CustomUNet
from models.custom_unet_plus_plus import CustomUNetPlusPlus
from models.SegMateNoSE import SegMateNoSE
from models.vanilla_segmate import VanillaSegMate
from models.vanilla_segmate_fastvit import SegMateFastViT as VanillaSegMateFastViT
from models.vanilla_segmate_mambaout import SegMateMambaOut as VanillaSegMateMambaOut
from models.segmate_fastvit import SegMateFastViT
from models.segmate_mambaout import SegMateMambaOut
from models.vanilla_segmate_v2 import SegMate
from models.segmate_v2 import SegMate as SegMateV2
from models.segmate_v2_film import SegMateFiLM
from models.segmate_fastvit_film import SegMateFastViTFiLM
from models.segmate_mambaout_film import SegMateMambaOutFiLM

import torch.nn as nn

MODEL_CLASSES = {
    "CustomUNetPlusPlus": CustomUNetPlusPlus,
    "SegMateNoSE": SegMateNoSE,
    "VanillaSegMate": VanillaSegMate,
    "SegMateFastViT": SegMateFastViT,
    "SegMateMambaOut": SegMateMambaOut,
    "VanillaSegMateFastViT": VanillaSegMateFastViT,
    "VanillaSegMateMambaOut": VanillaSegMateMambaOut,
    "SegMate": SegMate,
    "SegMateV2": SegMateV2,
    "SegMateFiLM": SegMateFiLM,
    "SegMateFastViTFiLM": SegMateFastViTFiLM,
    "SegMateMambaOutFiLM": SegMateMambaOutFiLM,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Keep all 9 organs for consistency (3 with data, 6 zero-filled)
ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta", "trachea"
]


# ----------------------------
# 2.5D wrapper: 3 slices -> 1 pseudo-slice -> backbone (expects 1ch)
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
    Measure inference time and peak VRAM on 10 batches.
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
    logging.info(f"Dataset: SegTHOR (processed_dataset_segthor)")

    # Backbone is still your usual model_class (expects 1-channel)
    models_with_encoder = [
        "SegMateFastViT", "SegMateMambaOut", "SegMate", "SegMateFiLM",
        "SegMateFastViTFiLM", "SegMateMambaOutFiLM",
    ]
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

    # Load into wrapper (contains fusion + backbone)
    model.load_state_dict(state_dict, strict=True)

    logging.info("Counting model parameters...")
    num_params = count_params(model)

    logging.info("Getting hardware info...")
    hardware_info = get_device_info()

    # Use SegTHOR dataset
    _film_classes = {"SegMateFiLM", "SegMateFastViTFiLM", "SegMateMambaOutFiLM"}
    if model_class.__name__ in _film_classes:
        from dataloaders.balanced_dataloader_25D_film import get_dataloaders_balanced_film as get_dataloaders
    else:
        from dataloaders.balanced_dataloader_25D import get_dataloaders_balanced as get_dataloaders

    _, val_loader, test_loader = get_dataloaders("processed_dataset_segthor", batch_size=1, num_workers=2)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logging.info("Running evaluation on SegTHOR validation set (2.5D)...")
    evaluate_model(split="val", model_path=model_path, model_class=model_class, deep_supervision=deep_supervision, encoder_name=encoder_name)
    val_metrics = pd.read_csv("evaluation/metrics_summary_segthor_25D_val.csv")

    logging.info("Running evaluation on SegTHOR test set (2.5D)...")
    evaluate_model(split="test", model_path=model_path, model_class=model_class, deep_supervision=deep_supervision, encoder_name=encoder_name)
    test_metrics = pd.read_csv("evaluation/metrics_summary_segthor_25D_test.csv")

    logging.info("Measuring inference time + peak VRAM (approx.)...")
    inf_per_slice, inf_total, peak_vram_mb = measure_inference_time_and_peak_vram(model, test_loader, device)

    logging.info("Generating final report...\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluation/segthor_model_report_25D_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("MODEL REPORT - SegTHOR Dataset (2.5D)\n")
        f.write("="*60 + "\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Dataset: SegTHOR (3 organs: esophagus, trachea, aorta)\n")
        f.write(f"Device Used: {hardware_info}\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"Input Shape: [1, 3, H, W]\n")
        f.write(f"Output Shape: [1, 9, H, W]\n")
        f.write(f"Validation Set Size: {val_size} slices\n")
        f.write(f"Test Set Size: {test_size} slices\n")
        f.write(f"Inference Time (1 slice): {inf_per_slice:.4f} sec\n")
        f.write(f"Estimated Full CT (300 slices): {inf_total:.2f} sec\n")

        if peak_vram_mb is not None:
            f.write(f"Peak VRAM (10-slice benchmark): {peak_vram_mb:.2f} MB\n")
        else:
            f.write(f"Peak VRAM (10-slice benchmark): n/a (CPU)\n")

        f.write("\n" + "="*60 + "\n")
        f.write("VALIDATION METRICS\n")
        f.write("="*60 + "\n")
        f.write(val_metrics.to_string(index=False))
        f.write("\n\n" + "="*60 + "\n")
        f.write("TEST METRICS\n")
        f.write("="*60 + "\n")
        f.write(test_metrics.to_string(index=False))

    logging.info(f"[SUCCESS] Report saved to {report_path}")
    
    # Auto-log to experiment tracking CSV
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from scripts.log_segthor_experiment import log_experiment
        
        # Extract model name from path
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        if encoder_name:
            model_name = f"{model_name}_{encoder_name}"
        
        log_experiment(report_path, model_name=model_name, notes="Auto-logged from evaluation")
    except Exception as e:
        logging.warning(f"Could not auto-log experiment: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SegTHOR evaluation report (2.5D)"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable deep supervision during evaluation")
    parser.add_argument("--model_type", type=str, default="SegMateFiLM",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model class to use (default: SegMateFiLM)")
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Encoder name (e.g., tf_efficientnetv2_s, fastvit_sa24)")
    args = parser.parse_args()

    model_class = MODEL_CLASSES[args.model_type]
    generate_report(args.model, deep_supervision=args.deep_supervision, model_class=model_class, encoder_name=args.encoder_name)
