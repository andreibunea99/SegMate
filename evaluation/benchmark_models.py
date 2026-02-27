# evaluation/benchmark_models.py
# Full benchmark for 2D vs 2.5D CT slice OAR segmentation (your SegMate setup).
# Metrics:
# - avg Dice (macro), avg HD95 (macro)
# - latency per slice (mean/p50/p95), throughput
# - peak VRAM per inference (max/mean)
# - params (M), ckpt size (MB)

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt
from fvcore.nn import FlopCountAnalysis

# same style as your eval scripts
sys.path.append(".")

from dataloaders.balanced_dataloader import get_dataloaders_balanced as get_dataloaders_2d
from dataloaders.balanced_dataloader_25D import get_dataloaders_balanced as get_dataloaders_25d

from models.custom_unet_plus_plus import CustomUNetPlusPlus
from models.SegMateNoSE import SegMateNoSE
from models.vanilla_segmate import VanillaSegMate
from models.segmate_fastvit import SegMateFastViT
from models.segmate_v2 import SegMate
from models.segmate_v2_film import SegMateFiLM
from models.segmate_mambaout import SegMateMambaOut
from models.vanilla_segmate_v2 import SegMate as VanillaSegMateV2
from models.vanilla_segmate_mambaout import SegMateMambaOut as VanillaSegMateMambaOut
from models.vanilla_segmate_fastvit import SegMateFastViT as VanillaSegMateFastViT
from models.unet_baseline import UNetBaseline


# ----------------------------
# 2.5D wrapper (same as your report/eval)
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

    def forward(self, x25d):
        x1 = self.fusion(x25d)  # [B,1,H,W]
        return self.model(x1)


# ----------------------------
# Registry
# ----------------------------
@dataclass
class ModelSpec:
    model_class: Callable[..., torch.nn.Module]
    ckpt_path: str
    model_type: str                 # "2D" or "25D"
    model_kwargs: Dict[str, Any]
    name: Optional[str] = None
    force_deep_supervision: Optional[bool] = None   # None=auto detect from ckpt


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_size_mb(path: str) -> float:
    return (os.path.getsize(path) / (1024 ** 2)) if os.path.isfile(path) else float("nan")


@torch.inference_mode()
def compute_gflops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """Compute GFLOPs for a single forward pass using fvcore."""
    flops = FlopCountAnalysis(model, input_tensor)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9


def avg_slices_per_volume(data_root: str, split: str = "Test") -> float:
    """Count average number of axial slices per patient volume from chunk files."""
    img_dir = os.path.join(data_root, f"images{split}")
    if not os.path.isdir(img_dir):
        return float("nan")
    vol_slices: Dict[str, int] = {}
    for fname in os.listdir(img_dir):
        if not fname.endswith(".npz"):
            continue
        patient_id = fname.rsplit("_chunk", 1)[0]
        npz = np.load(os.path.join(img_dir, fname))
        vol_slices[patient_id] = vol_slices.get(patient_id, 0) + len(npz["indices"])
        npz.close()
    if not vol_slices:
        return float("nan")
    return float(np.mean(list(vol_slices.values())))


def _load_state_dict(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def _ckpt_has_deep_supervision(state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("deep_head") for k in state.keys())


def _filter_state_dict_for_model(model: nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Drop keys that don't exist in model OR have mismatched shapes.
    This makes loading robust across small head diffs.
    """
    model_sd = model.state_dict()
    out = {}
    for k, v in state.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        out[k] = v
    return out


def build_and_load_model(
    spec: ModelSpec,
    device: torch.device,
    num_classes: int,
    use_25d_wrapper: bool,
) -> torch.nn.Module:
    """
    - Instantiates backbone with num_classes from data (auto).
    - deep_supervision: auto-detect from checkpoint unless forced.
    - For 2.5D: wraps backbone with fusion wrapper, and loads checkpoint into wrapper (like your scripts).
    """
    state = _load_state_dict(spec.ckpt_path, device)

    deep_sup = spec.force_deep_supervision
    if deep_sup is None:
        deep_sup = _ckpt_has_deep_supervision(state)

    # build backbone
    kwargs = dict(spec.model_kwargs or {})
    kwargs["num_classes"] = num_classes

    # some of your models accept deep_supervision; others might not
    backbone = None
    try:
        backbone = spec.model_class(**kwargs, deep_supervision=deep_sup).to(device)
    except TypeError:
        backbone = spec.model_class(**kwargs).to(device)

    if use_25d_wrapper:
        model = TwoPointFiveDWrapper(backbone).to(device)
    else:
        model = backbone

    # strict load first; if fails, do safe filtered load
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # fallback: filter incompatible keys (deep heads / wrong class heads)
        filtered = _filter_state_dict_for_model(model, state)
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        print("\n[WARN] strict=True failed, loaded with filtered strict=False.")
        print(f"       Reason: {str(e).splitlines()[0]}")
        if unexpected:
            print(f"       Unexpected keys dropped: {len(unexpected)}")
        if missing:
            print(f"       Missing keys (not loaded): {len(missing)}")

    model.eval()
    return model


# ----------------------------
# Metrics
# ----------------------------
def dice_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool = True) -> np.ndarray:
    dices = np.full((num_classes,), np.nan, dtype=np.float64)
    cls_range = range(1, num_classes) if ignore_bg else range(num_classes)
    for c in cls_range:
        p = (pred == c)
        g = (gt == c)
        denom = p.sum() + g.sum()
        if denom == 0:
            dices[c] = np.nan
        else:
            inter = np.logical_and(p, g).sum()
            dices[c] = (2.0 * inter) / (denom + 1e-8)
    return dices


def _surface_distances_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(bool)
    b = b.astype(bool)
    if a.sum() == 0 and b.sum() == 0:
        return np.array([0.0], dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return np.array([np.inf], dtype=np.float64)

    a_border = np.logical_xor(a, binary_erosion(a))
    b_border = np.logical_xor(b, binary_erosion(b))
    dt_b = distance_transform_edt(~b_border)
    dt_a = distance_transform_edt(~a_border)
    d_ab = dt_b[a_border]
    d_ba = dt_a[b_border]
    return np.concatenate([d_ab, d_ba]).astype(np.float64)


def hd95_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_bg: bool = True) -> np.ndarray:
    hds = np.full((num_classes,), np.nan, dtype=np.float64)
    cls_range = range(1, num_classes) if ignore_bg else range(num_classes)
    for c in cls_range:
        p = (pred == c)
        g = (gt == c)
        if p.sum() == 0 and g.sum() == 0:
            hds[c] = np.nan
            continue
        d = _surface_distances_2d(p, g)
        finite = d[np.isfinite(d)]
        hds[c] = np.percentile(finite, 95) if finite.size > 0 else np.inf
    return hds


# ----------------------------
# Benchmark
# ----------------------------
@torch.inference_mode()
def benchmark_one_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_bg: bool,
    warmup_iters: int,
    max_slices: Optional[int],
    use_amp: bool,
) -> Dict[str, Any]:

    # warmup
    it = iter(loader)
    for _ in range(warmup_iters):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(loader)
            x, _ = next(it)

        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            out = model(x)
            _ = out[0] if isinstance(out, tuple) else out
        if device.type == "cuda":
            torch.cuda.synchronize()

    dice_all: List[np.ndarray] = []
    hd_all: List[np.ndarray] = []
    lat_ms: List[float] = []
    peak_mib: List[float] = []
    n_seen = 0

    for x, y_onehot in loader:
        x = x.to(device, non_blocking=True)

        # y is one-hot [B,C,H,W]
        gt = torch.argmax(y_onehot, dim=1).cpu().numpy().astype(np.int32)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device.type == "cuda")):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        lat_ms.append((t1 - t0) * 1000.0)

        if device.type == "cuda":
            peak_mib.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int32)

        for i in range(pred.shape[0]):
            dice_all.append(dice_per_class(pred[i], gt[i], num_classes=num_classes, ignore_bg=ignore_bg))
            hd_all.append(hd95_per_class(pred[i], gt[i], num_classes=num_classes, ignore_bg=ignore_bg))
            n_seen += 1

        if max_slices is not None and n_seen >= max_slices:
            break

    dice_all = np.stack(dice_all, axis=0)
    hd_all = np.stack(hd_all, axis=0)
    cls_idx = list(range(1, num_classes)) if ignore_bg else list(range(num_classes))

    dice_macro_per_class = np.nanmean(dice_all, axis=0)
    hd_macro_per_class = np.nanmean(hd_all, axis=0)

    avg_dice = float(np.nanmean(dice_macro_per_class[cls_idx]))
    avg_hd95 = float(np.nanmean(hd_macro_per_class[cls_idx]))

    lat_ms = np.asarray(lat_ms, dtype=np.float64)
    return {
        "n_slices": int(n_seen),
        "avg_dice": avg_dice,
        "avg_hd95": avg_hd95,
        "latency_ms_mean": float(lat_ms.mean()) if lat_ms.size else float("nan"),
        "latency_ms_p50": float(np.percentile(lat_ms, 50)) if lat_ms.size else float("nan"),
        "latency_ms_p95": float(np.percentile(lat_ms, 95)) if lat_ms.size else float("nan"),
        "throughput_slices_per_s": float(1000.0 / lat_ms.mean()) if lat_ms.size else float("nan"),
        "vram_peak_mib_max": float(np.max(peak_mib)) if peak_mib else float("nan"),
        "vram_peak_mib_mean": float(np.mean(peak_mib)) if peak_mib else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="processed_dataset")
    parser.add_argument("--ignore_bg", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out_csv", type=str, default="evaluation/benchmark_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available.")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[Device] {gpu_name}")

    # Per-slice benchmark => batch_size=1 (same as your other scripts)
    _, _, test_loader_2d = get_dataloaders_2d(args.data_root, batch_size=1, num_workers=args.num_workers)
    _, _, test_loader_25d = get_dataloaders_25d(args.data_root, batch_size=1, num_workers=args.num_workers)

    # Auto-detect num_classes from labels (your logs show 9)
    _, y2d = next(iter(test_loader_2d))
    x25_sample, y25 = next(iter(test_loader_25d))
    num_classes_2d = int(y2d.shape[1])
    num_classes_25d = int(y25.shape[1])
    input_h, input_w = x25_sample.shape[2], x25_sample.shape[3]

    print(f"[Sanity] 2D  y channels = {num_classes_2d}")
    print(f"[Sanity] 25D y channels = {num_classes_25d}")
    print(f"[Info] Input spatial dims: {input_h}x{input_w}")

    # Compute average slices per volume for per-volume GFLOPs
    n_avg_slices = avg_slices_per_volume(args.data_root, split="Test")
    print(f"[Info] Average slices per test volume: {n_avg_slices:.1f}")

    USE_AMP = not args.no_amp

    # Benchmark models from BENCHMARK_MODELS_CROSS_VALIDATION.md
    MODELS: Dict[str, ModelSpec] = {
        "exp20": ModelSpec(
            model_class=SegMateFiLM,
            ckpt_path="archive/exp20/segmate_film_tf_efficientnetv2_m_25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
            name="SegMate FiLM - EfficientNetV2-M",
        ),
        "exp14": ModelSpec(
            model_class=SegMate,
            ckpt_path="archive/exp14/segmate_tf_efficientnetv2_m_25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
            name="SegMate - EfficientNetV2-M",
        ),
        "exp18": ModelSpec(
            model_class=VanillaSegMateV2,
            ckpt_path="archive/exp18/vanilla_segmate_tf_efficientnetv2_m_25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
            name="Vanilla - EfficientNetV2-M",
        ),
        "exp15": ModelSpec(
            model_class=SegMateMambaOut,
            ckpt_path="archive/exp15/segmate_mambaout_tiny_25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
            name="SegMate - MambaOut-Tiny",
        ),
        "exp17": ModelSpec(
            model_class=VanillaSegMateMambaOut,
            ckpt_path="archive/exp17/segmate_mambaout_tiny_25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
            name="Vanilla - MambaOut-Tiny",
        ),
        "exp12": ModelSpec(
            model_class=SegMateFastViT,
            ckpt_path="archive/exp12/segmate_fastvit25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
            name="SegMate - FastViT-T12",
        ),
        "exp16": ModelSpec(
            model_class=VanillaSegMateFastViT,
            ckpt_path="archive/exp16/vanilla_segmate_fastvit_t12_25D_best.pth",
            model_type="25D",
            model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
            name="Vanilla - FastViT-T12",
        ),
        # Standard U-Net baselines (2D, Dice-only)
        "exp_b1": ModelSpec(
            model_class=UNetBaseline,
            ckpt_path="checkpoints/unet_baseline_tf_efficientnetv2_m_2D_best.pth",
            model_type="2D",
            model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
            name="U-Net Baseline - EfficientNetV2-M",
        ),
        "exp_b2": ModelSpec(
            model_class=UNetBaseline,
            ckpt_path="checkpoints/unet_baseline_mambaout_tiny_2D_best.pth",
            model_type="2D",
            model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
            name="U-Net Baseline - MambaOut-Tiny",
        ),
        "exp_b3": ModelSpec(
            model_class=UNetBaseline,
            ckpt_path="checkpoints/unet_baseline_fastvit_t12_2D_best.pth",
            model_type="2D",
            model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
            name="U-Net Baseline - FastViT-T12",
        ),
    }

    rows = []
    for _, spec in MODELS.items():
        is_25d = spec.model_type.upper() == "25D"
        loader = test_loader_25d if is_25d else test_loader_2d
        num_classes = num_classes_25d if is_25d else num_classes_2d

        print(f"\n=== Benchmark: {spec.name} ({spec.model_type}) ===")
        print(f"ckpt: {spec.ckpt_path}")
        print(f"num_classes (auto): {num_classes}")

        model = build_and_load_model(
            spec=spec,
            device=device,
            num_classes=num_classes,
            use_25d_wrapper=is_25d,
        )

        # Compute GFLOPs per slice
        if is_25d:
            dummy_input = torch.randn(1, 3, input_h, input_w, device=device)
        else:
            dummy_input = torch.randn(1, 1, input_h, input_w, device=device)
        gflops_per_slice = compute_gflops(model, dummy_input)
        gflops_per_volume = gflops_per_slice * n_avg_slices
        print(f"  GFLOPs/slice: {gflops_per_slice:.2f}, GFLOPs/vol: {gflops_per_volume:.1f} (x{n_avg_slices:.0f} slices)")

        row = {
            "model": spec.name,
            "type": spec.model_type.upper(),
            "gpu": gpu_name,
            "params_M": count_params(model) / 1e6,
            "ckpt_MB": checkpoint_size_mb(spec.ckpt_path),
            "gflops_per_slice": gflops_per_slice,
            "gflops_per_volume": gflops_per_volume,
            "avg_slices_per_vol": n_avg_slices,
        }

        stats = benchmark_one_model(
            model=model,
            loader=loader,
            device=device,
            num_classes=num_classes,
            ignore_bg=args.ignore_bg,
            warmup_iters=args.warmup,
            max_slices=args.max_slices,
            use_amp=USE_AMP,
        )
        row.update(stats)
        rows.append(row)

        del model
        torch.cuda.empty_cache()

    import pandas as pd
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = pd.DataFrame(rows)
    cols = [
        "model", "type", "gpu",
        "avg_dice", "avg_hd95",
        "gflops_per_slice", "gflops_per_volume", "avg_slices_per_vol",
        "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
        "throughput_slices_per_s",
        "vram_peak_mib_max", "vram_peak_mib_mean",
        "params_M", "ckpt_MB",
        "n_slices",
    ]
    df = df[[c for c in cols if c in df.columns]]

    print("\n=== RESULTS ===")
    print(df.to_string(index=False))
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
