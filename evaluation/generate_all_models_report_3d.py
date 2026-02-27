# evaluation/generate_all_models_report_3d.py
"""
3D Volume-Level Evaluation — All Models, All Datasets.

Runs evaluate_model_3d() for every model/dataset combination, saves per-patient
raw CSVs and summary tables, and generates a Markdown cross-validation report.

Supports multi-GPU: with --num_gpus 3, runs 3 models in parallel across GPUs.

Usage:
    # Single GPU (default)
    python evaluation/generate_all_models_report_3d.py --datasets totalseg

    # 3 GPUs in parallel
    python evaluation/generate_all_models_report_3d.py --datasets totalseg --num_gpus 3
"""

import os
import sys
import random
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch
import torch.multiprocessing as mp

sys.path.append(".")

from evaluation.benchmark_models import ModelSpec
from evaluation.metrics_evaluation_3d import evaluate_model_3d, summarize_3d_results

# ── Model imports ──
from models.segmate_v2_film import SegMateFiLM
from models.segmate_v2 import SegMate as SegMateV2
from models.vanilla_segmate_v2 import SegMate as VanillaSegMateV2
from models.segmate_mambaout import SegMateMambaOut
from models.vanilla_segmate_mambaout import SegMateMambaOut as VanillaSegMateMambaOut
from models.segmate_fastvit import SegMateFastViT
from models.vanilla_segmate_fastvit import SegMateFastViT as VanillaSegMateFastViT
from models.segmate_fastvit_film import SegMateFastViTFiLM
from models.segmate_mambaout_film import SegMateMambaOutFiLM
from models.unet_baseline import UNetBaseline


# ═══════════════════════════════════════════════════════════════
# Model Registries
# ═══════════════════════════════════════════════════════════════

PRETRAINED_MODELS = {
    "exp20": ModelSpec(
        model_class=SegMateFiLM,
        ckpt_path="archive/exp20/segmate_film_tf_efficientnetv2_m_25D_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="SegMate FiLM - EfficientNetV2-M",
    ),
    "exp14": ModelSpec(
        model_class=SegMateV2,
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
    # New FiLM variants (checkpoints populated after training)
    "exp21": ModelSpec(
        model_class=SegMateFastViTFiLM,
        ckpt_path="archive/exp21/segmate_fastvit_film_fastvit_t12_25D_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="SegMate FiLM - FastViT-T12",
    ),
    "exp22": ModelSpec(
        model_class=SegMateMambaOutFiLM,
        ckpt_path="archive/exp22/segmate_mambaout_film_mambaout_tiny_25D_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="SegMate FiLM - MambaOut-Tiny",
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

SEGTHOR_FINETUNED_MODELS = {
    "ft20": ModelSpec(
        model_class=SegMateFiLM,
        ckpt_path="segthor_checkpoints/segmate_film_tf_efficientnetv2_m_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="FT SegMate FiLM - EfficientNetV2-M",
    ),
    "ft14": ModelSpec(
        model_class=SegMateV2,
        ckpt_path="segthor_checkpoints/segmatev2_tf_efficientnetv2_m_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="FT SegMate - EfficientNetV2-M",
    ),
    "ft18": ModelSpec(
        model_class=VanillaSegMateV2,
        ckpt_path="segthor_checkpoints/segmate_tf_efficientnetv2_m_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="FT Vanilla - EfficientNetV2-M",
    ),
    "ft15": ModelSpec(
        model_class=SegMateMambaOut,
        ckpt_path="segthor_checkpoints/segmate_mambaout_mambaout_tiny_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="FT SegMate - MambaOut-Tiny",
    ),
    "ft17": ModelSpec(
        model_class=VanillaSegMateMambaOut,
        ckpt_path="segthor_checkpoints/vanilla_mambaout_mambaout_tiny_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="FT Vanilla - MambaOut-Tiny",
    ),
    "ft12": ModelSpec(
        model_class=SegMateFastViT,
        ckpt_path="segthor_checkpoints/segmate_fastvit_fastvit_t12_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="FT SegMate - FastViT-T12",
    ),
    "ft16": ModelSpec(
        model_class=VanillaSegMateFastViT,
        ckpt_path="segthor_checkpoints/vanilla_fastvit_fastvit_t12_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="FT Vanilla - FastViT-T12",
    ),
    "ft21": ModelSpec(
        model_class=SegMateFastViTFiLM,
        ckpt_path="segthor_checkpoints/segmate_fastvit_film_fastvit_t12_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="FT SegMate FiLM - FastViT-T12",
    ),
    "ft22": ModelSpec(
        model_class=SegMateMambaOutFiLM,
        ckpt_path="segthor_checkpoints/segmate_mambaout_film_mambaout_tiny_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="FT SegMate FiLM - MambaOut-Tiny",
    ),
}

AMOS22_FINETUNED_MODELS = {
    "ft20_amos": ModelSpec(
        model_class=SegMateFiLM,
        ckpt_path="amos22_checkpoints/segmate_film_tf_efficientnetv2_m_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="FT SegMate FiLM - EfficientNetV2-M",
    ),
    "ft14_amos": ModelSpec(
        model_class=SegMateV2,
        ckpt_path="amos22_checkpoints/segmatev2_tf_efficientnetv2_m_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="FT SegMate - EfficientNetV2-M",
    ),
    "ft18_amos": ModelSpec(
        model_class=VanillaSegMateV2,
        ckpt_path="amos22_checkpoints/segmate_tf_efficientnetv2_m_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "tf_efficientnetv2_m"},
        name="FT Vanilla - EfficientNetV2-M",
    ),
    "ft15_amos": ModelSpec(
        model_class=SegMateMambaOut,
        ckpt_path="amos22_checkpoints/segmate_mambaout_mambaout_tiny_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="FT SegMate - MambaOut-Tiny",
    ),
    "ft17_amos": ModelSpec(
        model_class=VanillaSegMateMambaOut,
        ckpt_path="amos22_checkpoints/vanilla_mambaout_mambaout_tiny_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="FT Vanilla - MambaOut-Tiny",
    ),
    "ft12_amos": ModelSpec(
        model_class=SegMateFastViT,
        ckpt_path="amos22_checkpoints/segmate_fastvit_fastvit_t12_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="FT SegMate - FastViT-T12",
    ),
    "ft16_amos": ModelSpec(
        model_class=VanillaSegMateFastViT,
        ckpt_path="amos22_checkpoints/vanilla_fastvit_fastvit_t12_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="FT Vanilla - FastViT-T12",
    ),
    "ft21_amos": ModelSpec(
        model_class=SegMateFastViTFiLM,
        ckpt_path="amos22_checkpoints/segmate_fastvit_film_fastvit_t12_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "fastvit_t12"},
        name="FT SegMate FiLM - FastViT-T12",
    ),
    "ft22_amos": ModelSpec(
        model_class=SegMateMambaOutFiLM,
        ckpt_path="amos22_checkpoints/segmate_mambaout_film_mambaout_tiny_finetuned_best.pth",
        model_type="25D",
        model_kwargs={"in_channels": 1, "encoder_name": "mambaout_tiny"},
        name="FT SegMate FiLM - MambaOut-Tiny",
    ),
}


# ═══════════════════════════════════════════════════════════════
# Dataset configurations
# ═══════════════════════════════════════════════════════════════

DATASETS = {
    "totalseg": {
        "root": "processed_dataset",
        "dataset_key": "totalseg",
        "models": "pretrained",
    },
    "segthor": {
        "root": "processed_dataset_segthor",
        "dataset_key": "segthor",
        "models": "pretrained+finetuned",
    },
    "amos22": {
        "root": "processed_dataset_amos22",
        "dataset_key": "amos22",
        "models": "pretrained+finetuned",
    },
}


def _get_models_for_dataset(dataset_key: str) -> dict:
    """Return the appropriate model registry for a dataset."""
    cfg = DATASETS[dataset_key]
    models = dict(PRETRAINED_MODELS)
    if "finetuned" in cfg["models"]:
        if dataset_key == "segthor":
            models.update(SEGTHOR_FINETUNED_MODELS)
        elif dataset_key == "amos22":
            models.update(AMOS22_FINETUNED_MODELS)
    return models


# ═══════════════════════════════════════════════════════════════
# Worker function for multi-GPU
# ═══════════════════════════════════════════════════════════════

def _worker_evaluate(model_key, spec, cfg, split, batch_size, use_amp, raw_csv, gpu_id,
                     patient_subset=None):
    """
    Evaluate a single model on a specific GPU. Called from ProcessPoolExecutor.
    Returns (model_key, raw_csv) on success, (model_key, None) on failure.
    """
    try:
        raw_df = evaluate_model_3d(
            spec=spec,
            data_root=cfg["root"],
            split=split,
            dataset_key=cfg["dataset_key"],
            batch_size=batch_size,
            use_amp=use_amp,
            save_path=raw_csv,
            gpu_id=gpu_id,
            patient_subset=patient_subset,
        )
        raw_df.to_csv(raw_csv, index=False)
        print(f"  [GPU {gpu_id}] Saved: {raw_csv}")
        return model_key, raw_csv
    except Exception as e:
        print(f"\n[ERROR] [GPU {gpu_id}] {spec.name}: {e}")
        import traceback
        traceback.print_exc()
        return model_key, None


def _generate_markdown_report(results_dir: str, all_summaries: dict) -> str:
    """Generate a Markdown cross-validation report."""
    lines = [
        "# 3D Volume-Level Cross-Validation Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Metrics computed on full 3D patient volumes (not slice-averaged).",
        "Values reported as mean +/- std across patients.",
        "",
    ]

    for dataset_key, model_summaries in all_summaries.items():
        lines.append(f"## {dataset_key.upper()}")
        lines.append("")

        if not model_summaries:
            lines.append("No results available.")
            lines.append("")
            continue

        # Build comparison table: one row per model, columns = organs
        # Show Dice mean +/- std for each organ + TOTAL
        first_summary = list(model_summaries.values())[0]
        organs = first_summary["organ"].tolist()

        lines.append("### Dice (mean +/- std)")
        lines.append("")

        header = "| Model | " + " | ".join(organs) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(organs)) + "|"
        lines.append(header)
        lines.append(sep)

        for model_key, summary_df in model_summaries.items():
            model_name = PRETRAINED_MODELS.get(model_key) or SEGTHOR_FINETUNED_MODELS.get(model_key) or AMOS22_FINETUNED_MODELS.get(model_key)
            display_name = model_name.name if model_name else model_key
            cells = []
            for _, row in summary_df.iterrows():
                mean = row.get("dice_mean", float("nan"))
                std = row.get("dice_std", float("nan"))
                if pd.notna(mean):
                    cells.append(f"{mean:.4f}+/-{std:.4f}")
                else:
                    cells.append("n/a")
            lines.append(f"| {display_name} | " + " | ".join(cells) + " |")

        lines.append("")

        # HD95 table
        lines.append("### HD95 (mean +/- std)")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        for model_key, summary_df in model_summaries.items():
            model_name = PRETRAINED_MODELS.get(model_key) or SEGTHOR_FINETUNED_MODELS.get(model_key) or AMOS22_FINETUNED_MODELS.get(model_key)
            display_name = model_name.name if model_name else model_key
            cells = []
            for _, row in summary_df.iterrows():
                mean = row.get("hd95_mean", float("nan"))
                std = row.get("hd95_std", float("nan"))
                if pd.notna(mean):
                    cells.append(f"{mean:.2f}+/-{std:.2f}")
                else:
                    cells.append("n/a")
            lines.append(f"| {display_name} | " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="3D Volume-Level Evaluation — All Models")
    parser.add_argument(
        "--datasets", type=str, default="totalseg,segthor,amos22",
        help="Comma-separated dataset keys (default: totalseg,segthor,amos22)",
    )
    parser.add_argument("--split", type=str, default="test", help="Data split (default: test)")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size for inference")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use in parallel (default: 1). "
             "Models are distributed round-robin across GPUs.",
    )
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Randomly subsample N patients per dataset (default: all). "
             "Same subset used for every model. Seed controlled by --seed.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for patient subsampling (default: 42).",
    )
    args = parser.parse_args()

    dataset_keys = [d.strip() for d in args.datasets.split(",")]
    results_base = "evaluation/3d_results"
    use_amp = not args.no_amp
    num_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    if num_gpus > 1:
        print(f"\n[Multi-GPU] Using {num_gpus} GPUs: "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(num_gpus)))
    else:
        print(f"\n[Single GPU] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    all_summaries = {}

    for dkey in dataset_keys:
        if dkey not in DATASETS:
            print(f"[WARN] Unknown dataset key: {dkey}, skipping.")
            continue

        cfg = DATASETS[dkey]
        models = _get_models_for_dataset(dkey)
        raw_dir = os.path.join(results_base, dkey, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Patient subsampling: sample once, use for all models
        patient_subset = None
        if args.max_patients is not None:
            from evaluation.volume_dataloader_3d import VolumeIterator
            vol_iter = VolumeIterator(cfg["root"], args.split)
            all_patients = vol_iter.patient_ids
            if args.max_patients < len(all_patients):
                rng = random.Random(args.seed)
                sampled = sorted(rng.sample(all_patients, args.max_patients))
                patient_subset = set(sampled)
                # Save the sampled list for reproducibility
                subset_path = os.path.join(results_base, dkey, "patient_subset.txt")
                with open(subset_path, "w") as sf:
                    sf.write(f"# seed={args.seed}, max_patients={args.max_patients}\n")
                    for pid in sampled:
                        sf.write(pid + "\n")
                print(f"[Subset] Sampled {len(patient_subset)}/{len(all_patients)} patients "
                      f"(seed={args.seed}), saved to {subset_path}")
            else:
                print(f"[Subset] max_patients={args.max_patients} >= total={len(all_patients)}, using all")
            del vol_iter

        # Filter to models with existing checkpoints
        valid_models = {}
        for model_key, spec in models.items():
            if not os.path.isfile(spec.ckpt_path):
                print(f"[SKIP] {spec.name}: checkpoint not found ({spec.ckpt_path})")
            else:
                valid_models[model_key] = spec

        print(f"\n{'='*70}")
        print(f"Dataset: {dkey.upper()} ({cfg['root']})")
        print(f"Models: {len(valid_models)} (with checkpoints)")
        print(f"GPUs: {num_gpus}")
        if patient_subset:
            print(f"Patient subset: {len(patient_subset)} patients")
        print(f"{'='*70}")

        model_summaries = {}
        summary_rows = []

        if num_gpus > 1:
            # ── Multi-GPU: run models in parallel ──
            # Use 'spawn' to avoid CUDA fork issues
            ctx = mp.get_context("spawn")
            futures = {}

            with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
                for i, (model_key, spec) in enumerate(valid_models.items()):
                    gpu_id = i % num_gpus
                    raw_csv = os.path.join(raw_dir, f"{model_key}.csv")
                    future = executor.submit(
                        _worker_evaluate,
                        model_key, spec, cfg, args.split,
                        args.batch_size, use_amp, raw_csv, gpu_id,
                        patient_subset,
                    )
                    futures[future] = model_key

                for future in as_completed(futures):
                    model_key, result_csv = future.result()
                    if result_csv and os.path.isfile(result_csv):
                        raw_df = pd.read_csv(result_csv)
                        summary_df = summarize_3d_results(raw_df)
                        model_summaries[model_key] = summary_df

                        total_row = summary_df[summary_df["organ"] == "TOTAL"]
                        if not total_row.empty:
                            row = total_row.iloc[0].to_dict()
                            spec = valid_models[model_key]
                            row["model"] = spec.name
                            row["model_key"] = model_key
                            summary_rows.append(row)
        else:
            # ── Single GPU: sequential ──
            for model_key, spec in valid_models.items():
                raw_csv = os.path.join(raw_dir, f"{model_key}.csv")

                try:
                    raw_df = evaluate_model_3d(
                        spec=spec,
                        data_root=cfg["root"],
                        split=args.split,
                        dataset_key=cfg["dataset_key"],
                        batch_size=args.batch_size,
                        use_amp=use_amp,
                        save_path=raw_csv,
                        patient_subset=patient_subset,
                    )

                    raw_df.to_csv(raw_csv, index=False)
                    print(f"  Saved raw: {raw_csv}")

                    summary_df = summarize_3d_results(raw_df)
                    model_summaries[model_key] = summary_df

                    total_row = summary_df[summary_df["organ"] == "TOTAL"]
                    if not total_row.empty:
                        row = total_row.iloc[0].to_dict()
                        row["model"] = spec.name
                        row["model_key"] = model_key
                        summary_rows.append(row)

                except Exception as e:
                    print(f"\n[ERROR] {spec.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save summary CSV for this dataset
        if summary_rows:
            summary_csv = os.path.join(results_base, dkey, "summary.csv")
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
            print(f"\nSaved summary: {summary_csv}")

        all_summaries[dkey] = model_summaries

    # Generate Markdown report
    md_report = _generate_markdown_report(results_base, all_summaries)
    md_path = os.path.join(results_base, "CROSS_VALIDATION_RESULTS_3D.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"\nMarkdown report: {md_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
