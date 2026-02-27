import os
import sys
import torch
import numpy as np
import pydicom
from glob import glob
from datetime import datetime
from skimage.transform import resize
import matplotlib.pyplot as plt

from models.custom_unet import CustomUNet
from rtstruct.rtstruct_generator import export_rtstruct


def load_model(model_path, device):
    model = CustomUNet(num_classes=8).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def load_ct_series(dicom_folder):
    slices = [pydicom.dcmread(f) for f in glob(os.path.join(dicom_folder, "*.dcm"))]
    slices = [s for s in slices if hasattr(s, 'ImagePositionPatient') and s.Modality == "CT"]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices])
    return volume, slices


def preprocess_volume(volume):
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400
    volume_resized = np.stack([resize(s, (256, 256), preserve_range=True) for s in volume])
    return volume_resized


def run_inference(model, volume, device):
    preds = []
    with torch.no_grad():
        for i, slice in enumerate(volume):
            input_tensor = torch.tensor(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            print(f"\n➡️ Slice {i} input shape: {input_tensor.shape}")

            output, _ = model(input_tensor)
            print(f"⬅️ Raw model output shape: {output.shape}")

            prob = torch.sigmoid(output)
            print(f"✅ After sigmoid: min={prob.min().item()}, max={prob.max().item()}, shape={prob.shape}")

            preds.append(prob.squeeze(0).cpu().numpy())
    return np.stack(preds)



def resize_prediction(preds, original_shape):
    N = preds.shape[0]
    num_classes = preds.shape[1]
    H, W = original_shape
    masks = np.zeros((N, num_classes, H, W), dtype=np.uint8)
    for i in range(N):
        for c in range(num_classes):
            mask = resize(preds[i, c], (H, W), order=1, preserve_range=True) > 0.5
            masks[i, c] = mask.astype(np.uint8)
    return masks


def save_slice_visualizations(volume, masks, out_dir, slice_indices=[20, 30, 40]):
    os.makedirs(out_dir, exist_ok=True)
    organ_names = ["Heart", "Left Lung", "Right Lung", "Cord", "Esophagus", "Liver", "Left Kidney", "Right Kidney"]

    for idx in slice_indices:
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        axs = axs.flatten()

        axs[0].imshow(volume[idx], cmap="gray")
        axs[0].set_title(f"Original Slice {idx}")
        axs[0].axis("off")

        for i in range(8):
            mask = masks[idx, i]
            axs[i+1].imshow(volume[idx], cmap="gray")
            axs[i+1].imshow(mask, alpha=0.5, cmap="jet")
            axs[i+1].set_title(f"{organ_names[i]}")
            axs[i+1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"slice_{idx}.png"))
        plt.close()


if __name__ == "__main__":
    dicom_folder = "dataset/Patient_2405"
    model_path = "checkpoints/effb3_best.pth"
    out_file = f"output/rtstruct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dcm"

    os.makedirs("output", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("Loading model...")
    model = load_model(model_path, device)

    print("Loading CT series...")
    volume, slices = load_ct_series(dicom_folder)

    print("Preprocessing...")
    preprocessed = preprocess_volume(volume)

    print("Running inference...")
    preds = run_inference(model, preprocessed, device)
    print(f"Model output stats: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")

    print("Resizing predictions...")
    masks = resize_prediction(preds, volume.shape[1:])

    print("Saving slice visualizations...")
    save_slice_visualizations(volume, masks, "output/slice_viz", slice_indices=[20, 30, 40])

    print("Fixing mask shape for RTSTRUCT...")
    masks = np.moveaxis(masks, 1, -1)  # [Z, C, H, W] → [Z, H, W, C]

    print("Exporting RTSTRUCT...")
    export_rtstruct(masks, slices, out_file)
    print(f"✅ RTSTRUCT saved to: {out_file}")
