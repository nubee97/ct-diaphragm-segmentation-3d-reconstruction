"""
main_fixed_dicom_png.py

Reliable DICOM + PNG segmentation pipeline for a 2D U-Net model.

What this script supports:
1. Train U-Net with DICOM images and PNG masks.
2. Validate with a real train/validation split.
3. Save best_model.pth and last_model.pth checkpoints.
4. Predict masks from a DICOM folder using a saved checkpoint.
5. Save predicted PNG masks and optional overlays.
6. Optionally stack predicted masks into a 3D volume and export a .ply mesh + HTML preview.

Expected project files in the same folder:
- model.py containing UNet

Example training:
python main_fixed_dicom_png.py train \
  --dicom-root "C:/Users/jiseo/Downloads/DICOM" \
  --mask-root "C:/Users/jiseo/Downloads/PNG_LABELS" \
  --save-dir "C:/Users/jiseo/Downloads/diaphragm_runs" \
  --epochs 20 \
  --batch-size 8 \
  --image-size 512 512

Example prediction:
python main_fixed_dicom_png.py predict \
  --dicom-root "C:/Users/jiseo/Downloads/DICOM_TEST" \
  --checkpoint "C:/Users/jiseo/Downloads/diaphragm_runs/best_model.pth" \
  --output-dir "C:/Users/jiseo/Downloads/diaphragm_predictions" \
  --image-size 512 512 \
  --save-overlays \
  --export-mesh
"""

from __future__ import annotations

import argparse
import json
import gc
import os
import random
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import measure, morphology
from torch.utils.data import DataLoader, Dataset, random_split, Subset

try:
    import open3d as o3d
except Exception:  # open3d is only required for mesh export
    o3d = None

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:  # plotly is only required for HTML mesh preview
    go = None
    pio = None

try:
    from scipy import ndimage as scipy_ndimage
except Exception:  # scipy is optional; HD95 will be skipped if unavailable
    scipy_ndimage = None

from model import UNet


# -----------------------------------------------------------------------------
# Reproducibility and device helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS for readable ETA printing."""
    if seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


# -----------------------------------------------------------------------------
# DICOM / PNG matching helpers
# -----------------------------------------------------------------------------


PHASE_NAMES = {
    "in",
    "ex",
    "insp",
    "exp",
    "inspiration",
    "expiration",
    "inhale",
    "exhale",
}

IMAGE_EXTENSIONS = {".dcm", ".dicom"}
MASK_EXTENSIONS = {".png"}


@dataclass(frozen=True)
class SampleRecord:
    dicom_path: Path
    mask_path: Optional[Path]
    patient_id: str
    phase: str
    slice_location: float
    instance_number: int


def natural_sort_key(path: Path) -> List[object]:
    text = str(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def numeric_tokens(text: str) -> List[str]:
    return re.findall(r"\d+", text)


def normalize_number_token(token: str) -> str:
    stripped = token.lstrip("0")
    return stripped if stripped else "0"


def make_match_keys(path: Path, instance_number: Optional[int] = None) -> set[str]:
    """
    Generate flexible matching keys so DICOM files like IM-0002-0497.dcm can match
    masks like 497_axial.png, 0497.png, IM-0002-0497.png, etc.
    """
    stem = path.stem.lower()
    cleaned = stem.replace("_axial", "").replace("-axial", "")

    keys = {stem, cleaned}

    for token in numeric_tokens(stem):
        keys.add(token)
        keys.add(normalize_number_token(token))

    if instance_number is not None and instance_number >= 0:
        inst = str(int(instance_number))
        keys.add(inst)
        keys.add(inst.zfill(3))
        keys.add(inst.zfill(4))

    return {k for k in keys if k}


def infer_patient_and_phase(path: Path, root: Path) -> Tuple[str, str]:
    """
    Supports common folder shapes:
      root / patient_id / in / file.dcm
      root / patient_id / ex / file.dcm
      root / patient_id / file.dcm
      root / file.dcm
    """
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts

    phase = "default"
    patient_id = path.parent.name

    for index, part in enumerate(rel_parts[:-1]):
        part_lower = part.lower()
        if part_lower in PHASE_NAMES:
            phase = part_lower
            if index > 0:
                patient_id = rel_parts[index - 1]
            break
    else:
        if len(rel_parts) >= 2:
            patient_id = rel_parts[0]
        else:
            patient_id = "unknown_patient"

    return str(patient_id), str(phase)


def safe_read_dicom_header(path: Path) -> Tuple[float, int]:
    """Read only DICOM metadata needed for sorting/matching."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        instance = int(getattr(ds, "InstanceNumber", -1))
        slice_location = float(getattr(ds, "SliceLocation", instance if instance >= 0 else 0.0))
        return slice_location, instance
    except Exception:
        # Fallback to the last number in the filename
        tokens = numeric_tokens(path.stem)
        fallback = int(normalize_number_token(tokens[-1])) if tokens else -1
        return float(fallback if fallback >= 0 else 0), fallback


def collect_files(root: Path, extensions: set[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files, key=natural_sort_key)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class DicomPngSegmentationDataset(Dataset):
    """
    Dataset for DICOM images and PNG masks.

    If mask_root is provided, the dataset returns image-mask pairs for training.
    If mask_root is None, the dataset returns only DICOM images for prediction.
    """

    def __init__(
        self,
        dicom_root: str | Path,
        mask_root: Optional[str | Path] = None,
        image_size: Optional[Tuple[int, int]] = None,
        require_masks: bool = True,
        verbose: bool = True,
        mask_clean_mode: str = "auto",
        mask_threshold: float = 0.85,
        mask_min_area: int = 10,
        mask_max_area_ratio: float = 0.25,
        mask_border_ratio: float = 0.08,
        mask_dilate_radius: int = 1,
    ) -> None:
        self.dicom_root = Path(dicom_root)
        self.mask_root = Path(mask_root) if mask_root else None
        self.image_size = image_size  # (height, width)
        self.require_masks = require_masks
        self.verbose = verbose

        # PNG label cleaning settings.
        # These are important when the PNG folder contains visualization PNGs
        # rather than clean 0/255 binary masks.
        self.mask_clean_mode = mask_clean_mode
        self.mask_threshold = mask_threshold
        self.mask_min_area = mask_min_area
        self.mask_max_area_ratio = mask_max_area_ratio
        self.mask_border_ratio = mask_border_ratio
        self.mask_dilate_radius = mask_dilate_radius

        dicom_files = collect_files(self.dicom_root, IMAGE_EXTENSIONS)
        if not dicom_files:
            raise RuntimeError(f"No DICOM files found under: {self.dicom_root}")

        label_index: Dict[Tuple[Optional[str], Optional[str], str], Path] = {}
        if self.mask_root is not None:
            mask_files = collect_files(self.mask_root, MASK_EXTENSIONS)
            if not mask_files:
                raise RuntimeError(f"No PNG mask files found under: {self.mask_root}")
            label_index = self._build_label_index(mask_files)

        self.samples: List[SampleRecord] = []
        skipped: List[Path] = []

        for dicom_path in dicom_files:
            patient_id, phase = infer_patient_and_phase(dicom_path, self.dicom_root)
            slice_location, instance_number = safe_read_dicom_header(dicom_path)

            mask_path = None
            if self.mask_root is not None:
                match_keys = make_match_keys(dicom_path, instance_number)
                mask_path = self._find_matching_mask(label_index, patient_id, phase, match_keys)
                if mask_path is None and self.require_masks:
                    skipped.append(dicom_path)
                    continue

            self.samples.append(
                SampleRecord(
                    dicom_path=dicom_path,
                    mask_path=mask_path,
                    patient_id=patient_id,
                    phase=phase,
                    slice_location=slice_location,
                    instance_number=instance_number,
                )
            )

        if self.require_masks and not self.samples:
            example = ""
            if skipped:
                example = f" Example skipped DICOM: {skipped[0]}"
            raise RuntimeError(
                "No valid DICOM/PNG pairs were created. Check that your DICOM and PNG "
                f"filenames share a slice number or matching stem.{example}"
            )

        if self.verbose:
            print(f"Loaded {len(self.samples)} sample(s) from {self.dicom_root}")
            if skipped:
                print(f"Skipped {len(skipped)} DICOM file(s) with no matching PNG mask.")
                for path in skipped[:10]:
                    print(f"  [Skipped] {path}")
                if len(skipped) > 10:
                    print("  ...")

    def _build_label_index(self, mask_files: Sequence[Path]) -> Dict[Tuple[Optional[str], Optional[str], str], Path]:
        label_index: Dict[Tuple[Optional[str], Optional[str], str], Path] = {}
        assert self.mask_root is not None

        for mask_path in mask_files:
            patient_id, phase = infer_patient_and_phase(mask_path, self.mask_root)
            keys = make_match_keys(mask_path)

            for key in keys:
                # Most specific: same patient and same phase
                label_index[(patient_id, phase, key)] = mask_path
                # Useful fallback: same patient but phase not mirrored in mask folder
                label_index[(patient_id, None, key)] = mask_path
                # Last fallback: global match by slice number/stem
                label_index[(None, None, key)] = mask_path

        return label_index

    @staticmethod
    def _find_matching_mask(
        label_index: Dict[Tuple[Optional[str], Optional[str], str], Path],
        patient_id: str,
        phase: str,
        match_keys: Iterable[str],
    ) -> Optional[Path]:
        # Priority order avoids accidentally matching the wrong patient when many patients
        # have the same slice numbers.
        for key in match_keys:
            found = label_index.get((patient_id, phase, key))
            if found is not None:
                return found

        for key in match_keys:
            found = label_index.get((patient_id, None, key))
            if found is not None:
                return found

        for key in match_keys:
            found = label_index.get((None, None, key))
            if found is not None:
                return found

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = read_dicom_image(sample.dicom_path)
        image = normalize_image(image)
        image = resize_2d(image, self.image_size, is_mask=False)

        item: Dict[str, object] = {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "patient_id": sample.patient_id,
            "phase": sample.phase,
            "slice_location": torch.tensor(sample.slice_location, dtype=torch.float32),
            "instance_number": torch.tensor(sample.instance_number, dtype=torch.int64),
            "dicom_path": str(sample.dicom_path),
        }

        if sample.mask_path is not None:
            mask = read_png_mask(
                sample.mask_path,
                mode=self.mask_clean_mode,
                threshold=self.mask_threshold,
                min_area=self.mask_min_area,
                max_area_ratio=self.mask_max_area_ratio,
                border_ratio=self.mask_border_ratio,
                dilate_radius=self.mask_dilate_radius,
            )
            mask = resize_2d(mask, self.image_size, is_mask=True)
            mask = (mask > 0.5).astype(np.float32)
            item["mask"] = torch.from_numpy(mask).unsqueeze(0).float()
            item["mask_path"] = str(sample.mask_path)

        return item


# -----------------------------------------------------------------------------
# Image loading / preprocessing
# -----------------------------------------------------------------------------


def read_dicom_image(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    return image


def is_binary_like(gray_uint8: np.ndarray) -> bool:
    """
    Returns True if a PNG already looks like a binary mask.

    A clean mask should usually contain only values like 0/255 or 0/1.
    Anti-aliased masks may contain a few extra values, so we allow a small
    number of unique values or almost all pixels near 0 or near 255.
    """
    unique_values = np.unique(gray_uint8)
    if len(unique_values) <= 5:
        return True

    near_binary_ratio = np.mean((gray_uint8 <= 5) | (gray_uint8 >= 250))
    return bool(near_binary_ratio >= 0.995)


def extract_colored_overlay_mask(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Try to extract colored plotted annotation pixels from a visualization PNG.

    This helps when labels were accidentally saved as matplotlib visualization
    images with colored intersection points/lines over the CT image.
    """
    rgb = rgb_uint8.astype(np.int16)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    red = (r > 120) & (r > g + 35) & (r > b + 35)
    green = (g > 120) & (g > r + 35) & (g > b + 35)
    blue = (b > 120) & (b > r + 35) & (b > g + 35)

    yellow = (r > 150) & (g > 150) & (b < 120)
    magenta = (r > 150) & (b > 150) & (g < 120)
    cyan = (g > 150) & (b > 150) & (r < 120)

    return red | green | blue | yellow | magenta | cyan


def remove_plot_border_artifacts(mask: np.ndarray, border_ratio: float) -> np.ndarray:
    """
    Remove likely matplotlib title/axis/border artifacts near image edges.

    Set --mask-border-ratio 0 if your true label touches the image border.
    """
    if border_ratio <= 0:
        return mask

    cleaned = mask.copy()
    h, w = cleaned.shape
    border_y = int(round(h * border_ratio))
    border_x = int(round(w * border_ratio))

    if border_y > 0:
        cleaned[:border_y, :] = False
        cleaned[h - border_y :, :] = False
    if border_x > 0:
        cleaned[:, :border_x] = False
        cleaned[:, w - border_x :] = False

    return cleaned


def cleanup_binary_mask(
    mask: np.ndarray,
    min_area: int = 10,
    max_area_ratio: float = 0.25,
    border_ratio: float = 0.08,
    dilate_radius: int = 1,
) -> np.ndarray:
    """
    Clean a binary mask:
    - remove plot border/title/axis artifacts
    - remove very small connected components
    - remove extremely large components that are unlikely to be a diaphragm label
    - optionally dilate thin plotted points/lines slightly
    """
    mask_bool = mask.astype(bool)

    mask_bool = remove_plot_border_artifacts(mask_bool, border_ratio=border_ratio)

    if mask_bool.sum() == 0:
        return mask_bool.astype(np.float32)

    labeled = measure.label(mask_bool, connectivity=2)
    h, w = mask_bool.shape
    max_area = h * w * max_area_ratio if max_area_ratio > 0 else h * w + 1

    cleaned = np.zeros_like(mask_bool, dtype=bool)

    for region in measure.regionprops(labeled):
        area = int(region.area)
        if area < min_area:
            continue
        if area > max_area:
            continue
        cleaned[labeled == region.label] = True

    # If the component filtering removed everything, fall back to the border-cleaned mask.
    # This avoids returning an empty label accidentally.
    if cleaned.sum() == 0 and mask_bool.sum() > 0:
        cleaned = mask_bool

    if dilate_radius > 0 and cleaned.sum() > 0:
        cleaned = morphology.dilation(cleaned, morphology.disk(dilate_radius))

    return cleaned.astype(np.float32)


def read_png_mask(
    path: Path,
    mode: str = "auto",
    threshold: float = 0.85,
    min_area: int = 10,
    max_area_ratio: float = 0.25,
    border_ratio: float = 0.08,
    dilate_radius: int = 1,
) -> np.ndarray:
    """
    Read and clean a PNG segmentation label.

    Supported modes:
        auto:
            Detects clean binary masks automatically. If the PNG is a
            visualization image, it first tries to extract colored annotation
            pixels, then falls back to bright-pixel extraction.

        binary:
            Use this when the PNG files are already clean 0/255 masks.

        color:
            Extract only colored overlay pixels from visualization PNGs.

        bright:
            Extract very bright pixels. Useful for white intersection lines/points,
            but may also capture white text/bones if the PNG is a full screenshot.

        none:
            Only normalize the PNG to 0..1. Not recommended for your current labels.

    Important:
        This function can improve bad visualization-style PNGs, but it cannot
        perfectly recover missing labels if the original PNG does not contain
        true segmentation information.
    """
    mode = mode.lower().strip()
    if mode not in {"auto", "binary", "color", "bright", "none"}:
        raise ValueError(f"Unsupported mask clean mode: {mode}")

    pil_rgb = Image.open(path).convert("RGB")
    rgb_uint8 = np.asarray(pil_rgb).astype(np.uint8)
    gray_uint8 = np.asarray(pil_rgb.convert("L")).astype(np.uint8)
    gray_float = gray_uint8.astype(np.float32) / 255.0

    if mode == "none":
        return gray_float.astype(np.float32)

    if mode == "binary":
        mask = gray_float > 0.5
        return cleanup_binary_mask(
            mask,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            border_ratio=0.0,
            dilate_radius=0,
        )

    if mode == "color":
        mask = extract_colored_overlay_mask(rgb_uint8)
        return cleanup_binary_mask(
            mask,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            border_ratio=border_ratio,
            dilate_radius=dilate_radius,
        )

    if mode == "bright":
        mask = gray_float >= threshold
        return cleanup_binary_mask(
            mask,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            border_ratio=border_ratio,
            dilate_radius=dilate_radius,
        )

    # AUTO MODE
    if is_binary_like(gray_uint8):
        mask = gray_float > 0.5
        return cleanup_binary_mask(
            mask,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            border_ratio=0.0,
            dilate_radius=0,
        )

    color_mask = extract_colored_overlay_mask(rgb_uint8)
    if int(color_mask.sum()) >= max(5, min_area):
        return cleanup_binary_mask(
            color_mask,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            border_ratio=border_ratio,
            dilate_radius=dilate_radius,
        )

    bright_mask = gray_float >= threshold
    return cleanup_binary_mask(
        bright_mask,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        border_ratio=border_ratio,
        dilate_radius=dilate_radius,
    )


def save_clean_mask_debug_image(
    original_mask_path: Path,
    cleaned_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a before/after image so you can visually check mask cleaning quality."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = np.asarray(Image.open(original_mask_path).convert("L"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original PNG")
    axes[0].axis("off")

    axes[1].imshow(cleaned_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Cleaned Binary Mask")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_clean_mask_png(mask: np.ndarray, output_path: Path) -> None:
    """Write a binary mask as a clean 0/255 PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(output_path)


def save_training_mask_debug_previews(dataset: "DicomPngSegmentationDataset", save_dir: Path, count: int = 12) -> None:
    """
    Save a few cleaned mask previews before training starts.
    This lets you verify that the labels are binary and not full screenshots.
    """
    if count <= 0:
        return

    preview_dir = save_dir / "clean_mask_debug"
    saved = 0

    for sample in dataset.samples:
        if sample.mask_path is None:
            continue

        cleaned = read_png_mask(
            sample.mask_path,
            mode=dataset.mask_clean_mode,
            threshold=dataset.mask_threshold,
            min_area=dataset.mask_min_area,
            max_area_ratio=dataset.mask_max_area_ratio,
            border_ratio=dataset.mask_border_ratio,
            dilate_radius=dataset.mask_dilate_radius,
        )
        safe_name = f"{sample.patient_id}_{sample.phase}_{sample.mask_path.stem}_debug.png"
        save_clean_mask_debug_image(sample.mask_path, cleaned, preview_dir / safe_name)

        saved += 1
        if saved >= count:
            break

    if saved:
        print(f"Saved {saved} cleaned mask debug preview(s) to: {preview_dir}")



def normalize_image(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val - min_val < eps:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val + eps)


def resize_2d(array: np.ndarray, image_size: Optional[Tuple[int, int]], is_mask: bool) -> np.ndarray:
    if image_size is None:
        return array.astype(np.float32)

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
    if is_mask:
        resized = F.interpolate(tensor, size=image_size, mode="nearest")
    else:
        resized = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# Training / validation
# -----------------------------------------------------------------------------


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=dims)
    denominator = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Dice loss.

    This is important for tiny/thin masks because pure BCE can become dominated
    by background pixels and the model may learn to predict all black.
    """
    probs = torch.sigmoid(logits)
    probs = probs.float()
    targets = targets.float()

    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + weighted BCE loss.

    Cleaned diaphragm masks can be very sparse/thin.
    With only BCEWithLogitsLoss, the model can get low loss by predicting
    mostly background. Dice loss directly rewards overlap with the mask.
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.7,
        pos_weight: Optional[float] = 20.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

        if pos_weight is not None and float(pos_weight) > 0:
            self.register_buffer("pos_weight_tensor", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight_tensor = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight_tensor is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight_tensor.to(device=logits.device, dtype=logits.dtype),
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        dice = soft_dice_loss_from_logits(logits, targets, eps=self.smooth)
        return (self.bce_weight * bce) + (self.dice_weight * dice)


def estimate_pos_weight_from_dataset(
    dataset: torch.utils.data.Dataset,
    max_samples: int = 256,
    max_pos_weight: float = 100.0,
) -> float:
    """
    Estimate positive-class BCE weight from masks.

    Formula:
        pos_weight = background_pixels / foreground_pixels

    The result is capped because very tiny masks can produce huge weights.
    """
    if max_samples <= 0:
        return 20.0

    base_dataset = dataset
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        indices = list(dataset.indices)
        base_dataset = dataset.dataset
    else:
        indices = list(range(len(dataset)))

    if not indices:
        return 20.0

    if len(indices) > max_samples:
        step = max(1, len(indices) // max_samples)
        indices = indices[::step][:max_samples]

    total_fg = 0.0
    total_pixels = 0.0

    for idx in indices:
        try:
            if hasattr(base_dataset, "samples"):
                sample = base_dataset.samples[int(idx)]
                if sample.mask_path is None:
                    continue
                mask = read_png_mask(
                    sample.mask_path,
                    mode=getattr(base_dataset, "mask_clean_mode", "binary"),
                    threshold=getattr(base_dataset, "mask_threshold", 0.85),
                    min_area=getattr(base_dataset, "mask_min_area", 10),
                    max_area_ratio=getattr(base_dataset, "mask_max_area_ratio", 0.25),
                    border_ratio=getattr(base_dataset, "mask_border_ratio", 0.08),
                    dilate_radius=getattr(base_dataset, "mask_dilate_radius", 1),
                )
            else:
                item = dataset[int(idx)]
                mask = item["mask"].squeeze().detach().cpu().numpy()

            mask = (mask > 0.5).astype(np.float32)
            total_fg += float(mask.sum())
            total_pixels += float(mask.size)
        except Exception as exc:
            print(f"[Warning] Could not inspect mask for pos_weight estimate at index {idx}: {exc}")

    if total_pixels <= 0 or total_fg <= 0:
        print("[Warning] Could not estimate pos_weight from masks. Using fallback pos_weight=20.0")
        return 20.0

    total_bg = total_pixels - total_fg
    estimated = total_bg / max(total_fg, 1.0)
    estimated = max(1.0, min(float(estimated), float(max_pos_weight)))
    return estimated


def build_criterion(
    args: argparse.Namespace,
    device: torch.device,
    train_dataset: Optional[torch.utils.data.Dataset] = None,
) -> nn.Module:
    """Build the training loss."""
    loss_name = getattr(args, "loss", "dice_bce")

    if loss_name == "bce":
        print("Loss: BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss().to(device)

    pos_weight = float(getattr(args, "pos_weight", 20.0))
    if getattr(args, "auto_pos_weight", False) and train_dataset is not None:
        pos_weight = estimate_pos_weight_from_dataset(
            train_dataset,
            max_samples=int(getattr(args, "pos_weight_sample_count", 256)),
            max_pos_weight=float(getattr(args, "max_pos_weight", 100.0)),
        )
        print(f"Auto-estimated pos_weight: {pos_weight:.4f}")

    criterion = DiceBCELoss(
        bce_weight=float(getattr(args, "bce_weight", 0.3)),
        dice_weight=float(getattr(args, "dice_weight", 0.7)),
        pos_weight=pos_weight,
    ).to(device)

    print(
        "Loss: DiceBCELoss "
        f"(bce_weight={criterion.bce_weight}, "
        f"dice_weight={criterion.dice_weight}, "
        f"pos_weight={pos_weight})"
    )
    return criterion


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_every: int = 25,
) -> float:
    """Train one epoch and print batch-level ETA/progress."""
    model.train()
    total_loss = 0.0
    epoch_start = time.time()
    total_batches = max(len(dataloader), 1)

    for batch_idx, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

        should_log = log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches)
        if should_log:
            elapsed = time.time() - epoch_start
            avg_batch_time = elapsed / batch_idx
            remaining_batches = total_batches - batch_idx
            epoch_eta = remaining_batches * avg_batch_time
            running_loss = total_loss / batch_idx
            print(
                f"  Epoch {epoch:03d}/{total_epochs} | "
                f"Batch {batch_idx:04d}/{total_batches:04d} | "
                f"Running Loss: {running_loss:.5f} | "
                f"Epoch ETA: {format_duration(epoch_eta)}",
                flush=True,
            )

    return total_loss / total_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        total_loss += float(loss.item())
        total_dice += float(dice_coeff(preds, masks).item())

    return total_loss / max(len(dataloader), 1), total_dice / max(len(dataloader), 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_dice: float,
    args: argparse.Namespace,
    best_val_loss: Optional[float] = None,
    best_val_dice: Optional[float] = None,
) -> None:
    """Save a training checkpoint that can be resumed later."""
    safe_args = {}
    for key, value in vars(args).items():
        # Avoid saving function objects such as args.func.
        # PyTorch 2.6+ blocks these by default during torch.load.
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, tuple)):
            safe_args[key] = value

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_dice": val_dice,
        "best_val_loss": best_val_loss if best_val_loss is not None else val_loss,
        "best_val_dice": best_val_dice if best_val_dice is not None else val_dice,
        "args": safe_args,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))


# def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict:
#     checkpoint_path = Path(checkpoint_path)
#     if not checkpoint_path.exists():
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

#     checkpoint = torch.load(str(checkpoint_path), map_location=device)
#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         state_dict = checkpoint["model_state_dict"]
#     else:
#         state_dict = checkpoint

#     # Handles checkpoints saved from DataParallel.
#     cleaned_state_dict = {}
#     for key, value in state_dict.items():
#         cleaned_key = key.replace("module.", "", 1) if key.startswith("module.") else key
#         cleaned_state_dict[cleaned_key] = value

#     model.load_state_dict(cleaned_state_dict)
#     return checkpoint if isinstance(checkpoint, dict) else {}
def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # PyTorch 2.6 changed torch.load default behavior.
    # This checkpoint was created locally by our own training code, so weights_only=False is safe here.
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that do not support weights_only.
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}

def load_training_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[int, float, float]:
    """Load model + optimizer so training can continue from the next epoch."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)

    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        except Exception as exc:
            print(f"[Warning] Could not fully restore optimizer state: {exc}")

    completed_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
    best_val_dice = float(checkpoint.get("best_val_dice", checkpoint.get("val_dice", -float("inf"))))

    print(f"Resumed from checkpoint: {checkpoint_path}")
    print(f"Completed epoch in checkpoint: {completed_epoch}")
    print(f"Best validation Dice so far: {best_val_dice:.5f}")
    print(f"Best validation Loss so far: {best_val_loss:.5f}")

    return completed_epoch + 1, best_val_loss, best_val_dice


def save_preview(
    output_path: Path,
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
) -> None:
    """Save a simple side-by-side preview without opening a GUI window."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = image.squeeze().detach().cpu().numpy()
    mask_np = mask.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("DICOM Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth PNG")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def append_training_log(log_path: Path, epoch: int, train_loss: float, val_loss: float, val_dice: float, epoch_seconds: float) -> None:
    """Append epoch metrics to a CSV log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as file:
        if write_header:
            file.write("epoch,train_loss,val_loss,val_dice,epoch_seconds\n")
        file.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{val_dice:.8f},{epoch_seconds:.2f}\n")


def patient_level_split_indices(
    dataset: DicomPngSegmentationDataset,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int], dict]:
    """
    Split by patient ID, not by slice.

    This prevents data leakage where neighboring slices from the same patient
    appear in both training and validation.
    """
    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        patient_to_indices[sample.patient_id].append(index)

    patients = sorted(patient_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    total_patients = len(patients)
    if total_patients < 3:
        raise RuntimeError(
            "Patient-level split requires at least 3 patients. "
            f"Found only {total_patients} patient(s)."
        )

    test_count = int(round(total_patients * test_split))
    val_count = int(round(total_patients * val_split))

    if test_split > 0:
        test_count = max(1, test_count)
    else:
        test_count = 0

    val_count = max(1, val_count)

    if val_count + test_count >= total_patients:
        raise RuntimeError(
            "Validation/test split is too large for the number of patients. "
            f"Patients={total_patients}, val_count={val_count}, test_count={test_count}"
        )

    test_patients = patients[:test_count]
    val_patients = patients[test_count:test_count + val_count]
    train_patients = patients[test_count + val_count:]

    def collect(patient_list: Sequence[str]) -> List[int]:
        indices: List[int] = []
        for patient_id in patient_list:
            indices.extend(patient_to_indices[patient_id])
        return sorted(indices)

    train_indices = collect(train_patients)
    val_indices = collect(val_patients)
    test_indices = collect(test_patients)

    split_info = {
        "split_mode": "patient",
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "num_patients": {
            "train": len(train_patients),
            "val": len(val_patients),
            "test": len(test_patients),
            "total": total_patients,
        },
        "num_samples": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
            "total": len(dataset),
        },
    }

    return train_indices, val_indices, test_indices, split_info


def save_split_json(path: Path, split_info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2, ensure_ascii=False)
    print(f"Saved split JSON: {path}")


def load_split_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split JSON not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def filter_dataset_by_patients(
    dataset: DicomPngSegmentationDataset,
    patient_ids: Sequence[str],
) -> Subset:
    patient_set = set(str(p) for p in patient_ids)
    indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample.patient_id in patient_set
    ]
    if not indices:
        raise RuntimeError("No samples matched the requested patient split.")
    return Subset(dataset, indices)



class AugmentedSegmentationDataset(Dataset):
    """
    Training-only augmentation wrapper for image/mask segmentation pairs.
    Validation/test data remain untouched.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotate_degrees: float = 7.0,
        translate_ratio: float = 0.03,
        brightness: float = 0.10,
        contrast: float = 0.10,
        noise_std: float = 0.01,
    ):
        self.dataset = dataset
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate_degrees = float(rotate_degrees)
        self.translate_ratio = float(translate_ratio)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.dataset)

    def _random_affine_pair(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotate_degrees <= 0 and self.translate_ratio <= 0:
            return image, mask

        angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
        angle_rad = np.deg2rad(angle)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        tx = random.uniform(-self.translate_ratio, self.translate_ratio) * 2.0
        ty = random.uniform(-self.translate_ratio, self.translate_ratio) * 2.0

        theta = torch.tensor(
            [[[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]],
            dtype=image.dtype,
            device=image.device,
        )

        image_b = image.unsqueeze(0)
        mask_b = mask.unsqueeze(0)
        grid = F.affine_grid(theta, size=image_b.size(), align_corners=False)

        image_aug = F.grid_sample(
            image_b,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        mask_aug = F.grid_sample(
            mask_b,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        return image_aug, (mask_aug > 0.5).float()

    def _jitter_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.contrast > 0:
            contrast_factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean

        if self.brightness > 0:
            brightness_shift = random.uniform(-self.brightness, self.brightness)
            image = image + brightness_shift

        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std

        return image.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = dict(self.dataset[index])
        image = item["image"].clone()
        mask = item["mask"].clone()

        if random.random() < self.hflip_prob:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        image, mask = self._random_affine_pair(image, mask)
        image = self._jitter_image(image)

        item["image"] = image.float()
        item["mask"] = (mask > 0.5).float()
        return item


def describe_augmentation(args: argparse.Namespace) -> None:
    print("Training augmentation: ENABLED")
    print(
        f"  hflip_prob={args.aug_hflip_prob}, "
        f"vflip_prob={args.aug_vflip_prob}, "
        f"rotate_degrees={args.aug_rotate_degrees}, "
        f"translate_ratio={args.aug_translate_ratio}, "
        f"brightness={args.aug_brightness}, "
        f"contrast={args.aug_contrast}, "
        f"noise_std={args.aug_noise_std}"
    )


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = DicomPngSegmentationDataset(
        dicom_root=args.dicom_root,
        mask_root=args.mask_root,
        image_size=tuple(args.image_size) if args.image_size else None,
        require_masks=True,
        mask_clean_mode=args.mask_clean_mode,
        mask_threshold=args.mask_threshold,
        mask_min_area=args.mask_min_area,
        mask_max_area_ratio=args.mask_max_area_ratio,
        mask_border_ratio=args.mask_border_ratio,
        mask_dilate_radius=args.mask_dilate_radius,
    )

    save_training_mask_debug_previews(dataset, save_dir, count=args.mask_debug_count)

    if len(dataset) < 2:
        raise RuntimeError("At least 2 image/mask pairs are required for train/validation split.")

    split_mode = getattr(args, "split_mode", "random")

    if split_mode == "patient":
        split_json_path = Path(args.split_json) if args.split_json else save_dir / "patient_split.json"

        if getattr(args, "use_existing_split", False):
            split_info = load_split_json(split_json_path)
            train_dataset = filter_dataset_by_patients(dataset, split_info["train_patients"])
            valid_dataset = filter_dataset_by_patients(dataset, split_info["val_patients"])
            print(f"Loaded existing patient split JSON: {split_json_path}")
        else:
            train_indices, val_indices, test_indices, split_info = patient_level_split_indices(
                dataset=dataset,
                val_split=args.val_split,
                test_split=args.test_split,
                seed=args.seed,
            )
            train_dataset = Subset(dataset, train_indices)
            valid_dataset = Subset(dataset, val_indices)
            save_split_json(split_json_path, split_info)

        print(
            "Patient-level split: "
            f"{split_info['num_patients']['train']} train patient(s), "
            f"{split_info['num_patients']['val']} validation patient(s), "
            f"{split_info['num_patients']['test']} test patient(s)"
        )
    else:
        valid_size = max(1, int(round(len(dataset) * args.val_split)))
        train_size = len(dataset) - valid_size
        if train_size <= 0:
            raise RuntimeError("Validation split is too large. Reduce --val-split.")

        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=generator)

    if getattr(args, "augment", False):
        describe_augmentation(args)
        train_dataset = AugmentedSegmentationDataset(
            train_dataset,
            hflip_prob=args.aug_hflip_prob,
            vflip_prob=args.aug_vflip_prob,
            rotate_degrees=args.aug_rotate_degrees,
            translate_ratio=args.aug_translate_ratio,
            brightness=args.aug_brightness,
            contrast=args.aug_contrast,
            noise_std=args.aug_noise_std,
        )
    else:
        print("Training augmentation: DISABLED")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = build_criterion(args, device, train_dataset=train_dataset)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_val_dice = -float("inf")
    start_epoch = 1

    resume_path: Optional[Path] = None
    if getattr(args, "resume", None):
        resume_path = Path(args.resume)
    elif getattr(args, "auto_resume", False):
        candidate = save_dir / "last_model.pth"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        start_epoch, best_val_loss, best_val_dice = load_training_checkpoint(model, optimizer, resume_path, device)
        if start_epoch > args.epochs:
            print(f"Checkpoint already reached epoch {start_epoch - 1}; requested --epochs {args.epochs}. Nothing to train.")
            return

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(valid_loader)}")
    print(f"Starting epoch: {start_epoch}/{args.epochs}")

    training_start = time.time()
    completed_epoch_times: List[float] = []
    current_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            epoch_start = time.time()

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch=epoch,
                total_epochs=args.epochs,
                log_every=args.log_every,
            )
            val_loss, val_dice = validate(model, valid_loader, criterion, device, threshold=args.threshold)

            epoch_seconds = time.time() - epoch_start
            completed_epoch_times.append(epoch_seconds)
            avg_epoch_seconds = sum(completed_epoch_times) / len(completed_epoch_times)
            remaining_epochs = args.epochs - epoch
            eta_seconds = remaining_epochs * avg_epoch_seconds
            estimated_finish = datetime.now() + timedelta(seconds=eta_seconds)

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val Dice: {val_dice:.5f} | "
                f"Epoch Time: {format_duration(epoch_seconds)} | "
                f"Total ETA: {format_duration(eta_seconds)} | "
                f"Est. Finish: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )

            append_training_log(save_dir / "training_log.csv", epoch, train_loss, val_loss, val_dice, epoch_seconds)

            save_checkpoint(
                save_dir / "last_model.pth",
                model,
                optimizer,
                epoch,
                val_loss,
                val_dice,
                args,
                best_val_loss=best_val_loss,
                best_val_dice=best_val_dice,
            )

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_checkpoint(
                    save_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    val_dice,
                    args,
                    best_val_loss=best_val_loss,
                    best_val_dice=best_val_dice,
                )

            improved = val_dice > best_val_dice or (np.isclose(val_dice, best_val_dice) and val_loss < best_val_loss)
            if improved:
                best_val_dice = val_dice
                best_val_loss = val_loss
                save_checkpoint(
                    save_dir / "best_model.pth",
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    val_dice,
                    args,
                    best_val_loss=best_val_loss,
                    best_val_dice=best_val_dice,
                )
                save_checkpoint(
                    save_dir / f"best_model_epoch_{epoch:03d}_dice_{val_dice:.4f}_loss_{val_loss:.4f}.pth",
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    val_dice,
                    args,
                    best_val_loss=best_val_loss,
                    best_val_dice=best_val_dice,
                )
                print(f"  Saved new best model: {save_dir / 'best_model.pth'}")

            if args.save_previews:
                preview_batch = next(iter(valid_loader))
                images = preview_batch["image"].to(device)
                masks = preview_batch["mask"].to(device)
                with torch.no_grad():
                    preds = (torch.sigmoid(model(images)) >= args.threshold).float()
                save_preview(save_dir / "previews" / f"epoch_{epoch:03d}.png", images[0], masks[0], preds[0])

    except KeyboardInterrupt:
        interrupt_path = save_dir / "interrupt_model.pth"
        print("\nTraining interrupted. Saving emergency checkpoint...")
        save_checkpoint(
            interrupt_path,
            model,
            optimizer,
            current_epoch,
            val_loss if "val_loss" in locals() else float("inf"),
            val_dice if "val_dice" in locals() else -float("inf"),
            args,
            best_val_loss=best_val_loss,
            best_val_dice=best_val_dice,
        )
        print(f"Emergency checkpoint saved: {interrupt_path}")
        print(f"Resume later with: python main.py train --resume \"{interrupt_path}\" ...")
        return

    total_seconds = time.time() - training_start
    print("Training completed.")
    print(f"Total training time: {format_duration(total_seconds)}")
    print(f"Best validation Dice: {best_val_dice:.5f}")
    print(f"Best validation Loss: {best_val_loss:.5f}")
    print(f"Best checkpoint: {save_dir / 'best_model.pth'}")


# -----------------------------------------------------------------------------
# Evaluation metrics
# -----------------------------------------------------------------------------


def binary_metrics_from_masks(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    tp = float(np.logical_and(pred_bool, target_bool).sum())
    fp = float(np.logical_and(pred_bool, ~target_bool).sum())
    fn = float(np.logical_and(~pred_bool, target_bool).sum())
    tn = float(np.logical_and(~pred_bool, ~target_bool).sum())

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pixels": float(pred_bool.sum()),
        "target_pixels": float(target_bool.sum()),
    }


def hd95_distance(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """
    Compute approximate 2D HD95 using distance transforms.

    Returns None if scipy is unavailable or one mask is empty.
    """
    if scipy_ndimage is None:
        return None

    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if pred_bool.sum() == 0 or target_bool.sum() == 0:
        return None

    pred_border = pred_bool ^ morphology.erosion(pred_bool)
    target_border = target_bool ^ morphology.erosion(target_bool)

    if pred_border.sum() == 0 or target_border.sum() == 0:
        return None

    dist_to_target = scipy_ndimage.distance_transform_edt(~target_border)
    dist_to_pred = scipy_ndimage.distance_transform_edt(~pred_border)

    distances_pred_to_target = dist_to_target[pred_border]
    distances_target_to_pred = dist_to_pred[target_border]

    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    if all_distances.size == 0:
        return None

    return float(np.percentile(all_distances, 95))


@torch.no_grad()
def evaluate_model(args: argparse.Namespace) -> None:
    """
    Evaluate a trained checkpoint against DICOM + PNG masks and export CSV metrics.

    Supports patient-level test evaluation by passing:
        --split-json runs/.../patient_split.json --split-name test
    """
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DicomPngSegmentationDataset(
        dicom_root=args.dicom_root,
        mask_root=args.mask_root,
        image_size=tuple(args.image_size) if args.image_size else None,
        require_masks=True,
        mask_clean_mode=args.mask_clean_mode,
        mask_threshold=args.mask_threshold,
        mask_min_area=args.mask_min_area,
        mask_max_area_ratio=args.mask_max_area_ratio,
        mask_border_ratio=args.mask_border_ratio,
        mask_dilate_radius=args.mask_dilate_radius,
    )

    eval_dataset: torch.utils.data.Dataset = dataset

    if args.split_json:
        split_info = load_split_json(args.split_json)
        key = f"{args.split_name}_patients"
        if key not in split_info:
            raise RuntimeError(f"Split JSON does not contain key: {key}")
        eval_dataset = filter_dataset_by_patients(dataset, split_info[key])
        print(f"Evaluating split '{args.split_name}' with {len(eval_dataset)} sample(s).")
    else:
        print(f"Evaluating all matched samples: {len(eval_dataset)}")

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(n_channels=1, n_classes=1).to(device)
    checkpoint_info = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    print(f"Device: {device}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    if checkpoint_info:
        print(
            f"Checkpoint epoch: {checkpoint_info.get('epoch', 'unknown')} | "
            f"Val Dice: {checkpoint_info.get('val_dice', 'unknown')} | "
            f"Val Loss: {checkpoint_info.get('val_loss', 'unknown')}"
        )

    rows: List[Dict[str, object]] = []
    preview_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= args.threshold).astype(np.float32)
        images_np = images.cpu().numpy()

        for index in range(preds.shape[0]):
            pred_mask = preds[index, 0]
            target_mask = masks[index, 0]
            metrics = binary_metrics_from_masks(pred_mask, target_mask)
            hd95 = hd95_distance(pred_mask, target_mask)
            metrics["hd95"] = "" if hd95 is None else hd95

            patient_id = batch["patient_id"][index]
            phase = batch["phase"][index]
            dcm_path = Path(batch["dicom_path"][index])
            mask_path = Path(batch["mask_path"][index]) if "mask_path" in batch else Path("")

            row = {
                "patient_id": patient_id,
                "phase": phase,
                "dicom_path": str(dcm_path),
                "mask_path": str(mask_path),
                **metrics,
            }
            rows.append(row)

            if args.save_overlays and preview_count < args.max_eval_previews:
                relative_out_dir = output_dir / "overlays" / patient_id / phase
                save_overlay_png(images_np[index, 0], pred_mask, relative_out_dir / f"{dcm_path.stem}_overlay.png")
                save_mask_png(pred_mask, output_dir / "predicted_masks" / patient_id / phase / f"{dcm_path.stem}_mask.png")
                preview_count += 1

    if not rows:
        raise RuntimeError("No evaluation rows were created.")

    csv_path = output_dir / "evaluation_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8") as file:
        file.write(",".join(fieldnames) + "\n")
        for row in rows:
            values = []
            for field in fieldnames:
                value = row[field]
                if isinstance(value, float):
                    values.append(f"{value:.8f}")
                else:
                    text = str(value).replace('"', '""')
                    values.append(f'"{text}"' if "," in text else text)
            file.write(",".join(values) + "\n")

    # Summary
    numeric_keys = ["dice", "iou", "precision", "recall", "specificity", "accuracy"]
    summary = {}
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std())
        summary[f"median_{key}"] = float(np.median(values))

    hd95_values = [float(row["hd95"]) for row in rows if row["hd95"] != ""]
    if hd95_values:
        summary["mean_hd95"] = float(np.mean(hd95_values))
        summary["median_hd95"] = float(np.median(hd95_values))

    summary["num_samples"] = len(rows)
    summary["threshold"] = args.threshold
    summary["checkpoint"] = str(args.checkpoint)

    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Evaluation completed.")
    print(f"Metrics CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(
        f"Mean Dice: {summary['mean_dice']:.5f} | "
        f"Mean IoU: {summary['mean_iou']:.5f} | "
        f"Mean Precision: {summary['mean_precision']:.5f} | "
        f"Mean Recall: {summary['mean_recall']:.5f}"
    )


# -----------------------------------------------------------------------------
# Prediction / output saving
# -----------------------------------------------------------------------------


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(path)


def save_overlay_png(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (image.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.stack([image_uint8, image_uint8, image_uint8], axis=-1)

    mask_bool = mask.astype(bool)
    overlay = rgb.copy()
    overlay[mask_bool, 0] = 255
    overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.35).astype(np.uint8)
    overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.35).astype(np.uint8)

    blended = (0.65 * rgb + 0.35 * overlay).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(path)


def find_original_mesh(original_mesh_root: Optional[str], patient_id: str, phase: str) -> Optional[Path]:
    if not original_mesh_root:
        return None

    root = Path(original_mesh_root)
    candidates = [
        root / f"{patient_id}_{phase}.ply",
        root / patient_id / f"{phase}.ply",
        root / f"{patient_id}.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def export_volume_mesh(
    volume: np.ndarray,
    output_ply: Path,
    output_html: Optional[Path] = None,
    original_mesh_path: Optional[Path] = None,
    threshold: float = 0.5,
) -> None:
    """
    Convert a predicted 3D binary volume into a mesh.
    volume shape is expected as (slices, height, width).
    """
    unique_values = np.unique(volume)
    if len(unique_values) < 2:
        print(f"[Mesh skipped] Volume has only one value: {unique_values[0]}")
        return

    min_val = float(volume.min())
    max_val = float(volume.max())
    level = threshold
    if not (min_val < level < max_val):
        level = (min_val + max_val) / 2.0

    verts, faces, _, _ = measure.marching_cubes(volume.astype(np.float32), level=level)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    if o3d is None:
        raise ImportError("open3d is required for mesh export. Install it with: pip install open3d")

    mesh = o3d.geometry.TriangleMesh()
    # marching_cubes returns z,y,x; Open3D expects x,y,z
    xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])
    mesh.vertices = o3d.utility.Vector3dVector(xyz)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_ply), mesh)
    print(f"Saved generated mesh: {output_ply}")

    if output_html is not None:
        if go is None or pio is None:
            print("[HTML skipped] plotly is not installed. Install it with: pip install plotly")
            return

        output_html.parent.mkdir(parents=True, exist_ok=True)
        mesh_trace = go.Mesh3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.55,
            color="cyan",
            name="Generated prediction mesh",
        )
        data = [mesh_trace]

        if original_mesh_path is not None and original_mesh_path.exists():
            original_mesh = o3d.io.read_triangle_mesh(str(original_mesh_path))
            original_vertices = np.asarray(original_mesh.vertices)
            original_faces = np.asarray(original_mesh.triangles)
            if len(original_vertices) > 0 and len(original_faces) > 0:
                data.append(
                    go.Mesh3d(
                        x=original_vertices[:, 0],
                        y=original_vertices[:, 1],
                        z=original_vertices[:, 2],
                        i=original_faces[:, 0],
                        j=original_faces[:, 1],
                        k=original_faces[:, 2],
                        opacity=0.35,
                        color="red",
                        name="Original mesh",
                    )
                )

        fig = go.Figure(data=data)
        fig.update_layout(
            title="3D Segmentation Mesh Preview",
            scene=dict(
                xaxis_title="Width / X",
                yaxis_title="Height / Y",
                zaxis_title="Slice / Z",
                aspectmode="data",
            ),
        )
        pio.write_html(fig, str(output_html), auto_open=False)
        print(f"Saved mesh HTML preview: {output_html}")


@torch.no_grad()
def predict_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DicomPngSegmentationDataset(
        dicom_root=args.dicom_root,
        mask_root=None,
        image_size=tuple(args.image_size) if args.image_size else None,
        require_masks=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(n_channels=1, n_classes=1).to(device)
    checkpoint_info = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    print(f"Device: {device}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    if checkpoint_info:
        print(
            f"Checkpoint epoch: {checkpoint_info.get('epoch', 'unknown')} | "
            f"Val Dice: {checkpoint_info.get('val_dice', 'unknown')} | "
            f"Val Loss: {checkpoint_info.get('val_loss', 'unknown')}"
        )

    volume_groups: Dict[Tuple[str, str], List[Tuple[float, np.ndarray, str]]] = defaultdict(list)

    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= args.threshold).float().cpu().numpy()
        images_np = images.cpu().numpy()

        batch_size = preds.shape[0]
        for index in range(batch_size):
            patient_id = batch["patient_id"][index]
            phase = batch["phase"][index]
            dcm_path = Path(batch["dicom_path"][index])
            slice_location = float(batch["slice_location"][index].item())

            pred_mask = preds[index, 0].astype(np.float32)
            image_np = images_np[index, 0].astype(np.float32)

            relative_out_dir = output_dir / patient_id / phase
            mask_path = relative_out_dir / "predicted_masks" / f"{dcm_path.stem}_mask.png"
            save_mask_png(pred_mask, mask_path)

            if args.save_overlays:
                overlay_path = relative_out_dir / "overlays" / f"{dcm_path.stem}_overlay.png"
                save_overlay_png(image_np, pred_mask, overlay_path)

            volume_groups[(patient_id, phase)].append((slice_location, pred_mask, dcm_path.stem))

    print(f"Saved predicted masks to: {output_dir}")

    if args.export_mesh:
        for (patient_id, phase), slices in volume_groups.items():
            slices_sorted = sorted(slices, key=lambda item: item[0])
            volume = np.stack([item[1] for item in slices_sorted], axis=0)

            patient_phase_dir = output_dir / patient_id / phase
            npy_path = patient_phase_dir / f"{patient_id}_{phase}_predicted_volume.npy"
            np.save(npy_path, volume.astype(np.float32))
            print(f"Saved predicted volume: {npy_path}")

            original_mesh_path = find_original_mesh(args.original_mesh_root, patient_id, phase)
            output_ply = patient_phase_dir / f"{patient_id}_{phase}_generated_mesh.ply"
            output_html = patient_phase_dir / f"{patient_id}_{phase}_mesh_preview.html"
            export_volume_mesh(
                volume=volume,
                output_ply=output_ply,
                output_html=output_html,
                original_mesh_path=original_mesh_path,
                threshold=args.threshold,
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Prediction completed.")



# -----------------------------------------------------------------------------
# Standalone mask-cleaning command
# -----------------------------------------------------------------------------


def clean_masks_command(args: argparse.Namespace) -> None:
    """
    Create a new folder of clean binary PNG masks from the current PNG folder.

    This does NOT require mesh files or dataset_generation_batch.py.
    It only uses the existing PNG files and applies the same cleaning logic used
    during training.
    """
    mask_root = Path(args.mask_root)
    output_mask_root = Path(args.output_mask_root)
    output_mask_root.mkdir(parents=True, exist_ok=True)

    mask_files = collect_files(mask_root, MASK_EXTENSIONS)
    if not mask_files:
        raise RuntimeError(f"No PNG masks found under: {mask_root}")

    print(f"Found {len(mask_files)} PNG file(s) under: {mask_root}")
    print(f"Saving cleaned binary masks to: {output_mask_root}")
    print(f"Mask cleaning mode: {args.mask_clean_mode}")

    report_path = output_mask_root / "clean_mask_report.csv"
    debug_dir = output_mask_root / "_debug_previews"
    debug_saved = 0

    with report_path.open("w", encoding="utf-8") as report:
        report.write("relative_path,original_unique_count,cleaned_unique_count,cleaned_foreground_pixels\n")

        for index, mask_path in enumerate(mask_files, start=1):
            relative_path = mask_path.relative_to(mask_root)
            output_path = output_mask_root / relative_path

            original_gray = np.asarray(Image.open(mask_path).convert("L"))
            original_unique_count = len(np.unique(original_gray))

            cleaned = read_png_mask(
                mask_path,
                mode=args.mask_clean_mode,
                threshold=args.mask_threshold,
                min_area=args.mask_min_area,
                max_area_ratio=args.mask_max_area_ratio,
                border_ratio=args.mask_border_ratio,
                dilate_radius=args.mask_dilate_radius,
            )

            write_clean_mask_png(cleaned, output_path)

            cleaned_uint8 = (cleaned > 0.5).astype(np.uint8) * 255
            cleaned_unique_count = len(np.unique(cleaned_uint8))
            foreground_pixels = int((cleaned_uint8 > 0).sum())

            report.write(
                f"{relative_path.as_posix()},"
                f"{original_unique_count},"
                f"{cleaned_unique_count},"
                f"{foreground_pixels}\n"
            )

            if args.save_debug and debug_saved < args.mask_debug_count:
                safe_debug_name = relative_path.as_posix().replace("/", "_").replace("\\\\", "_")
                save_clean_mask_debug_image(mask_path, cleaned, debug_dir / f"{safe_debug_name}_debug.png")
                debug_saved += 1

            if index == 1 or index % 250 == 0 or index == len(mask_files):
                print(f"  Cleaned {index}/{len(mask_files)} mask(s)", flush=True)

    print("Mask cleaning completed.")
    print(f"Clean mask report: {report_path}")
    if args.save_debug:
        print(f"Debug previews: {debug_dir}")
    print("")
    print("Next recommended training command:")
    print(
        'python main.py train `\n'
        '  --dicom-root "DICOM" `\n'
        f'  --mask-root "{output_mask_root}" `\n'
        '  --save-dir "runs/diaphragm_unet_clean_test" `\n'
        '  --epochs 1 `\n'
        '  --batch-size 4 `\n'
        '  --image-size 512 512 `\n'
        '  --mask-clean-mode binary `\n'
        '  --loss dice_bce `\n'
        '  --pos-weight 20 `\n'
        '  --save-previews'
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"), help="Optional resize size, for example: --image-size 512 512")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask prediction.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count. Use 0 on Windows if multiprocessing causes issues.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")


def add_mask_cleaning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mask-clean-mode",
        choices=["auto", "binary", "color", "bright", "none"],
        default="auto",
        help=(
            "How to convert PNG labels into binary masks. "
            "Use auto for current visualization-style PNGs; use binary for already-clean 0/255 masks."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.85,
        help="Bright-pixel threshold used by bright/auto modes. 0.85 means pixels >= 85%% brightness become foreground.",
    )
    parser.add_argument(
        "--mask-min-area",
        type=int,
        default=10,
        help="Remove connected components smaller than this pixel area during mask cleaning.",
    )
    parser.add_argument(
        "--mask-max-area-ratio",
        type=float,
        default=0.25,
        help="Remove connected components larger than this fraction of the PNG area. Use 0 to disable.",
    )
    parser.add_argument(
        "--mask-border-ratio",
        type=float,
        default=0.08,
        help="Remove likely plot title/axis artifacts near the image borders. Use 0 to disable.",
    )
    parser.add_argument(
        "--mask-dilate-radius",
        type=int,
        default=3,
        help="Dilate extracted mask by this radius. Helps thicken thin plotted points/lines. Use 0 to disable. Default: 3",
    )
    parser.add_argument(
        "--mask-debug-count",
        type=int,
        default=12,
        help="Number of before/after cleaned mask debug previews to save.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and predict U-Net segmentation masks from DICOM images and PNG masks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train U-Net using DICOM images and PNG masks.")
    train_parser.add_argument("--dicom-root", required=True, help="Folder containing training DICOM files.")
    train_parser.add_argument("--mask-root", required=True, help="Folder containing training PNG mask files.")
    train_parser.add_argument("--save-dir", required=True, help="Folder where checkpoints and previews will be saved.")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    train_parser.add_argument("--val-split", type=float, default=0.2, help="Validation ratio. Default: 0.2")
    train_parser.add_argument("--split-mode", choices=["random", "patient"], default="random", help="Use random slice split or patient-level split. Default: random")
    train_parser.add_argument("--test-split", type=float, default=0.0, help="Patient-level test ratio when --split-mode patient. Default: 0")
    train_parser.add_argument("--split-json", default=None, help="Where to save/load patient split JSON. Default: save-dir/patient_split.json")
    train_parser.add_argument("--use-existing-split", action="store_true", help="Load an existing patient split JSON instead of creating a new split.")
    train_parser.add_argument("--save-previews", action="store_true", help="Save one validation preview image after each epoch.")

    # Training-only augmentation settings.
    train_parser.add_argument("--augment", action="store_true", help="Enable training-only data augmentation.")
    train_parser.add_argument("--aug-hflip-prob", type=float, default=0.5, help="Random horizontal flip probability. Default: 0.5")
    train_parser.add_argument("--aug-vflip-prob", type=float, default=0.0, help="Random vertical flip probability. Default: 0.0")
    train_parser.add_argument("--aug-rotate-degrees", type=float, default=7.0, help="Random rotation range in degrees. Default: 7")
    train_parser.add_argument("--aug-translate-ratio", type=float, default=0.03, help="Random translation ratio. Default: 0.03")
    train_parser.add_argument("--aug-brightness", type=float, default=0.10, help="Brightness jitter range. Default: 0.10")
    train_parser.add_argument("--aug-contrast", type=float, default=0.10, help="Contrast jitter range. Default: 0.10")
    train_parser.add_argument("--aug-noise-std", type=float, default=0.01, help="Gaussian noise std. Default: 0.01")
    train_parser.add_argument("--resume", default=None, help="Resume training from a checkpoint path, usually runs/.../last_model.pth or interrupt_model.pth.")
    train_parser.add_argument("--auto-resume", action="store_true", help="Automatically resume from save-dir/last_model.pth if it exists.")
    train_parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint_epoch_XXX.pth every N epochs. Default: 1")
    train_parser.add_argument("--log-every", type=int, default=25, help="Print batch progress every N batches. Use 0 to disable batch logs.")

    # Loss settings. Dice+BCE is recommended for sparse diaphragm masks.
    train_parser.add_argument("--loss", choices=["dice_bce", "bce"], default="dice_bce", help="Training loss. Default: dice_bce")
    train_parser.add_argument("--bce-weight", type=float, default=0.3, help="BCE contribution when --loss dice_bce. Default: 0.3")
    train_parser.add_argument("--dice-weight", type=float, default=0.7, help="Dice contribution when --loss dice_bce. Default: 0.7")
    train_parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive-class BCE weight. Useful for tiny foreground masks. Default: 20.0")
    train_parser.add_argument("--auto-pos-weight", action="store_true", help="Estimate pos_weight from training masks automatically.")
    train_parser.add_argument("--pos-weight-sample-count", type=int, default=256, help="Number of masks to inspect for --auto-pos-weight.")
    train_parser.add_argument("--max-pos-weight", type=float, default=100.0, help="Maximum cap for --auto-pos-weight. Default: 100")

    add_common_args(train_parser)
    add_mask_cleaning_args(train_parser)
    train_parser.set_defaults(func=train_model)

    clean_parser = subparsers.add_parser("clean-masks", help="Create clean binary PNG masks from the current PNG folder.")
    clean_parser.add_argument("--mask-root", required=True, help="Folder containing current PNG masks or visualization PNGs.")
    clean_parser.add_argument("--output-mask-root", required=True, help="Folder where cleaned binary PNG masks will be saved.")
    clean_parser.add_argument("--save-debug", action="store_true", help="Save before/after debug previews for cleaned masks.")
    add_mask_cleaning_args(clean_parser)
    clean_parser.set_defaults(func=clean_masks_command)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint against DICOM + PNG masks and export metrics.")
    eval_parser.add_argument("--dicom-root", required=True, help="Folder containing DICOM files.")
    eval_parser.add_argument("--mask-root", required=True, help="Folder containing PNG mask files.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint, usually best_model.pth.")
    eval_parser.add_argument("--output-dir", required=True, help="Folder where evaluation metrics and overlays will be saved.")
    eval_parser.add_argument("--split-json", default=None, help="Optional patient_split.json to evaluate a specific split.")
    eval_parser.add_argument("--split-name", choices=["train", "val", "test"], default="test", help="Which split to evaluate when --split-json is provided.")
    eval_parser.add_argument("--save-overlays", action="store_true", help="Save a limited number of overlay previews.")
    eval_parser.add_argument("--max-eval-previews", type=int, default=100, help="Maximum overlay/mask previews to save during evaluation.")
    add_common_args(eval_parser)
    add_mask_cleaning_args(eval_parser)
    eval_parser.set_defaults(func=evaluate_model)

    predict_parser = subparsers.add_parser("predict", help="Predict PNG masks from DICOM files.")
    predict_parser.add_argument("--dicom-root", required=True, help="Folder containing DICOM files to predict.")
    predict_parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint, usually best_model.pth.")
    predict_parser.add_argument("--output-dir", required=True, help="Folder where prediction outputs will be saved.")
    predict_parser.add_argument("--save-overlays", action="store_true", help="Save red overlay previews on top of DICOM images.")
    predict_parser.add_argument("--export-mesh", action="store_true", help="Stack predicted masks and export a 3D PLY mesh + HTML preview.")
    predict_parser.add_argument("--original-mesh-root", default=None, help="Optional folder containing original .ply meshes for HTML comparison.")
    add_common_args(predict_parser)
    predict_parser.set_defaults(func=predict_model)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
