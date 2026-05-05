import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class SinusDataset(Dataset):
    """
    PyTorch Dataset for paired DICOM image slices and PNG masks.

    Supports BOTH layouts:

    Layout A: phase folders
        img_dir/
            patient_id/
                in/*.dcm
                ex/*.dcm

        label_dir/
            patient_id/
                in/*_axial.png
                ex/*_axial.png

    Layout B: direct patient folders
        img_dir/
            patient_id/*.dcm

        label_dir/
            patient_id/*_axial.png

    Example:
        DICOM/10051764/IM-0002-0372.dcm
        PNG/10051764/372_axial.png
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        target_size: Tuple[int, int] = (512, 512),
        require_labels: bool = True,
        verbose: bool = False,
    ) -> None:
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.require_labels = require_labels
        self.verbose = verbose

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")

        self.samples: List[Dict[str, Optional[str]]] = []
        self._build_index()

        if len(self.samples) == 0:
            raise ValueError(
                "No valid image/label pairs were found. "
                f"img_dir={self.img_dir}, label_dir={self.label_dir}"
            )

    def _build_index(self) -> None:
        skipped_missing_label = 0
        skipped_invalid_dicom = 0

        for patient in sorted(os.listdir(self.img_dir)):
            patient_path = os.path.join(self.img_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            scan_dirs = self._discover_scan_dirs(patient_path)

            for phase, dicom_dir in scan_dirs:
                dicom_files = sorted(
                    f for f in os.listdir(dicom_dir)
                    if f.lower().endswith(".dcm")
                )

                for file_name in dicom_files:
                    img_path = os.path.join(dicom_dir, file_name)

                    try:
                        slice_number = self._extract_slice_number(file_name)
                    except ValueError:
                        skipped_invalid_dicom += 1
                        if self.verbose:
                            print(f"[Skip] Could not parse slice number from filename: {img_path}")
                        continue

                    label_path = self._resolve_label_path(patient, phase, slice_number)

                    if self.require_labels and label_path is None:
                        skipped_missing_label += 1
                        if self.verbose:
                            print(f"[Skip] Missing label for image: {img_path}")
                        continue

                    self.samples.append(
                        {
                            "patient_id": patient,
                            "phase": phase,
                            "image_path": img_path,
                            "label_path": label_path,
                            "slice_number": str(slice_number),
                        }
                    )

        if self.verbose:
            print(f"Indexed {len(self.samples)} valid samples.")
            print(f"Skipped missing labels: {skipped_missing_label}")
            print(f"Skipped invalid DICOM filenames: {skipped_invalid_dicom}")

    @staticmethod
    def _discover_scan_dirs(patient_path: str) -> List[Tuple[Optional[str], str]]:
        """
        Detect whether a patient uses phase folders or direct DICOM files.

        Returns:
            [(phase, directory_path), ...]

        phase is:
            "in" or "ex" for phase-folder layout
            None for direct patient-folder layout
        """
        scan_dirs: List[Tuple[Optional[str], str]] = []

        # Layout A: patient/in/*.dcm and patient/ex/*.dcm
        for phase in ("in", "ex"):
            phase_dir = os.path.join(patient_path, phase)
            if os.path.isdir(phase_dir):
                has_dicom = any(
                    f.lower().endswith(".dcm") for f in os.listdir(phase_dir)
                )
                if has_dicom:
                    scan_dirs.append((phase, phase_dir))

        # Layout B: patient/*.dcm directly under the patient folder
        has_direct_dicoms = any(
            f.lower().endswith(".dcm") for f in os.listdir(patient_path)
        )
        if has_direct_dicoms:
            scan_dirs.append((None, patient_path))

        return scan_dirs

    @staticmethod
    def _extract_slice_number(file_name: str) -> int:
        """
        Extract the last integer found in the filename stem.

        Examples:
            IM-0002-0372.dcm -> 372
            IMG-0012.dcm -> 12
            410.dcm -> 410
            patient_ex_0045.dcm -> 45
            372_axial.png -> 372
        """
        stem = os.path.splitext(os.path.basename(file_name))[0]
        matches = re.findall(r"(\d+)", stem)
        if not matches:
            raise ValueError(f"No numeric slice number found in filename: {file_name}")
        return int(matches[-1])

    def _resolve_label_path(self, patient: str, phase: Optional[str], slice_number: int) -> Optional[str]:
        """
        Finds the matching PNG label.

        If phase is "in" or "ex", it first checks:
            label_dir/patient/phase/

        If phase is None, it checks:
            label_dir/patient/

        It also falls back between both locations when possible.
        """
        candidate_dirs: List[str] = []

        patient_label_dir = os.path.join(self.label_dir, patient)

        if phase is not None:
            candidate_dirs.append(os.path.join(patient_label_dir, phase))
            candidate_dirs.append(patient_label_dir)  # fallback
        else:
            candidate_dirs.append(patient_label_dir)
            candidate_dirs.append(os.path.join(patient_label_dir, "in"))
            candidate_dirs.append(os.path.join(patient_label_dir, "ex"))

        candidate_names = [
            f"{slice_number}_axial.png",
            f"{slice_number:03d}_axial.png",
            f"{slice_number:04d}_axial.png",
            f"{slice_number:05d}_axial.png",
        ]

        for base_dir in candidate_dirs:
            if not os.path.isdir(base_dir):
                continue

            for name in candidate_names:
                full_path = os.path.join(base_dir, name)
                if os.path.exists(full_path):
                    return full_path

            # Fallback: search any PNG ending with _axial.png whose number matches.
            for file_name in os.listdir(base_dir):
                lower_name = file_name.lower()
                if not lower_name.endswith(".png"):
                    continue
                if not lower_name.endswith("_axial.png"):
                    continue

                try:
                    file_slice = self._extract_slice_number(file_name)
                except ValueError:
                    continue

                if file_slice == slice_number:
                    return os.path.join(base_dir, file_name)

        return None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_dicom_image(dicom_path: str) -> Tuple[np.ndarray, pydicom.dataset.FileDataset]:
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array.astype(np.float32)

        slope = float(getattr(dicom_data, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom_data, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        finite_mask = np.isfinite(image)
        if not finite_mask.any():
            raise ValueError(f"DICOM contains no finite pixel values: {dicom_path}")

        finite_values = image[finite_mask]
        p1, p99 = np.percentile(finite_values, [1, 99])

        if p99 <= p1:
            min_val = float(finite_values.min())
            max_val = float(finite_values.max())
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = np.zeros_like(image, dtype=np.float32)
        else:
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1)

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        return image, dicom_data

    @staticmethod
    def process_label(label_image: Image.Image) -> torch.Tensor:
        label_array = np.array(label_image)

        if label_array.ndim == 2:
            mask = (label_array > 0).astype(np.float32)
        elif label_array.ndim == 3:
            channels = label_array.shape[2]

            if channels >= 3:
                red = label_array[:, :, 0]
                green = label_array[:, :, 1]
                blue = label_array[:, :, 2]

                red_mask = (red > 200) & (green < 50) & (blue < 50)
                non_black_mask = np.any(label_array[:, :, :3] > 0, axis=2)

                mask = red_mask.astype(np.float32) if red_mask.any() else non_black_mask.astype(np.float32)
            else:
                mask = (label_array[:, :, 0] > 0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported label image shape: {label_array.shape}")

        return torch.from_numpy(mask).unsqueeze(0).float()

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        label_path = sample["label_path"]

        image_np, dicom_data = self._load_dicom_image(img_path)
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()

        if tuple(image_tensor.shape[-2:]) != tuple(self.target_size):
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if label_path is not None:
            label_image = Image.open(label_path)
            mask_tensor = self.process_label(label_image)

            if tuple(mask_tensor.shape[-2:]) != tuple(self.target_size):
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0),
                    size=self.target_size,
                    mode="nearest",
                ).squeeze(0)
        else:
            mask_tensor = torch.zeros((1, *self.target_size), dtype=torch.float32)

        slice_location = None
        if "ImagePositionPatient" in dicom_data and len(dicom_data.ImagePositionPatient) >= 3:
            try:
                slice_location = float(dicom_data.ImagePositionPatient[2])
            except Exception:
                slice_location = None

        return {
            "image": image_tensor.contiguous(),
            "mask": mask_tensor.contiguous(),
            "patient_id": sample["patient_id"],
            "phase": sample["phase"] if sample["phase"] is not None else "none",
            "slice_number": int(sample["slice_number"]),
            "image_path": img_path,
            "label_path": label_path,
            "slice_location": slice_location,
        }
if __name__ == "__main__":
    dataset = SinusDataset(
        img_dir="DICOM",
        label_dir="PNG",
        target_size=(512, 512),
        require_labels=True,
        verbose=True
    )

    print("Total samples:", len(dataset))

    sample = dataset[0]
    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Patient ID:", sample["patient_id"])
    print("Phase:", sample["phase"])
    print("Slice number:", sample["slice_number"])
    print("Image path:", sample["image_path"])
    print("Label path:", sample["label_path"])