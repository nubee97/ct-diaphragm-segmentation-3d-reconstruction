"""
Batch CT/DICOM + diaphragm-mesh dataset generator.

This script converts CT DICOM volumes + diaphragm mesh intersections into
training-ready PNG pairs:

    output_root/
        images/   patient_phase_slice.png   # clean grayscale CT slice
        masks/    patient_phase_slice.png   # binary mask from mesh intersection
        overlays/ patient_phase_slice.png   # optional visual QA overlay
        html/     patient_phase.html        # optional 3D Plotly QA file
        manifest.csv
        errors.log
        summary.json

It is designed to process many patients/phases safely. If one patient fails,
the error is logged and the script continues with the next patient.

Expected input layout, phase-based:

    images_root/
        PATIENT_ID/
            in/   DICOM files...
            ex/   DICOM files...

    results_root/
        PATIENT_ID/
            in/   11.lung_diaphragm_contact_surface_mesh(manually).ply
            ex/   11.lung_diaphragm_contact_surface_mesh(manually).ply

Also supported, no phase folder:

    images_root/PATIENT_ID/DICOM files...
    results_root/PATIENT_ID/11.lung_diaphragm_contact_surface_mesh(manually).ply

Example:

    python dataset_generation_batch.py \
        --images-root "C:/Users/jiseo/Downloads/ct-project/images" \
        --results-root "C:/Users/jiseo/Downloads/ct-project/result" \
        --output-root "C:/Users/jiseo/Downloads/training_dataset" \
        --phases in ex \
        --save-overlays

Notes:
- The generated mask is a diaphragm-intersection line/point mask, not a filled
  organ segmentation mask.
- Default coordinate mapping follows the original script's logic: mesh X/Y are
  converted using DICOM PixelSpacing, then rotated into image pixel coordinates.
- Use --no-reverse-slices if your saved CT image and mask appear vertically
  mismatched compared with the original visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import trimesh
from PIL import Image, ImageDraw

try:
    import plotly.graph_objects as go
except Exception:  # Plotly is only needed when --save-html is used.
    go = None


DEFAULT_MESH_NAME = "11.lung_diaphragm_contact_surface_mesh(manually).ply"
DEFAULT_LUNG_WINDOW = (-1000, 400)


@dataclass
class ProcessResult:
    patient_id: str
    phase: str
    status: str
    dicom_dir: str
    mesh_path: str
    slices_total: int = 0
    slices_saved: int = 0
    masks_saved: int = 0
    overlays_saved: int = 0
    intersections_found: int = 0
    error: str = ""


# -----------------------------------------------------------------------------
# DICOM loading / image conversion
# -----------------------------------------------------------------------------


def is_probably_dicom(path: Path) -> bool:
    """Quick filename filter. Real validation happens with pydicom.dcmread."""
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".txt", ".csv", ".json", ".ply", ".obj"}:
        return False
    return True


def read_dicom_file(path: Path):
    """Read one DICOM file safely. Returns None when the file is invalid."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        # Accessing pixel_array will fail later if PixelData is missing, so filter early.
        if not hasattr(ds, "PixelData"):
            return None
        return ds
    except Exception:
        return None


def load_scan(directory: Path, recursive: bool = False) -> List[pydicom.dataset.FileDataset]:
    """Load and sort DICOM slices from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"DICOM directory does not exist: {directory}")

    candidates = directory.rglob("*") if recursive else directory.iterdir()
    slices = []
    for path in candidates:
        if not is_probably_dicom(path):
            continue
        ds = read_dicom_file(path)
        if ds is not None:
            slices.append(ds)

    if not slices:
        raise ValueError(f"No readable DICOM slices found in: {directory}")

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        if hasattr(ds, "SliceLocation"):
            try:
                return float(ds.SliceLocation)
            except Exception:
                pass
        if hasattr(ds, "InstanceNumber"):
            try:
                return float(ds.InstanceNumber)
            except Exception:
                pass
        return 0.0

    slices.sort(key=sort_key)
    return slices


def get_pixels_hu(slices: Sequence[pydicom.dataset.FileDataset]) -> np.ndarray:
    """Convert DICOM slices to a 3D HU volume."""
    pixel_arrays = []
    for idx, ds in enumerate(slices):
        try:
            arr = ds.pixel_array.astype(np.int16)
        except Exception as exc:
            raise ValueError(f"Could not read pixel_array for DICOM slice {idx}: {exc}") from exc
        pixel_arrays.append(arr)

    image = np.stack(pixel_arrays).astype(np.int16)
    image[image == -2000] = 0

    for slice_number, ds in enumerate(slices):
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slope = float(getattr(ds, "RescaleSlope", 1))

        if slope != 1:
            image[slice_number] = (slope * image[slice_number].astype(np.float64)).astype(np.int16)
        image[slice_number] = image[slice_number] + np.int16(intercept)

    return image.astype(np.int16)


def get_pixel_spacing(ds) -> Tuple[float, float]:
    spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    return float(spacing[0]), float(spacing[1])


def get_slice_thickness(slices: Sequence[pydicom.dataset.FileDataset]) -> float:
    first = slices[0]
    if hasattr(first, "SliceThickness"):
        return float(first.SliceThickness)

    # Fallback: estimate from adjacent ImagePositionPatient values.
    z_values = []
    for ds in slices:
        if hasattr(ds, "ImagePositionPatient"):
            try:
                z_values.append(float(ds.ImagePositionPatient[2]))
            except Exception:
                continue
    if len(z_values) >= 2:
        diffs = np.diff(sorted(z_values))
        diffs = diffs[np.abs(diffs) > 1e-6]
        if len(diffs) > 0:
            return float(np.median(np.abs(diffs)))
    return 1.0


def normalize_hu_to_uint8(image_2d: np.ndarray, window_min: int, window_max: int) -> np.ndarray:
    """Window HU values and convert to uint8 PNG."""
    clipped = np.clip(image_2d, window_min, window_max).astype(np.float32)
    normalized = (clipped - window_min) / max(float(window_max - window_min), 1.0)
    return (normalized * 255.0).astype(np.uint8)


# -----------------------------------------------------------------------------
# Mesh intersection
# -----------------------------------------------------------------------------


def line_plane_intersection(
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
    line_points: Sequence[Sequence[float]],
) -> Optional[np.ndarray]:
    """Fallback line/plane intersection for one triangle edge."""
    p0, n = np.array(plane_point, dtype=float), np.array(plane_normal, dtype=float)
    p1, p2 = np.array(line_points[0], dtype=float), np.array(line_points[1], dtype=float)
    line_vec = p2 - p1
    denominator = np.dot(line_vec, n)
    if np.isclose(denominator, 0):
        return None
    t = np.dot((p0 - p1), n) / denominator
    if 0 <= t <= 1:
        return p1 + t * line_vec
    return None


def mesh_plane_intersection_fallback(
    mesh: trimesh.Trimesh,
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
) -> np.ndarray:
    """Original-style edge-by-edge mesh/plane intersection fallback."""
    intersection_points = []
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    for triangle in vertices[faces]:
        for i in range(3):
            point1, point2 = triangle[i], triangle[(i + 1) % 3]
            intersect_point = line_plane_intersection(plane_point, plane_normal, [point1, point2])
            if intersect_point is not None:
                intersection_points.append(intersect_point)

    if not intersection_points:
        return np.empty((0, 3), dtype=float)
    return unique_points(np.asarray(intersection_points, dtype=float))


def unique_points(points: np.ndarray, decimals: int = 4) -> np.ndarray:
    """Remove duplicate or near-duplicate 3D points."""
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=float)
    rounded = np.round(points.astype(float), decimals=decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx]


def mesh_plane_intersection(
    mesh: trimesh.Trimesh,
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
) -> np.ndarray:
    """Return 3D points where a mesh intersects a plane."""
    plane_point_np = np.asarray(plane_point, dtype=float)
    plane_normal_np = np.asarray(plane_normal, dtype=float)

    try:
        # trimesh returns line segments with shape (N, 2, 3).
        segments = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=plane_normal_np,
            plane_origin=plane_point_np,
        )
        if segments is None or len(segments) == 0:
            return np.empty((0, 3), dtype=float)
        points = np.asarray(segments, dtype=float).reshape(-1, 3)
        return unique_points(points)
    except Exception:
        return mesh_plane_intersection_fallback(mesh, plane_point_np, plane_normal_np)


# -----------------------------------------------------------------------------
# Mask / overlay generation
# -----------------------------------------------------------------------------


def intersection_points_to_pixels(
    intersection_points: np.ndarray,
    image_shape: Tuple[int, int],
    pixel_spacing: Tuple[float, float],
    rotate_like_original: bool = True,
) -> np.ndarray:
    """
    Convert 3D mesh intersection points to 2D image pixel coordinates.

    This follows the mapping used in the original script:
      x_mm / PixelSpacing[0], y_mm / PixelSpacing[1], then rotate 90 degrees.
    """
    if intersection_points is None or len(intersection_points) == 0:
        return np.empty((0, 2), dtype=int)

    height, width = image_shape
    spacing_x, spacing_y = pixel_spacing
    normalized = np.asarray(intersection_points, dtype=float).copy()
    normalized[:, 0] = normalized[:, 0] / max(spacing_x, 1e-8)
    normalized[:, 1] = normalized[:, 1] / max(spacing_y, 1e-8)

    if rotate_like_original:
        pixels = np.zeros((normalized.shape[0], 2), dtype=float)
        pixels[:, 0] = normalized[:, 1]                  # image column / x
        pixels[:, 1] = height - normalized[:, 0]          # image row / y
    else:
        pixels = np.zeros((normalized.shape[0], 2), dtype=float)
        pixels[:, 0] = normalized[:, 0]
        pixels[:, 1] = normalized[:, 1]

    # Keep only points inside the image.
    valid = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[valid]

    if len(pixels) == 0:
        return np.empty((0, 2), dtype=int)
    return np.round(pixels).astype(int)


def sort_points_for_polyline(points_xy: np.ndarray) -> np.ndarray:
    """Sort 2D points around their centroid for a reasonable QA polyline."""
    if points_xy is None or len(points_xy) < 3:
        return points_xy
    points = points_xy.astype(float)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points_xy[order]


def draw_mask(
    image_shape: Tuple[int, int],
    points_xy: np.ndarray,
    thickness: int = 3,
    draw_lines: bool = False,
) -> Image.Image:
    """Create a binary mask from 2D intersection pixels."""
    height, width = image_shape
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if points_xy is None or len(points_xy) == 0:
        return mask

    radius = max(int(thickness), 1)
    points = [(int(x), int(y)) for x, y in points_xy]

    if draw_lines and len(points) >= 2:
        ordered = sort_points_for_polyline(points_xy)
        line_points = [(int(x), int(y)) for x, y in ordered]
        draw.line(line_points, fill=255, width=max(radius * 2, 1), joint="curve")

    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    return mask


def draw_overlay(
    image_uint8: np.ndarray,
    points_xy: np.ndarray,
    thickness: int = 3,
    draw_lines: bool = False,
) -> Image.Image:
    """Create an RGB QA image with mask/intersection shown in red."""
    overlay = Image.fromarray(image_uint8).convert("RGB")
    draw = ImageDraw.Draw(overlay)

    if points_xy is None or len(points_xy) == 0:
        return overlay

    radius = max(int(thickness), 1)
    points = [(int(x), int(y)) for x, y in points_xy]

    if draw_lines and len(points) >= 2:
        ordered = sort_points_for_polyline(points_xy)
        line_points = [(int(x), int(y)) for x, y in ordered]
        draw.line(line_points, fill=(255, 0, 0), width=max(radius * 2, 1), joint="curve")

    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))

    return overlay


# -----------------------------------------------------------------------------
# Optional 3D QA visualization
# -----------------------------------------------------------------------------


def save_3d_html(
    patient_id: str,
    phase: str,
    image_3d: np.ndarray,
    all_intersection_points: List[np.ndarray],
    slice_thickness: float,
    output_path: Path,
    slice_step: int = 20,
):
    """Save one lightweight 3D QA HTML file per patient/phase."""
    if go is None:
        raise RuntimeError("Plotly is not installed. Install plotly or run without --save-html.")

    if image_3d.size == 0 or len(image_3d.shape) != 3:
        raise ValueError("Invalid image_3d data. Expected a non-empty 3D numpy array.")

    fig = go.Figure()
    slice_step = max(int(slice_step), 1)

    for slice_index in range(0, image_3d.shape[0], slice_step):
        z_coord = slice_index * slice_thickness
        axial_slice = image_3d[slice_index, :, :]
        x, y = np.meshgrid(np.arange(axial_slice.shape[1]), np.arange(axial_slice.shape[0]))
        z = np.ones(axial_slice.shape) * z_coord

        fig.add_trace(go.Surface(
            z=z,
            x=x,
            y=y,
            surfacecolor=axial_slice,
            colorscale="gray",
            opacity=0.35,
            showscale=False,
            name=f"Slice {slice_index}",
        ))

    valid_points = [p for p in all_intersection_points if p is not None and len(p) > 0]
    if valid_points:
        points = np.vstack(valid_points)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=3, color="red"),
            name="Intersection Points",
        ))

    fig.update_layout(
        title=f"{patient_id} {phase} - Diaphragm Mesh / Axial Slice Intersections",
        scene=dict(
            xaxis=dict(title="X", showgrid=False),
            yaxis=dict(title="Y", showgrid=False),
            zaxis=dict(title="Z", showgrid=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------


def directory_contains_dicom(directory: Path, recursive: bool = False) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for path in iterator:
        if is_probably_dicom(path) and read_dicom_file(path) is not None:
            return True
    return False


def find_mesh_path(results_root: Path, patient_id: str, phase: str, mesh_name: str) -> Optional[Path]:
    """Find mesh path using phase-based and fallback no-phase layouts."""
    candidates = []
    if phase and phase != "root":
        candidates.append(results_root / patient_id / phase / mesh_name)
    candidates.append(results_root / patient_id / mesh_name)

    # Flexible fallback: any matching mesh under patient result folder.
    patient_result_dir = results_root / patient_id
    if patient_result_dir.exists():
        candidates.extend(patient_result_dir.rglob(mesh_name))

    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def discover_jobs(
    images_root: Path,
    results_root: Path,
    phases: Sequence[str],
    mesh_name: str,
    patient_ids: Optional[Sequence[str]] = None,
    recursive_dicom: bool = False,
) -> List[Tuple[str, str, Path, Path]]:
    """
    Discover patient/phase jobs.

    Returns tuples: (patient_id, phase, dicom_dir, mesh_path)
    """
    images_root = Path(images_root)
    results_root = Path(results_root)

    if patient_ids:
        patient_dirs = [images_root / pid for pid in patient_ids]
    else:
        patient_dirs = [p for p in images_root.iterdir() if p.is_dir()]

    jobs = []
    for patient_dir in sorted(patient_dirs, key=lambda p: p.name):
        patient_id = patient_dir.name
        if not patient_dir.exists():
            logging.warning("Patient folder missing: %s", patient_dir)
            continue

        phase_jobs_added = False
        for phase in phases:
            phase_dir = patient_dir / phase
            if not phase_dir.exists() or not phase_dir.is_dir():
                continue
            if not directory_contains_dicom(phase_dir, recursive=recursive_dicom):
                continue
            mesh_path = find_mesh_path(results_root, patient_id, phase, mesh_name)
            if mesh_path is None:
                logging.warning("Skipping %s/%s: mesh not found", patient_id, phase)
                continue
            jobs.append((patient_id, phase, phase_dir, mesh_path))
            phase_jobs_added = True

        # Support patient folders where DICOM files are directly under the patient folder.
        if not phase_jobs_added and directory_contains_dicom(patient_dir, recursive=False):
            phase = "root"
            mesh_path = find_mesh_path(results_root, patient_id, phase, mesh_name)
            if mesh_path is None:
                logging.warning("Skipping %s/root: mesh not found", patient_id)
                continue
            jobs.append((patient_id, phase, patient_dir, mesh_path))

    return jobs


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------


def safe_filename(patient_id: str, phase: str, slice_index: int) -> str:
    phase_part = phase if phase else "root"
    return f"{patient_id}_{phase_part}_{slice_index:04d}.png"


def process_patient_phase(
    patient_id: str,
    phase: str,
    dicom_dir: Path,
    mesh_path: Path,
    output_root: Path,
    args,
) -> ProcessResult:
    result = ProcessResult(
        patient_id=patient_id,
        phase=phase,
        status="started",
        dicom_dir=str(dicom_dir),
        mesh_path=str(mesh_path),
    )

    slices = load_scan(dicom_dir, recursive=args.recursive_dicom)
    image_3d = get_pixels_hu(slices)
    result.slices_total = int(image_3d.shape[0])

    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Invalid or empty mesh: {mesh_path}")

    slice_thickness = get_slice_thickness(slices)
    pixel_spacing = get_pixel_spacing(slices[0])

    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    overlays_dir = output_root / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    all_intersections_for_html = []

    start_slice = max(args.start_slice, 0)
    end_slice = min(args.end_slice if args.end_slice is not None else image_3d.shape[0], image_3d.shape[0])

    for slice_index in range(start_slice, end_slice):
        plane_normal = [0, 0, 1]
        plane_point = [0, 0, slice_index * slice_thickness]

        intersection_points = mesh_plane_intersection(mesh, plane_point, plane_normal)
        if len(intersection_points) > 0:
            result.intersections_found += 1
        all_intersections_for_html.append(intersection_points)

        if args.skip_empty and len(intersection_points) == 0:
            continue

        image_index = image_3d.shape[0] - 1 - slice_index if args.reverse_slices else slice_index
        if image_index < 0 or image_index >= image_3d.shape[0]:
            continue

        image_uint8 = normalize_hu_to_uint8(image_3d[image_index], args.window_min, args.window_max)
        points_xy = intersection_points_to_pixels(
            intersection_points,
            image_shape=image_uint8.shape,
            pixel_spacing=pixel_spacing,
            rotate_like_original=args.rotate_like_original,
        )

        filename = safe_filename(patient_id, phase, slice_index)
        Image.fromarray(image_uint8).save(images_dir / filename)
        result.slices_saved += 1

        mask = draw_mask(
            image_shape=image_uint8.shape,
            points_xy=points_xy,
            thickness=args.mask_thickness,
            draw_lines=args.draw_lines,
        )
        mask.save(masks_dir / filename)
        result.masks_saved += 1

        if args.save_overlays:
            overlay = draw_overlay(
                image_uint8=image_uint8,
                points_xy=points_xy,
                thickness=args.mask_thickness,
                draw_lines=args.draw_lines,
            )
            overlay.save(overlays_dir / filename)
            result.overlays_saved += 1

    if args.save_html:
        html_path = output_root / "html" / f"{patient_id}_{phase}.html"
        save_3d_html(
            patient_id=patient_id,
            phase=phase,
            image_3d=image_3d,
            all_intersection_points=all_intersections_for_html,
            slice_thickness=slice_thickness,
            output_path=html_path,
            slice_step=args.html_slice_step,
        )

    result.status = "success"
    return result


def write_manifest(output_root: Path, results: Sequence[ProcessResult]):
    manifest_path = output_root / "manifest.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else list(ProcessResult("", "", "", "", "").__dict__.keys()))
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))

    summary = {
        "patients_attempted": len(results),
        "success_count": sum(1 for r in results if r.status == "success"),
        "failed_count": sum(1 for r in results if r.status != "success"),
        "total_images_saved": sum(r.slices_saved for r in results),
        "total_masks_saved": sum(r.masks_saved for r in results),
        "total_overlays_saved": sum(r.overlays_saved for r in results),
        "total_slices_with_intersections": sum(r.intersections_found for r in results),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def configure_logging(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "errors.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training PNG images/masks from CT DICOM + diaphragm mesh intersections.")

    parser.add_argument("--images-root", required=True, type=Path, help="Root folder containing patient DICOM folders.")
    parser.add_argument("--results-root", required=True, type=Path, help="Root folder containing patient mesh/result folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Folder where generated dataset will be saved.")

    parser.add_argument("--phases", nargs="+", default=["in", "ex"], help="Respiration phase folders to process. Default: in ex")
    parser.add_argument("--patient-ids", nargs="+", default=None, help="Optional list of patient IDs to process. Default: all patient folders.")
    parser.add_argument("--mesh-name", default=DEFAULT_MESH_NAME, help="Mesh filename to load from each result folder.")

    parser.add_argument("--start-slice", type=int, default=0, help="First axial slice index to process. Default: 0")
    parser.add_argument("--end-slice", type=int, default=None, help="Exclusive end axial slice index. Default: last slice")
    parser.add_argument("--skip-empty", action="store_true", help="Save only slices that have mesh intersections.")
    parser.add_argument("--recursive-dicom", action="store_true", help="Search DICOM files recursively inside each DICOM folder.")

    parser.add_argument("--window-min", type=int, default=DEFAULT_LUNG_WINDOW[0], help="CT window minimum HU. Default: -1000")
    parser.add_argument("--window-max", type=int, default=DEFAULT_LUNG_WINDOW[1], help="CT window maximum HU. Default: 400")

    parser.add_argument("--mask-thickness", type=int, default=3, help="Radius/line thickness for mask drawing. Default: 3")
    parser.add_argument("--draw-lines", action="store_true", help="Connect intersection points into a thicker polyline mask/overlay.")
    parser.add_argument("--save-overlays", action="store_true", help="Save QA overlay PNGs with red intersection points.")
    parser.add_argument("--save-html", action="store_true", help="Save optional 3D Plotly QA HTML per patient/phase.")
    parser.add_argument("--html-slice-step", type=int, default=20, help="Step size for CT slices shown in 3D HTML. Default: 20")

    parser.add_argument("--reverse-slices", dest="reverse_slices", action="store_true", default=True, help="Use original script's reversed axial image index. Default: on")
    parser.add_argument("--no-reverse-slices", dest="reverse_slices", action="store_false", help="Disable reversed axial image indexing.")
    parser.add_argument("--rotate-like-original", dest="rotate_like_original", action="store_true", default=True, help="Use original script's 90-degree point rotation. Default: on")
    parser.add_argument("--no-rotate-like-original", dest="rotate_like_original", action="store_false", help="Disable original point rotation.")

    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    configure_logging(output_root)

    logging.info("Starting batch dataset generation")
    logging.info("Images root: %s", args.images_root)
    logging.info("Results root: %s", args.results_root)
    logging.info("Output root: %s", output_root)

    jobs = discover_jobs(
        images_root=args.images_root,
        results_root=args.results_root,
        phases=args.phases,
        mesh_name=args.mesh_name,
        patient_ids=args.patient_ids,
        recursive_dicom=args.recursive_dicom,
    )

    if not jobs:
        logging.error("No valid patient/phase jobs found. Check your folder paths and mesh filename.")
        write_manifest(output_root, [])
        return

    logging.info("Found %d patient/phase job(s)", len(jobs))

    results: List[ProcessResult] = []
    for job_index, (patient_id, phase, dicom_dir, mesh_path) in enumerate(jobs, start=1):
        logging.info("[%d/%d] Processing patient=%s phase=%s", job_index, len(jobs), patient_id, phase)
        try:
            result = process_patient_phase(
                patient_id=patient_id,
                phase=phase,
                dicom_dir=dicom_dir,
                mesh_path=mesh_path,
                output_root=output_root,
                args=args,
            )
            logging.info(
                "Success patient=%s phase=%s | images=%d masks=%d intersections=%d",
                patient_id,
                phase,
                result.slices_saved,
                result.masks_saved,
                result.intersections_found,
            )
            results.append(result)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            logging.error("Failed patient=%s phase=%s | %s", patient_id, phase, error_message)
            logging.debug(traceback.format_exc())
            results.append(ProcessResult(
                patient_id=patient_id,
                phase=phase,
                status="failed",
                dicom_dir=str(dicom_dir),
                mesh_path=str(mesh_path),
                error=error_message,
            ))
            continue

    write_manifest(output_root, results)
    logging.info("Finished. Manifest saved to: %s", output_root / "manifest.csv")
    logging.info("Summary saved to: %s", output_root / "summary.json")


if __name__ == "__main__":
    main()
