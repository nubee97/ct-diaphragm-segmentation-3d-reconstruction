# CT-Based Diaphragm Segmentation and 3D Reconstruction

Research-oriented deep learning pipeline for automatic diaphragm segmentation from CT DICOM images using paired PNG segmentation masks. The project supports DICOM preprocessing, U-Net-based model training, validation, prediction, mask export, overlay generation, checkpoint saving, and future 3D reconstruction for diaphragm motion analysis.

---

## Overview

This repository contains code for a medical image analysis pipeline focused on **CT-based diaphragm segmentation**. The system learns from paired DICOM CT slices and PNG segmentation masks, then predicts diaphragm regions from unseen CT images.

The long-term goal of this project is to support **diaphragm function evaluation** by combining deep learning-based segmentation with 3D reconstruction and respiratory-phase analysis.

This project is being developed as part of a research workflow intended for future IEEE-style publication.

---

## Research Motivation

Manual diaphragm segmentation from CT scans is time-consuming, labor-intensive, and can vary depending on the annotator. A reliable automated segmentation pipeline can help reduce manual workload, improve consistency, and support quantitative analysis of diaphragm structure and motion.

This project aims to contribute toward a complete workflow for:

- CT-based diaphragm segmentation
- diaphragm motion analysis
- 3D anatomical reconstruction
- respiratory-phase comparison
- medical AI research and clinical decision-support development

---

## Core Features

- DICOM CT image loading and preprocessing
- PNG segmentation mask loading
- Automatic DICOM-to-PNG mask matching
- 2D U-Net-based binary segmentation
- Train/validation workflow
- Dice score validation
- Best-model and last-model checkpoint saving
- Prediction on unseen DICOM images
- Predicted mask export as PNG
- Optional overlay visualization
- Optional 3D volume and mesh reconstruction
- Research-friendly structure for IEEE-style experiments

---

## Methodology

The current workflow follows this pipeline:

```text
CT DICOM slices
        ↓
Image preprocessing and normalization
        ↓
PNG segmentation mask matching
        ↓
2D U-Net-based segmentation model
        ↓
Binary diaphragm mask prediction
        ↓
Validation using Dice score
        ↓
Prediction mask export
        ↓
Optional overlay visualization
        ↓
Future 3D volume and mesh reconstruction
```

---

## Model Architecture

The project uses a U-Net-style segmentation model:

```python
from model import UNet

model = UNet(n_channels=1, n_classes=1)
```

### Model Type

The current model is designed for **2D binary medical image segmentation**.

- Input: single-channel CT DICOM slice
- Output: single-channel binary segmentation mask
- Task: diaphragm segmentation
- Loss function: binary segmentation loss such as `BCEWithLogitsLoss`
- Main validation metric: Dice coefficient

### Recommended Model Direction

The recommended model direction for this project is:

```text
2D Residual Attention U-Net with Group Normalization
```

This architecture is suitable for medical imaging because:

- CT images are grayscale, so the model uses one input channel.
- The task is binary segmentation, so the model outputs one mask channel.
- Residual blocks improve gradient flow during training.
- Attention gates help the decoder focus on relevant anatomical regions.
- GroupNorm is more stable than BatchNorm when training with small batch sizes.
- Dropout can help reduce overfitting on limited medical datasets.

The model outputs **raw logits**. Sigmoid activation should be applied during validation and prediction, not inside the model.

---

## Project Structure

```text
ct-diaphragm-segmentation-3d-reconstruction/
│
├── main.py                         # Main training and prediction script
├── model.py                        # U-Net model architecture
├── dataset.py                      # Dataset-related utilities
├── dataset_generation_batch.py     # Dataset generation/preprocessing script
├── requirements.txt                # Python dependencies
├── terminal_run.txt                # Example terminal commands/logs
├── README.md                       # Project documentation
│
├── DICOM/                          # Local CT DICOM data - not uploaded to GitHub
├── PNG/                            # Local PNG segmentation masks - not uploaded to GitHub
├── runs/                           # Training outputs/checkpoints - not uploaded to GitHub
├── dataset/                        # Local generated/intermediate data - not uploaded to GitHub
├── .venv/                          # Local virtual environment - not uploaded to GitHub
└── __pycache__/                    # Python cache files - not uploaded to GitHub
```

---

## File Descriptions

### `main.py`

Main script for training and prediction.

Responsibilities include:

- loading DICOM CT images
- loading PNG segmentation masks
- matching DICOM files to corresponding PNG labels
- preprocessing image data
- training the segmentation model
- validating model performance
- saving model checkpoints
- running prediction on DICOM images
- saving predicted masks
- generating optional overlay previews
- supporting future 3D reconstruction workflow

### `model.py`

Contains the U-Net model architecture.

The expected model class is:

```python
class UNet(nn.Module):
    ...
```

The training script expects:

```python
UNet(n_channels=1, n_classes=1)
```

### `dataset.py`

Contains dataset-related code and utilities from earlier versions of the project.

Depending on the active version of `main.py`, this file may be used as a supporting module or maintained as part of the research development history.

### `dataset_generation_batch.py`

Dataset generation or preprocessing script used to prepare training data, segmentation masks, overlays, or intermediate outputs.

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `terminal_run.txt`

Contains example terminal commands, experiment logs, or running notes.

### `DICOM/`

Local folder containing CT DICOM images.

This folder should not be uploaded to GitHub because DICOM files may contain sensitive patient metadata.

### `PNG/`

Local folder containing paired PNG segmentation masks.

This folder should not be uploaded to GitHub if the labels are derived from medical data.

### `runs/`

Local folder for training outputs, checkpoints, prediction outputs, and preview images.

This folder should not be uploaded to GitHub because it may contain large files and generated medical outputs.

---

## Expected Dataset Format

The code expects DICOM images and PNG masks to be organized by patient ID.

Example structure:

```text
DICOM/
  10051764/
    IM-0002-0372.dcm
    IM-0002-0373.dcm
    IM-0002-0374.dcm

PNG/
  10051764/
    372_axial.png
    373_axial.png
    374_axial.png
```

The matching logic is designed to pair files like:

```text
IM-0002-0372.dcm  →  372_axial.png
IM-0002-0373.dcm  →  373_axial.png
IM-0002-0374.dcm  →  374_axial.png
```

This allows the model to learn from DICOM image slices and their corresponding PNG segmentation labels.

---

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/nubee97/ct-diaphragm-segmentation-3d-reconstruction.git
cd ct-diaphragm-segmentation-3d-reconstruction
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

If needed, install the core dependencies manually:

```powershell
python -m pip install torch pydicom pillow numpy scikit-image matplotlib plotly open3d
```

---

## Training

### Quick 1-Epoch Test

Run this first to confirm that the DICOM and PNG folders are correctly matched and that the training pipeline works.

```powershell
python main.py train `
  --dicom-root "DICOM" `
  --mask-root "PNG" `
  --save-dir "runs/diaphragm_unet_test" `
  --epochs 1 `
  --batch-size 4 `
  --image-size 512 512 `
  --save-previews
```

### Full Training

```powershell
python main.py train `
  --dicom-root "DICOM" `
  --mask-root "PNG" `
  --save-dir "runs/diaphragm_unet_v1" `
  --epochs 50 `
  --batch-size 4 `
  --image-size 512 512 `
  --save-previews
```

---

## Prediction

After training, use the saved checkpoint for prediction.

```powershell
python main.py predict `
  --dicom-root "DICOM" `
  --checkpoint "runs/diaphragm_unet_v1/best_model.pth" `
  --output-dir "runs/diaphragm_predictions" `
  --image-size 512 512 `
  --save-overlays
```

Expected prediction outputs may include:

```text
predicted_masks/
overlays/
```

---

## Optional 3D Reconstruction

The long-term goal of this project includes 3D reconstruction from predicted 2D masks.

The intended 3D workflow is:

```text
Predicted 2D masks
        ↓
Stack masks by slice location
        ↓
Create 3D binary volume
        ↓
Generate surface mesh
        ↓
Export PLY mesh
        ↓
Visualize diaphragm surface in 3D
```

Potential outputs may include:

```text
predicted_volume.npy
generated_mesh.ply
mesh_preview.html
```

This part is intended to support future diaphragm motion and shape analysis.

---

## Current Training Status

Example successful data-loading output:

```text
Loaded 3586 sample(s) from DICOM
Skipped 1 DICOM file(s) with no matching PNG mask.
Device: cpu
Training samples: 2869
Validation samples: 717
```

This means the pipeline successfully matched almost all DICOM slices with their corresponding PNG masks.

One skipped DICOM slice is usually acceptable if the corresponding PNG label does not exist.

---

## Checkpoints and Outputs

During training, the script may save:

```text
best_model.pth
last_model.pth
previews/
```

### `best_model.pth`

The best-performing checkpoint based on validation performance.

### `last_model.pth`

The most recent checkpoint saved during training.

### `previews/`

Validation preview images showing:

- original CT image
- ground-truth PNG mask
- predicted mask

---

## Research Evaluation Plan

For an IEEE-style research paper, the project should be evaluated using patient-level experiments and multiple segmentation metrics.

Recommended evaluation metrics:

| Category | Metric |
|---|---|
| Segmentation overlap | Dice coefficient |
| Segmentation overlap | Intersection over Union |
| Pixel-level performance | Precision |
| Pixel-level performance | Recall |
| Boundary accuracy | Hausdorff Distance |
| Boundary accuracy | Average Surface Distance |
| 3D reconstruction | Mesh-to-mesh surface distance |
| Clinical motion analysis | Inspiratory/expiratory diaphragm displacement |

---

## Important Research Note: Patient-Level Split

For serious medical AI research, validation should be performed at the **patient level**, not only at the random slice level.

A random slice-level split can cause data leakage because slices from the same patient may appear in both training and validation sets.

Recommended final research split:

```text
Training patients:    70–80%
Validation patients:  10–15%
Test patients:        10–15%
```

The final test set should contain completely unseen patients.

---

## Planned Research Experiments

Recommended experiments for publication:

1. Baseline U-Net vs Residual U-Net vs Attention Residual U-Net.
2. Slice-level split vs patient-level split comparison.
3. Inspiratory vs expiratory phase segmentation performance.
4. 2D segmentation accuracy using Dice, IoU, precision, and recall.
5. Boundary accuracy using Hausdorff Distance and Average Surface Distance.
6. 3D reconstruction quality using surface distance metrics.
7. Diaphragm motion estimation between respiratory phases.
8. Runtime and memory usage comparison.
9. Qualitative visualization of segmentation overlays and 3D meshes.
10. External validation using data from another institution if available.

---

## Potential IEEE Paper Direction

Possible research title:

```text
Automated CT-Based Diaphragm Segmentation and 3D Reconstruction Using a Lightweight U-Net Architecture
```

Alternative title:

```text
Deep Learning-Based Diaphragm Motion Assessment from Inspiratory and Expiratory CT Using 2D Segmentation and 3D Surface Reconstruction
```

Possible contribution statement:

```text
This study proposes a deep learning-based pipeline for automatic diaphragm segmentation from CT images and subsequent 3D surface reconstruction. The method combines a U-Net-based segmentation model with DICOM-to-mask preprocessing, patient-level evaluation, and mesh-based visualization to support quantitative diaphragm function analysis.
```

---

## Roadmap

- [ ] Add patient-level train/validation/test split
- [ ] Add IoU, precision, and recall metrics
- [ ] Add Hausdorff Distance and Average Surface Distance
- [ ] Add CSV result export for every patient
- [ ] Add training resume support
- [ ] Add estimated training time and progress logging
- [ ] Add automatic checkpoint resume
- [ ] Add 3D mesh-to-mesh evaluation
- [ ] Add respiratory-phase diaphragm displacement measurement
- [ ] Add publication-ready result figures
- [ ] Add IEEE-style experiment tables
- [ ] Add external validation experiments

---

## Privacy and Data Notice

This repository should not contain real patient DICOM data, PNG labels, trained checkpoints, or generated medical outputs.

The following folders should remain local and should not be uploaded to GitHub:

```text
DICOM/
PNG/
dataset/
runs/
.venv/
__pycache__/
```

DICOM files may contain protected metadata and must be handled according to institutional and medical data privacy requirements.

---

## Recommended `.gitignore`

Before uploading code to GitHub, use a `.gitignore` file similar to this:

```gitignore
.venv/
__pycache__/
*.pyc

# Patient/private medical data
DICOM/
PNG/
dataset/
*.dcm
*.dicom
*.nii
*.nii.gz

# Training outputs and generated files
runs/
*.pth
*.pt
*.npy
*.npz
*.ply
*.html

# OS/editor files
.vscode/
.DS_Store
```

---

## Limitations

Current limitations to address before publication:

- Patient-level train/validation/test split should be implemented.
- Full test-set evaluation needs to be added.
- More metrics are required beyond Dice score.
- External validation would improve reliability.
- 3D mesh accuracy should be quantitatively evaluated.
- Clinical interpretation should be reviewed with domain experts.
- The current project is research code and not a clinical product.

---

## Disclaimer

This project is intended for research and development only. It is not a certified medical device and should not be used for clinical diagnosis without proper validation, regulatory approval, and expert medical review.

---

## Author

**Pascal Nnubia Nnamdi**  
Computer Vision & Medical AI Researcher  
PhD Researcher in Computer Engineering  
Focus: Medical Image Analysis, CT Segmentation, 3D Reconstruction, and AI-Based Diaphragm Function Evaluation

---

## Citation

Citation information will be updated if this project contributes to a publication.

```bibtex
@misc{nnubia_ct_diaphragm_segmentation,
  title  = {CT-Based Diaphragm Segmentation and 3D Reconstruction},
  author = {Nnubia, Pascal Nnamdi},
  year   = {2026},
  note   = {Research code for CT diaphragm segmentation and 3D reconstruction}
}
```

---

## License

License information should be added before public release.

Recommended options:

- MIT License for open research code
- Apache-2.0 License for more explicit patent protection
- Private repository if dataset or institutional restrictions apply
