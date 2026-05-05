# CT-Based Diaphragm Segmentation and 3D Reconstruction

**Automated diaphragm segmentation in thoracic CT for COPD assessment using a U-Net-based deep learning pipeline.**

This repository provides a research-oriented medical image analysis pipeline for automatic diaphragm segmentation from CT DICOM images using paired PNG segmentation masks. The project supports DICOM preprocessing, DICOM-to-mask matching, U-Net-based segmentation training, validation, prediction, mask export, overlay visualization, and downstream 3D reconstruction for diaphragm motion analysis.

> **Research context:** This project is part of an ongoing medical AI research workflow focused on diaphragm function evaluation in chronic pulmonary disease, especially chronic obstructive pulmonary disease (COPD).

---

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Research Objective](#research-objective)
- [Key Contributions](#key-contributions)
- [Current Code Features](#current-code-features)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Dataset and Data Format](#dataset-and-data-format)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Prediction](#prediction)
- [3D Reconstruction Workflow](#3d-reconstruction-workflow)
- [Research Results](#research-results)
- [Comparative Analysis](#comparative-analysis)
- [Privacy and Data Protection](#privacy-and-data-protection)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Author](#author)
- [Citation](#citation)
- [License](#license)

---

## Overview

The diaphragm is the primary respiratory muscle responsible for generating negative intrathoracic pressure during inhalation. In patients with COPD and other chronic pulmonary diseases, diaphragm structure and function can be affected by hyperinflation, reduced mobility, and mechanical inefficiency.

This project aims to support automated diaphragm assessment by combining:

- thoracic CT DICOM image processing,
- binary diaphragm mask generation,
- U-Net-based segmentation,
- respiratory phase analysis,
- 2D overlay visualization,
- and future 3D diaphragm reconstruction.

The long-term research goal is to reduce manual segmentation workload, improve reproducibility, and provide quantitative tools for diaphragm function evaluation.

---

## Research Motivation

Manual diaphragm segmentation in thoracic CT is difficult because the diaphragm is thin, low-contrast, anatomically variable, and closely connected to surrounding organs. Traditional semi-automatic methods can provide useful results, but they often require expert intervention for review, correction, and noise removal.

For large-scale COPD assessment, this creates three major problems:

1. **Time burden** — manual or semi-automatic segmentation is slow.
2. **Inter-observer variability** — results may differ depending on the annotator.
3. **Limited scalability** — large patient cohorts are difficult to process consistently.

A fully automated deep learning model can address these limitations by learning diaphragm boundaries directly from CT slices and producing consistent segmentation masks for downstream clinical and research analysis.

---

## Research Objective

The objective of this project is to develop a fully automated CT-based diaphragm segmentation pipeline that can:

- segment diaphragm regions from thoracic CT DICOM slices,
- support inhalation and exhalation phase analysis,
- reduce dependence on manual expert correction,
- generate reproducible segmentation outputs,
- provide quantitative metrics such as Dice score and IoU,
- support 3D diaphragm surface reconstruction,
- and contribute to COPD-focused respiratory function assessment.

---

## Key Contributions

This project is designed around two main research contributions:

1. **Diaphragm-labeled CT dataset generation**  
   A labeled dataset is generated from diaphragm mesh information by projecting mesh intersection points onto axial CT slices and converting them into binary PNG masks.

2. **Fully automated U-Net-based segmentation**  
   A U-Net-based model is trained to segment diaphragm regions from thoracic CT images, reducing the need for expert-guided semi-automatic segmentation.

Additional contributions include:

- DICOM-aware preprocessing and mask matching,
- respiratory phase comparison between inhalation and exhalation,
- segmentation-based 3D visualization,
- surface area comparison between generated and original diaphragm meshes,
- and a reproducible research code structure for medical AI experiments.

---

## Current Code Features

The current repository is structured as a Python-based research codebase for CT diaphragm segmentation.

Core implementation features include:

- DICOM CT image loading
- PNG segmentation mask loading
- Automatic matching between DICOM slice IDs and PNG mask IDs
- CT image normalization and preprocessing
- 2D binary U-Net segmentation
- Train/validation workflow
- Dice-based validation
- Best and last checkpoint saving
- Prediction on unseen DICOM folders
- Predicted mask export as PNG
- Optional overlay image generation
- Local training output management under `runs/`
- Privacy-safe repository structure that excludes medical data and checkpoints

The public repository should contain code and documentation only. Real DICOM files, masks, checkpoints, and generated medical outputs should remain local.

---

## Methodology

The project follows the workflow below:

```text
Thoracic CT DICOM slices
        ↓
DICOM loading and preprocessing
        ↓
PNG segmentation mask matching
        ↓
2D U-Net-based binary segmentation
        ↓
Validation using segmentation metrics
        ↓
Predicted diaphragm mask export
        ↓
Overlay visualization
        ↓
3D volume and surface reconstruction
        ↓
Respiratory phase analysis
```

The labeled dataset generation step is based on a semi-automatic mesh-to-slice process. A diaphragm mesh is intersected with axial CT planes, and the resulting points are projected onto corresponding 2D CT slices. These projected regions are converted into binary masks for supervised segmentation training.

---

## Model Architecture

The active model direction is based on a U-Net-style encoder-decoder architecture for 2D binary medical image segmentation.

```python
from model import UNet

model = UNet(n_channels=1, n_classes=1)
```

### Model Type

| Item | Description |
|---|---|
| Input | Single-channel CT DICOM slice |
| Output | Single-channel binary diaphragm mask |
| Task | Diaphragm segmentation |
| Architecture | 2D U-Net |
| Loss | Binary segmentation loss such as BCE / BCEWithLogits |
| Optimizer | Adam |
| Main metrics | Dice coefficient, IoU, precision, recall |

### U-Net Design

The U-Net model uses an encoder-decoder structure:

- The **encoder** extracts spatial and semantic features using convolution and downsampling.
- The **decoder** restores spatial resolution through upsampling.
- **Skip connections** preserve fine anatomical details.
- A final binary output layer predicts the diaphragm region.

For research extension, a stronger future model direction is:

```text
2D Residual Attention U-Net with Group Normalization
```

This direction is suitable for medical CT segmentation because residual connections improve gradient flow, attention gates help the decoder focus on relevant anatomy, and GroupNorm can be more stable than BatchNorm when batch sizes are small.

---

## Dataset and Data Format

The research dataset contains CT scans from three clinical institutions:

| Institution | Number of Patients |
|---|---:|
| Dongguk University | 110 |
| Yonsei University | 40 |
| Kangwon University | 25 |
| **Total** | **175** |

The dataset includes both inhalation and exhalation phases. Slice thickness and pixel spacing vary across scans, so preprocessing is required to standardize images before model training.

### Expected Local Folder Format

The code expects DICOM images and PNG masks to be organized by patient ID.

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

The matching logic pairs DICOM slice files with corresponding PNG masks:

```text
IM-0002-0372.dcm  →  372_axial.png
IM-0002-0373.dcm  →  373_axial.png
IM-0002-0374.dcm  →  374_axial.png
```

This allows the model to learn from paired CT slices and binary diaphragm labels.

---

## Project Structure

```text
ct-diaphragm-segmentation-3d-reconstruction/
│
├── main.py                         # Main CLI script for training and prediction
├── model.py                        # U-Net model architecture
├── dataset.py                      # Dataset utilities and earlier dataset logic
├── dataset_generation_batch.py     # Mesh/DICOM-based dataset generation workflow
├── requirements.txt                # Python dependencies
├── terminal_run.txt                # Example commands and experiment notes
├── README.md                       # Project documentation
│
├── DICOM/                          # Local CT DICOM data - not uploaded
├── PNG/                            # Local PNG segmentation masks - not uploaded
├── PNG_clean/                      # Optional cleaned masks - not uploaded
├── PNG_clean_dilated/              # Optional cleaned/dilated masks - not uploaded
├── runs/                           # Training outputs/checkpoints - not uploaded
├── dataset/                        # Local generated/intermediate data - not uploaded
├── .venv/                          # Local virtual environment - not uploaded
└── __pycache__/                    # Python cache files - not uploaded
```

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

If needed, install core dependencies manually:

```powershell
python -m pip install torch torchvision pydicom pillow numpy scikit-image matplotlib plotly open3d trimesh opencv-python
```

---

## Training

### Quick 1-Epoch Test

Use this command first to confirm that DICOM images and PNG masks are correctly matched.

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

### Full Training Example

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

### Expected Training Outputs

```text
runs/diaphragm_unet_v1/
  best_model.pth
  last_model.pth
  previews/
  training_log.csv
```

`best_model.pth` stores the best-performing model based on validation performance.  
`last_model.pth` stores the latest model state from the final epoch.

---

## Prediction

After training, use the saved checkpoint to predict diaphragm masks from unseen DICOM images.

```powershell
python main.py predict `
  --dicom-root "DICOM" `
  --checkpoint "runs/diaphragm_unet_v1/best_model.pth" `
  --output-dir "runs/diaphragm_predictions" `
  --image-size 512 512 `
  --save-overlays
```

Expected prediction outputs:

```text
runs/diaphragm_predictions/
  predicted_masks/
  overlays/
```

---

## 3D Reconstruction Workflow

The long-term goal of this repository includes 3D reconstruction from predicted 2D diaphragm masks.

```text
Predicted 2D masks
        ↓
Stack masks by slice position
        ↓
Create 3D binary volume
        ↓
Generate diaphragm surface mesh
        ↓
Export PLY / HTML visualization
        ↓
Measure respiratory-phase surface differences
```

Potential outputs:

```text
predicted_volume.npy
generated_mesh.ply
mesh_preview.html
surface_area_results.csv
```

This 3D reconstruction stage supports diaphragm surface analysis, exhalation/inhalation comparison, and COPD-related respiratory function evaluation.

---

## Research Results

The manuscript reports experiments on thoracic CT scans from **175 patients** across three institutions. For the validation experiment, the model was trained on **100 patients** and validated on **75 patients**.

### Main Validation Metrics

| Metric | Result |
|---|---:|
| Validation Loss | 0.0016 |
| Mean IoU | 0.9126 |
| Dice Coefficient | 0.9272 |

These results indicate strong segmentation overlap between predicted diaphragm masks and the reference labels.

### Individual-Patient Stability

The reported validation loss ranges from **0.0011 to 0.0019**, while Dice coefficients range from **0.9231 to 0.9280** across individual patients. This suggests stable segmentation performance across anatomical variation.

---

## Comparative Analysis

The U-Net-based model was compared against a previous semi-automatic segmentation approach.

| Method | Dice Coefficient |
|---|---:|
| Semi-automatic segmentation | 0.8400 |
| Proposed U-Net model | 0.9272 |

The U-Net model showed improved consistency in diaphragm segmentation and stronger sensitivity to respiratory phase changes.

### Surface Area Comparison

The manuscript also compares generated diaphragm surfaces with original diaphragm meshes for selected cases.

| Case | Original Surface Area (mm²) | Generated Surface Area (mm²) | Difference (mm²) | Percentage Difference |
|---|---:|---:|---:|---:|
| A | 4540.65 | 4955.28 | 414.63 | 9.13% |
| B | 5208.46 | 6427.32 | 1218.86 | 23.39% |
| C | 5260.51 | 5616.73 | 356.22 | 6.77% |

Cases A and C show close alignment between the generated and original diaphragm surfaces, while Case B indicates that complex anatomical regions may still require further model refinement.

---

## Research Interpretation

The results suggest that U-Net-based segmentation can reduce manual intervention while maintaining strong quantitative performance. Compared with the semi-automatic method, the proposed approach provides:

- higher Dice-based segmentation accuracy,
- improved reproducibility,
- better scalability for larger patient cohorts,
- stronger consistency across respiratory phases,
- and better support for downstream 3D diaphragm analysis.

The model also supports COPD-focused respiratory assessment by enabling quantitative comparison between inhalation and exhalation diaphragm surfaces.

---

## Privacy and Data Protection

This repository should not include real patient DICOM files, PNG masks, trained checkpoints, or generated medical outputs.

The following folders and file types should remain local:

```text
DICOM/
PNG/
PNG_clean/
PNG_clean_dilated/
dataset/
runs/
.venv/
__pycache__/
*.dcm
*.pth
*.pt
*.npy
*.nii
*.nii.gz
```

DICOM files may contain sensitive patient metadata and must be handled according to institutional review board approval, hospital policy, and medical data privacy requirements.

---

## Recommended `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc

# Patient/private medical data
DICOM/
PNG/
PNG_clean/
PNG_clean_dilated/
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

## Roadmap

- [ ] Push full source code to the public repository
- [ ] Add patient-level train/validation/test split
- [ ] Add test-set evaluation on completely unseen patients
- [ ] Add IoU, precision, recall, and F1-score export
- [ ] Add Hausdorff Distance and Average Surface Distance
- [ ] Add CSV export for patient-level metrics
- [ ] Add automatic checkpoint resume
- [ ] Add training time estimation and progress logging
- [ ] Add 3D mesh reconstruction from predicted masks
- [ ] Add surface area comparison between inhalation and exhalation
- [ ] Add Z-value respiratory phase analysis
- [ ] Add publication-ready result figures
- [ ] Add external validation from another institution

---

## Limitations

Current limitations include:

- The public repository should not include the private medical dataset.
- Model performance depends on the quality of generated diaphragm masks.
- Complex diaphragm anatomy may still produce larger surface-area deviations.
- External validation is required before broader clinical claims.
- The project is research code and is not a certified clinical tool.
- The method should not be used for diagnosis without regulatory approval and expert medical validation.

---

## Disclaimer

This project is intended for research and development only. It is not a certified medical device and should not be used for clinical diagnosis, treatment planning, or patient management without proper validation, regulatory approval, and clinical expert review.

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

- **MIT License** for open research code
- **Apache-2.0 License** for explicit patent protection
- **Private repository** if institutional or dataset restrictions apply
