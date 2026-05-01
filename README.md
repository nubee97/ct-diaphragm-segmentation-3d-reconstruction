# CT-Based Diaphragm Segmentation and 3D Reconstruction

Research-oriented deep learning pipeline for automatic diaphragm segmentation from CT DICOM images using paired PNG masks. The project supports U-Net-based training, validation, prediction, mask export, overlay visualization, and future 3D reconstruction for diaphragm motion analysis.

---

## Overview

This repository contains code for a medical image analysis pipeline focused on **CT-based diaphragm segmentation**. The system learns from paired DICOM slices and PNG segmentation labels, then predicts diaphragm masks on unseen CT images.

The long-term goal of this project is to support **diaphragm function evaluation** by combining deep learning segmentation with 3D reconstruction and respiratory-phase analysis.

---

## Research Motivation

Manual diaphragm segmentation from CT scans is time-consuming and can vary between annotators. An automated segmentation pipeline can help improve consistency, reduce manual workload, and support quantitative analysis of diaphragm structure and motion.

This project is being developed as part of a research workflow intended for future IEEE-style publication.

---

## Core Features

- DICOM CT image loading and preprocessing
ators. An automated segmentation pipeline can help improve consistency, reduce manual workload, and support quantitative analysis of diaphragm structure and motion.

This project is being developed- PNG mask loading and DICOM-to-mask matching
- U-Net-based binary segmentation
- Training and validation pipeline
- Dice score evaluation
- Best-model and last-model checkpoint saving
- Prediction on unseen DICOM images
- Predicted mask export as PNG
- Optional overlay visualization
- Research-ready structure for future evaluation metrics and 3D analysis

---

## Methodology

The current workflow follows this pipeline:

```text
DICOM CT slices
        ↓
Image normalization and preprocessing
        ↓
PNG segmentation mask matching
        ↓
2D U-Net training
        ↓
Binary diaphragm mask prediction
        ↓
Validation using Dice score
        ↓
Mask and overlay export
        ↓
Future 3D volume and mesh reconstruction
