# Artificial Vision Final Project  
## Vegetation / Non-Vegetation Classification from Multi-Band Satellite Imagery Using CNN and Semi-Supervised Learning

---

# 1. Project Overview

This project develops a computer vision pipeline for classifying vegetation and non-vegetation regions from multi-band satellite imagery.

The work integrates:

- Atmospheric drought indicator analysis using precipitation data
- Multi-band satellite image preprocessing
- Convolutional Neural Network (CNN) based image classification
- Semi-supervised pseudo-labeling for leveraging unlabeled data

The goal is to build an automated framework capable of identifying vegetation coverage from satellite imagery while exploring the impact of pseudo-labeling on classification performance.

---

# 2. Problem Statement

Large-scale satellite datasets often contain:

- A small amount of labeled imagery
- A large amount of unlabeled imagery
- High computational cost for manual annotation

This project addresses:

> How can we classify vegetation from multi-band satellite imagery using limited labeled data while exploiting a large pool of unlabeled images?

---

# 3. Dataset Description

## Satellite Imagery

- Source: Sentinel-2 Multi-Band TIFF Satellite Images
- Bands Used:
  - Red
  - Green
  - Blue
  - Near Infrared (NIR)

## Dataset Composition

| Data Type | Count |
|----------|------|
| Total TIFF Images | 17,161 |
| Labeled Samples | 186 |
| Unlabeled Samples | 16,975 |

---

# 4. Methodology

---

## 4.1 Atmospheric Data Processing

Precipitation data was processed to compute:

- Standardized Precipitation Index (SPI)

Purpose:

- Integrate drought-related atmospheric analysis into vegetation monitoring framework.

---

## 4.2 Image Preprocessing Pipeline

### Step 1: TIFF Loading

Loaded 4-band GeoTIFF satellite imagery.

---

### Step 2: Tile Generation

Large TIFF images were split into:

- 128×128 / 256×256 tiles

Purpose:

- Make images suitable for CNN training.

---

### Step 3: False/RGB Color Conversion

Converted multi-band imagery into visualizable RGB / False-color composites.

Band Mapping Used:

- Red   → Band 4
- Green → Band 3
- Blue  → Band 2
- NIR   → Band 8

---

## 4.3 Manual Labeling

A subset of image tiles was manually labeled into:

- Vegetation
- Non-Vegetation

---

## 4.4 CNN Model Development

A custom CNN architecture was implemented using TensorFlow/Keras.

### Architecture Summary

- Conv2D (32 Filters)
- MaxPooling
- Conv2D (64 Filters)
- MaxPooling
- Conv2D (128 Filters)
- MaxPooling
- Flatten
- Dense Layers
- Sigmoid Output Layer

Loss Function:

- Binary Crossentropy

Optimizer:

- Adam

Evaluation Metric:

- Accuracy

---

## 4.5 Semi-Supervised Pseudo-Labeling

To leverage unlabeled data:

1. Baseline CNN predicts labels on unlabeled images.
2. High-confidence predictions selected using thresholds:

| Threshold | Meaning |
|----------|--------|
| > 0.95 | Vegetation |
| < 0.05 | Non-Vegetation |

3. Selected pseudo-labels added to training data.
4. Model retrained on combined dataset.

---

# 5. Experiments Conducted

---

## Experiment 1: Baseline CNN

Training using only manually labeled data.

Validation Accuracy:

**95.0%**

---

## Experiment 2: CNN + 50 Pseudo Labels

Retrained using:

- Original labeled data
- 50 pseudo-labeled samples

Validation Accuracy:

**92.5%**

---

## Experiment 3: CNN + 300 Pseudo Labels

Retrained using:

- Original labeled data
- 300 pseudo-labeled samples

Validation Accuracy:

**87.5%**

---

# 6. Results & Evaluation

---

## Confusion Matrix

See:

`results/confusion_matrix.png`

Summary:

- True Non-Veg Correct: 18
- False Positive: 2
- False Negative: 2
- True Veg Correct: 18

---

## Classification Report

| Metric | Value |
|-------|------|
| Precision | 0.90 |
| Recall | 0.90 |
| F1-Score | 0.90 |
| Accuracy | 0.90 |

---

## Training / Validation Curves

See:

`results/accuracy_loss_curves.png`

---

# 7. Key Findings

1. Baseline CNN achieved strongest performance.
2. Small pseudo-label additions slightly reduced performance.
3. Large pseudo-label additions degraded performance further.
4. Indicates pseudo-label noise can harm generalization.
5. Demonstrates importance of confidence filtering in semi-supervised learning.

---

# 8. Computational Challenges

The project encountered:

- GPU/RAM limitations in Google Colab
- Long preprocessing times for large TIFF datasets
- Difficulty scaling pseudo-labeling across full 17K dataset

Mitigation strategies included:

- Batch processing
- Drive caching
- Subset pseudo-label experiments

---

# 9. Repository Structure

```text
notebooks/     → Jupyter/Colab notebooks
src/           → Python scripts/modules
results/       → Output plots/metrics
figures/       → Example images/visuals
```

---

# 10. How to Run

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Execute Workflow

1. Run preprocessing notebook/script
2. Run CNN training notebook
3. Run pseudo-labeling notebook
4. Run evaluation notebook

---

# 11. Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Rasterio
- Matplotlib
- Seaborn
- Scikit-Learn

---

# 12. Future Work

Potential improvements include:

- Use transfer learning with pretrained CNNs
- Improve pseudo-label confidence calibration
- Use active learning instead of pseudo-labeling
- Deploy on dedicated GPU hardware for full dataset scaling

---

# 13. Author

**Peace Ebika**  
Artificial Vision Final Project  
Universidad de las Americas Puebla

---
