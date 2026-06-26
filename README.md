Air-Tissue Boundary (ATB) Prediction in rtMRI Videos

This repository contains the machine learning pipeline developed for the **Air-Tissue Boundary (ATB) Prediction Challenge**, an initiative by the **SPIRE Lab, Indian Institute of Science (IISc)**. 

The objective of this project is to accurately predict vocal tract air-tissue boundaries from mid-sagittal real-time Magnetic Resonance Imaging (rtMRI) video frames using temporal context information. The model shifts from complex sequence deep learning to an optimized, highly efficient multi-contour **Ridge Regression** architecture that establishes a strong, scalable baseline for articulatory tracking.

---

## 📌 Problem Statement

In speech production research, real-time MRI (rtMRI) is used to visualize the movement of upper airway articulators (lips, tongue, velum, pharynx, and larynx). Tracking the boundaries between air and tissue is crucial for understanding speech motor control and developing advanced speech synthesis/recognition systems.

### The Task
Given a target frame index $t$ from an rtMRI video sequence, the goal is to predict the complete air-tissue boundary contours using **only** the surrounding annotated frames at a temporal distance $n$ ($t - n$ and $t + n$). 
* The system evaluates context windows for $n \in [1, 5]$.
* The facial/vocal tract tissue boundaries are divided into **three independent contours**:
  * **Contour 1 (Red):** Upper boundary including the nose, upper lip (UL), and hard palate.
  * **Contour 2 (Blue):** Lower boundary including the lower lip (LL), jaw, tongue body, and alveolar ridge (AVR).
  * **Contour 3 (Green):** Posterior boundary including the velum (VEL), pharyngeal wall, and glottal boundary (GLTB).

---

## ⚙️ System Architecture & Methodology
1. Data Annotation & Structuring
The inputs are MATLAB structures (`.mat` files) containing manually annotated boundaries generated via a custom MATLAB UI. The `VideoDatasetLoader` dynamically extracts the data and formats it into iterable frames while handling variable cell-array layout variations natively.

### 2. Spatial Resampling & Normalization
Contours extracted from raw annotations can vary in point density. To guarantee a fixed-size feature vector for standard regression models:
* The pipeline computes the cumulative Euclidean distance along each contour curve.
* Linear/Cubic 1D interpolation (`scipy.interpolate.interp1d`) is used to resample each contour into a standardized grid of exactly **100 equidistant coordinate points** $((x_i, y_i) \text{ for } i=1 \dots 100)$.

### 3. Predictive Modeling Framework
* **Baseline Interp Model (`base_model.py`):** Uses a symmetric midpoint linear interpolation scheme where the target prediction is the direct average of past and future frames: $\hat{I}(t) = \frac{I(t-n) + I(t+n)}{2}$.
* **Machine Learning Model (`train_new.py`):** Trains three independent `scipy-learn` Ridge Regression variants. For a given contour, the coordinate matrices of the past and future frames are flattened and concatenated to build a high-dimensional feature vector:
  $$X = \big[x_{t-n,1}, y_{t-n,1}, \dots, x_{t+n,1}, y_{t+n,1}, \dots \big]$$
  The target array is the flattened coordinates vector of the central ground truth frame at time $t$. 
  * Independent tuning is applied per contour (e.g., lower penalty $\alpha = 0.5$ for the highly dynamic tongue contour `contour2`).

---

## 📊 Evaluation Metrics

The quality of predictions is assessed against manual ground truth boundaries using two complementary geometric metrics:

1. **Dynamic Time Warping (DTW) Distance:** Measures the optimal alignment path distance between the predicted coordinate array and the resampled ground-truth array to penalize structural shape distortions, normalized by the point budget ($N=100$).
2. **Point Accuracy Metric (%):** Quantifies the percentage of the 100 predicted contour points that fall within a strict Euclidean distance threshold ($\tau < 2.0$ pixels) of their corresponding ground-truth counterparts.

---

## 📁 Repository Structure

```📂
├── base_model.py      # Core ATBMultiModel class handling interpolation, resampling, and metrics
├── dataset.py         # VideoDatasetLoader parsing MATLAB structures into Python pipelines
├── train_new.py       # Training script for independent multi-contour Ridge Regression models
├── predict.py         # Prediction runner producing structured submission .mat files
└── README.md          # Comprehensive project documentation
