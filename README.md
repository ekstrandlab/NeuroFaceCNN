# NeuroFaceCNN
CNN-based deep learning framework for temporally preserved face classification from naturalistic fMRI data using voxel-by-time transformations

## Overview

**NeuroFaceCNN** is a deep learning framework for decoding face-related brain activity using naturalistic fMRI data. It transforms high-dimensional 4D brain volumes into compact 2D voxel-by-time matrices to preserve temporal dynamics and simplify spatial complexity. The approach enables efficient CNN-based classification of 10-second fMRI windows as “face” or “no-face” events and employs attribution methods (Integrated Gradients via DeepExplain) to reconstruct interpretable neural representations in anatomical brain space.

This repository supports the manuscript:

**Asadi, S. et al. (2025)** *Computationally Efficient Deep Learning for Temporally Preserved Face Classification in Naturalistic fMRI*



## Dataset

Source: Naturalistic Neuroimaging Database v2.0 (Aliko et al., 2020)

Participants: 85 healthy adults (42F, ages 18–58)

Stimuli: Movie-watching fMRI with detailed face event annotations

Labels: 3606 10-second segments labeled as "face" or "no-face" 



## Key Features

Temporal preservation: transforms 4D fMRI → 41,489×10 voxel-time matrices

Lightweight CNN: custom 2D convolutional model trained from scratch

Naturalistic fMRI: based on 85 participants watching full movies from NNDb v2.0

Balanced event labels: 1,803 face-onset + 1,803 face-offset trials

Attribution analysis: Integrated Gradients using DeepExplain

3D reconstruction: voxel-level IG maps returned to brain space (NIfTI)

Surface visualization: subject-level and group-level cortical maps




## Pipeline Overview
### 1. Preprocessing & Event Extraction
- Uses NNDb face-presence annotations  
- Defines face onset and face offset (≥10 s) windows  
- Applies 4-second hemodynamic lag  
- Extracts 10-TR (10-second) 4D segments per event

### 2. Group Brain Mask
- Built from 85 individual masks (MNI)  
- 99% overlap threshold → 41,489 stable voxels

### 3. Voxel-by-Time Transformation
- 4D block (X×Y×Z×10) → masked (41489 × 10) matrix  
- Stored as .npy files

### 4. CNN Model
- 2D CNN with:
  - LeakyReLU activations  
  - L2 weight decay  
  - BatchNorm + Dropout  
- 4-fold stratified cross-validation  
- Trained on GPU with mixed precision

### Performance
- Accuracy: ~84–85%  
- AUC: 0.91  
- Stable training and generalization

### 5. Integrated Gradients
- DeepExplain TF1.x graph mode  
- Attributions computed on pre-sigmoid logit  
- Zero baseline  
- 200 interpolation steps

### 6. Reconstruction to Brain Space
- IG matrix (41489 × 10) → 4D brain (X×Y×Z×10)  
- Saved as NIfTI volumes  
- Subject-average and group-level maps supported

### 7. Visualization
- Surface projections for each second


### 8. Moment-by-Moment fMRI Prediction

- Continuous decoding of face-related neural activity from naturalistic fMRI using the trained CNN model 

## Repository Structure

| File | Description |
|------|-------------|
| `1-preprocess_naturalistic_fmri.py` | Creates voxel-by-time matrices from raw 4D fMRI data while preserving temporal information|
| `2-load-data.py` | Loads and labels segmented fMRI samples |
| `3-train_voxel_time_cnn.py` | Trains custom CNN model on voxel-by-time input |
| `4-integrated_gradients.py` | Computes voxel-level IG attribution maps |
| `5-revert-IG-tobrain.py` | Maps 2D IG attributions back to brain space |
| `6-average_IG_per_subject.py` | Aggregates and averages IG maps by subject |
| `7-visualize_IG_surface_maps.py` | Projects IG maps onto fsaverage cortical surface |
| `8-moment_by_moment_face_prediction.py` | Performs continuous CNN-based prediction across unseen fMRI |
| `README.md` | Project overview and usage |

## Setup

This codebase requires Python 3.8+ and the following libraries:

- `tensorflow>=2.10`
- `numpy`, `matplotlib`, `nibabel`, `nilearn`, `scikit-learn`
- `DeepExplain` (for attribution)
- `nibabel`, `scipy`, `pandas`

Mixed precision training and GPU acceleration (e.g., RTX 3090) are recommended.



## Data Requirements
You must supply:
- fMRI NIfTI files
- Face annotations
- Group brain mask (85_subBrainMask_average_99.nii.gz)
  
NNDb dataset: https://www.naturalistic-neuroimaging-database.org/


## Results Summary
- CNN reliably distinguishes face-onset vs. face-offset patterns  
- Integrated Gradients highlight FFA, OFA, pSTS  
- Consistent attribution across subjects  
- Group maps aligned with known face-processing networks
- Moment-by-moment predictions demonstrate continuous decoding of neural activity in naturalistic viewing  


## Citation

If you use this code, please cite:

Asadi, S. et al. (2025). *Computationally Efficient Deep Learning for Temporally Preserved Face Classification in Naturalistic fMRI*.


