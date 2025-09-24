# Analysis of TME Patterns in Histological Images

## Abstract
This study presents MelanoMAP, a multimodal AI framework integrating whole-slide image (WSI) analysis with clinical and histopathological features to predict recurrence-free survival (RFS) in AJCC stage I/II cutaneous melanoma (CM). Using a Convolutional Neural Network (CNN)-based segmentation pipeline, we extracted image-derived biomarkers from haematoxylin and eosin (H&E), AMBRA1, and Loricrin-stained WSIs. These were combined with clinical predictors in survival models including Cox Proportional Hazards (Cox), Random Survival Forest (RSF), and Deep Survival Networks (DeepSurv). Decision Curve Analysis (DCA) and Kaplan-Meier stratification were applied to evaluate the model's clinical utility.

## Introduction
This study presents an automated computational pipeline for analyzing AM biomarker expression in histological images. The aim is to integrate deep learning-derived tumour microenvironment (TME) features with clinical parameters to enhance prognostication in non-ulcerated Stage I/II cutaneous melanoma. Using a combination of handcrafted (HC) features and convolutional neural networks (CNNs), we extracted intensity, texture, morphological, and spatial characteristics of AM-positive regions, which were then evaluated for their prognostic significance in relation to recurrence-free survival (RFS).

## Methods
### Dataset and Inclusion Criteria:
1. The dataset included 3,657 whole slide images (WSIs) with primary non-ulcerated AJCC (8th edition) stage I/II melanoma between 2004 and 2014. Patients were sourced from six international centers:

University Hospital North Durham, UK

James Cook University Hospital, UK

Roswell Park Comprehensive Cancer Center, USA

Peter MacCallum Cancer Centre, Australia

Hospital Clinic Barcelona, Spain

Northern Ireland Tissue Biobank, UK

2. Inclusion Crtieria:
   - Age ≥18 years
   - Complete peripheral margins in diagnostic biopsy
   - Minimum of 10 years clinical follow-up

3. Exclusion Crtieria:
   - Ulcerated melanoma
   - Acral lentiginous, mucosal, or non-cutaneous melanoma 
   - Previous melanoma history or multiple in-situ melanomas
   - Unresectable stage III/IV disease
   - Pregnancy or inability to provide consent

4. Primary Outcome: 

Recurrence-Free Survival (RFS), defined as the time from complete excision to first recurrence.

5. Ethics approval was obtained via Newcastle University Dermatology Biobank (REC REF 24/NE/0014).   

## Detailed Methods

### 1. Feature Selection and Data Imputation

#### 1.1 Clinical Feature Selection:
Key clinical variables were standardized to mitigate biases. Collinearity checks excluded variables with correlations exceeding ±0.7 (Supplementary Table 1). The final set of clinical and imaging derived features included:
  - Clinical features: Patient age, sex, anatomical location, histological subtype, Breslow depth, and mitotic count
Imaging-derived features:
  - Microenvironmental Features: Mean microenvironmental signal, loss of microenvironmental signal, intensity skewness, variability, mean area, compactness, perimeter, number of regions, spatial spread, and density.
  - Texture Features: Angular second moment (ASM), homogeneity, contrast, and energy.


#### 1.2 Data Imputation:
Three imputation methods were evaluated:
  - Mean imputation
  - K-nearest neighbours (KNN) imputation
  - Iterative random forest (RF)-based imputation

RF-based imputation achieved the highest accuracy (87.3%) and was applied only to the training dataset to prevent data leakage. The validation and test datasets contained no missing data and were not imputed.

### 2. Digital Biomarker Pipeline Development

#### 2.1 Image Segmentation:
-WSIs underwent automated segmentation using a modified U-Net-based CNN to identify:
  1.  Background
  2.  Epidermis
  3.  Tumour
  4.  Tumour Microenvironment (TME)
  5.  Immunohistochemistry (IHC) Staining Regions

Segmented images were divided into 256 × 256 pixel non-overlapping patches for further analysis.

#### 2.2 Feature Extraction:
- Handcrafted Features (HC): Traditional features capturing intensity, texture (GLCM-based), morphology (shape descriptors), and spatial patterns.
- Deep Learning Features: A CNN (EfficientNet-based) was trained to extract high-dimensional representations from each image patch.

For AMBRA1-stained slides, intensity and gradient distribution were analyzed to quantify TME signal.
For Loricrin-stained slides, gaps > 20 microns were detected using a CNN-based approach to assess protein expression loss.

### 3. Model Development and Evaluation

#### 3.1 Survival Model Development:
-Three survival models were developed to predict time-to-event outcomes:
  1.  Cox Proportional Hazards (Cox)
  2.  Random Survival Forest (RSF)
  3.  Deep Survival Model (DeepSurv)

Each model was trained using:
  1. Clinical-only features
  2. Imaging-only features
  3. Multimodal (clinical + imaging) features

#### 3.2 Performance Metrics:
- Model performance was assessed using:
  • Concordance Index (C-index)
  • Integrated Brier Score (IBS)
  • Calibration curves (2–5 years)

External validation was performed using an independent cohort from the Northern Ireland Tissue Biobank.

#### 3.3 Clinical Utility Assessment:
- To evaluate clinical applicability, Decision Curve Analysis (DCA) was conducted to assess the net benefit of the models across clinically relevant risk thresholds.

- Risk Stratification:
  - MelanoMAP risk score (0-1)
  - Patients were classified as high-risk (≥0.95) or low-risk (<0.95) based on the median predicted survival probability at 60 months.
  - Kaplan-Meier survival curves were used to validate risk stratification.

#### 3.4 Explainability and Feature Importance:


- SHAP (Shapley Additive Explanations) analysis was conducted to interpret model predictions and determine feature importance.

Software & Packages:
All computational analyses were performed in Python, with a full list of software versions provided in Supplementary Table 2.