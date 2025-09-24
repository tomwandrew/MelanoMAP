# MelanoMAP- Multimodal AI for Tumour Microenvironment Analysis in Cutaneous Melanoma

MelanoMAP is a computational pipeline designed for the automated analysis of whole slide images (WSIs) of cutaneous melanoma. The system integrates deep learning-based segmentation, handcrafted feature extraction, and survival modelling to predict recurrence-free survival (RFS) in AJCC Stage I/II melanoma.

## Key Components

### Whole Slide Image (WSI) Segmentation Pipeline
- Processes whole slide images (WSIs) by segmenting them into patches
- Uses OpenSlide for handling large WSI files
- Utilizes a U-Net-based deep learning model for segmentation.  
- Background, epidermis, tumour, and TME regions are segmented into 256x256 non-overlapping
patches for further analysis.

### Dual Feature Extraction Approaches

#### Handcrafted (HC) Analysis
- Extracted using traditional image processing techniques.
- Includes morphological, intensity, texture, and spatial characteristics.
- Features:
  - Intensity & Gradient-Based Features (mean signal, skewness, variability)
  - Texture Features (GLCM-based contrast, homogeneity, ASM)
  - Morphological Features (area, compactness, perimeter, density)

#### Deep Learning Analysis
- Deep learning models (EfficientNetV2) extracts high-dimensional image representations.
- Trained on labeled data to classify patches according to staining intensity, spatial distribution, expression loss and gap size.
- Aggregates patch-level predictions to make slide-level decisions

### Integration with QuPath
- Scripts for exporting binary masks and tiles from QuPath
- Allows for annotation and preprocessing of WSIs in QuPath before analysis

### Survival Prediction Models
- Three models predict recurrence-free survival (RFS):
  1. Cox Proportional Hazards (Cox)
  2. Random Survival Forest (RSF)
  3. Deep Survival Networks (DeepSurv)
- Models are trained using:
  - Clinical-only features
  - Imaging-only features
  - Multimodal features (clinical + imaging)
- Performance evaluated using:
  - Concordance Index (C-index)
  - Integrated Brier Score (IBS)
  - Kaplan-Meier stratification

## Main Workflows

### Whole Slide Segmentation
- Entry point for the segmentation pipeline
- Determines staining type (H+E, AM or Lo) and segment images.
  - Reads input WSI.
  - Segments tumour and TME regions.
  - Extracts 256x256 patches for feature analysis.

### H+E Marker Analysis
- Processes slides with H&E staining.
- Segments tumour, TME, and background regions.
- Extracts texture and morphological features from the tumour microenvironment.
- Uses handcrafted and deep learning models.
- Outputs heatmaps visualizing tumour TME interactions.

### AM Marker Analysis
- Processes slides with AM staining
- Detect and quantify AMBRA1 expression.
- Extracts intensity and spatial features.
- Uses handcrafted and deep learning models.
- Outputs a biomarker expression probability map.

### LO Marker Analysis
- Processes slides with LO staining
- Uses AI-based prediction for gene expression status
- Provides detailed statistics on prediction confidence

### Quality Control
- Ensures reliability of extracted patches.
- Filters out low-quality images based on:
  - Overexposure
  - Insufficient tumour regions
  - Excessive staining noise

## Technical Details

- **Deep Learning**: Uses EfficientNetV2 models trained for binary classification
- **Image Processing**: Extensive use of traditional computer vision techniques
- **Data Management**: Structured storage of results with CSV exports for analysis
- **Visualization**: Tools for visualizing results and creating heatmaps

## Setup and Deployment

### Requirements
- Docker and docker-compose
- CUDA-compatible GPU (optional but recommended)

### Installation
1. Clone this repository
2. Build the Docker container:
   ```
   docker-compose build
   ```
3. Run the container:
   ```
   docker-compose up -d
   ```

## Data Availability

The whole slide images (WSIs) used in this study contain patient-identifiable details and therefore cannot be publicly shared. Due to ethical and regulatory constraints, we are unable to provide a test dataset.  

However, de-identified clinicopathological data may be made available upon reasonable request, subject to institutional and ethical approvals. The provided scripts and pipelines ensure reproducibility, allowing users to apply these methods to their own datasets.  

For guidance on preprocessing and expected input formats, refer to the Setup and Usage sections.

### Directory Structure
- `app/`: Main application code
  - `ai_functionalities/`: Deep learning prediction models
  - `handcraft_detection/`: Traditional feature extraction methods
  - `segmentation/`: WSI segmentation pipeline
  - `utils/`: Utility functions
- `qupath/`: QuPath integration scripts
- `Dockerfile`: Container definition
- `requirements.txt`: Python dependencies

## Usage

### Processing AM Slides
```bash
python app/main_am.py
```

### Processing LO Slides
```bash
python app/main_lo.py
```

### Running the Complete Pipeline
```bash
bash app/pipeline.sh
```

## Dependencies

The system relies on several key libraries:
- PyTorch and torchvision for deep learning
- OpenSlide for WSI handling
- albumentations for image augmentation
- scikit-image for image processing
- pandas and numpy for data manipulation
- matplotlib and seaborn for visualization

See `requirements.txt` for a complete list of dependencies.
