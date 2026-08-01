# Diabetic Retinopathy Detection — APTOS-2019

A deep learning pipeline that classifies the severity of diabetic retinopathy (DR) from retinal fundus images, using the APTOS-2019 dataset. The project covers the full workflow: fundus-specific image preprocessing, class-imbalance handling, transfer learning with CNNs, and a custom Hybrid CNN + Vision Transformer architecture, evaluated with metrics appropriate for ordinal medical grading.

## Problem

Diabetic retinopathy is graded on a 5-point ordinal scale based on retinal damage severity:

| Label | Class |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Because the classes are ordinal (a Moderate case misclassified as Severe is a much smaller error than being misclassified as No DR), plain accuracy is a misleading metric on its own — this project tracks **Quadratic Weighted Kappa (QWK)** alongside accuracy and macro F1.

## Dataset

- **Source:** [APTOS-2019 dataset on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/aptos2019) (curated split of the original [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) competition data into train/validation/test sets)
- **Format:** Retinal fundus photographs (`.png`/`.jpg`) with an `id_code` → `diagnosis` CSV label file
- **Size used:** 2,930 labeled training images
- **Class distribution (imbalanced):**

| Class | 0 (No DR) | 1 (Mild) | 2 (Moderate) | 3 (Severe) | 4 (Proliferative) |
|---|---|---|---|---|---|
| Count | 1,434 | 300 | 808 | 154 | 234 |

> **Note:** This repo contains the code, not the raw image data. Due to the dataset's size and Kaggle's terms of use, download it directly from the [Kaggle dataset page](https://www.kaggle.com/datasets/mariaherrerot/aptos2019) and place it under `data/` following the structure below — please check the dataset's license/usage terms on Kaggle before redistributing the raw images yourself.

## Pipeline

### 1. Preprocessing
Fundus images have their own quirks (uneven illumination, black borders, noise), so raw resizing isn't enough:
- **Black border cropping** — removes the dark background surrounding the circular retinal scan
- **Denoising** — light Gaussian / Non-Local Means denoising
- **CLAHE contrast enhancement** — applied on the L-channel in LAB color space to make vessels and lesions more visible
- **Resize** to 224×224 and normalize (ImageNet mean/std)

### 2. Quality control
An automated filter flags low-quality images (too dark or too blurry, via Laplacian-variance blur detection) before they're used for training.

### 3. Data handling
- Stratified 80/20 train/validation split to preserve class ratios
- Custom PyTorch `Dataset` / `DataLoader` with on-the-fly preprocessing and augmentation (random flips, rotation)
- **Class-weighted `CrossEntropyLoss`** to counter the label imbalance shown above

### 4. Models
| Model | Approach |
|---|---|
| **ResNet50** | ImageNet-pretrained CNN, fine-tuned — main/best-performing model |
| **EfficientNet-B0** | ImageNet-pretrained CNN baseline for comparison |
| **Hybrid CNN + ViT** *(custom)* | ResNet18 backbone → project features into embedding space → Transformer encoder for global context → classification head |

### 5. Training
- Optimizer: AdamW, `lr=1e-4`, weight decay `1e-4`
- Scheduler: `ReduceLROnPlateau` (on validation QWK)
- Loss: class-weighted Cross-Entropy
- Checkpointing on best validation QWK

## Results

Best ResNet50 checkpoint (epoch 8 / 10):

| Metric | Score |
|---|---|
| Validation QWK | **0.8915** |
| Validation Accuracy | **81.23%** |
| Validation Macro F1 | **0.6806** |

Loss and QWK curves across training are generated in the notebook (`resnet50_history.csv`).

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `timm` · `OpenCV` · `Albumentations` · `scikit-learn` · `pandas` / `NumPy` · `Matplotlib`

## Project Structure

```
.
├── notebooks/
│   └── diabetic_retinopathy_detection.ipynb   # full pipeline: EDA → preprocessing → training → evaluation
├── data/                                       # (not included — see Dataset section)
├── models/                                     # saved checkpoints (best_resnet50.pt, etc.)
├── results/                                    # training history CSVs, plots
└── README.md
```

## How to Run

1. Download the dataset from Kaggle and place it under `data/` (see [Dataset](#dataset))
2. Install dependencies:
   ```bash
   pip install torch torchvision timm opencv-python albumentations scikit-learn pandas matplotlib
   ```
3. Open and run `notebooks/diabetic_retinopathy_detection.ipynb` top to bottom (GPU strongly recommended)

## Future Work

- Complete full training run and evaluation for the Hybrid CNN+ViT model
- Add Grad-CAM explainability to visualize what regions of the retina drive each prediction
- Test-time augmentation and ensembling across ResNet50 / EfficientNet-B0 / Hybrid model
- Evaluate on the held-out APTOS test split

## Acknowledgments

- Dataset: [APTOS-2019 on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/aptos2019), curated from the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) competition (Asia Pacific Tele-Ophthalmology Society)

## License

Code in this repository is released under the MIT License. The dataset itself is subject to its own license/terms on Kaggle.
