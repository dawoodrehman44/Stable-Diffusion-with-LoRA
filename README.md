#  Training Chest X-ray Models with Stable Diffusion-Synthesized Images to Promote Fairness and Performance

<p align="center">
  <img src="Figures/StabbleDiffusionwithLoRA.png" alt="Proposed Methodology" width="1600"/>
</p>

## Highlights

- Fine-tuned a Stable Diffusion model using LoRA for improved medical image representation.
- Replaces complex U-Net training with efficient low-rank adaptation (LoRA) — no need to train the full denoising network from scratch.
- Enhances disease classification performance in downstream deep learning models.
- Improves model focus on disease-specific regions in chest X-rays.
- Reduces disparity and promotes fairness across demographic subgroups (e.g., age, gender, race).

---

## Introduction

This repository supports our AAAI 2026 submission on leveraging diffusion models for medical image understanding. We fine-tune a pre-trained Stable Diffusion model using LoRA to overcome the computational cost and complexity of full U-Net retraining in medical diffusion workflows. 

Our approach targets the following core goals:

- **Classification Boost**: Generate diffusion-guided features that improve disease classification accuracy.
- **Focus Quality**: Encourage attention to disease-relevant regions through learned diffusion features.
- **Fairness and Equity**: Improve fairness across demographic attributes by reducing disparity in model outputs.

The codebase includes training scripts, inference pipelines, evaluation tools, and documentation for reproducing results on public datasets like CheXpert and MIMIC-CXR



This repository contains two core training pipelines:

- **Diffusion Training (`main_train_diffusion_model.py`)** — Fine-tunes Stable Diffusion v1.5 using LoRA on medical captions.
- **Classification Training (`main_train_classification.py`)** — Trains a ResNet-based model on real and synthetic images using CXR-CLIP features.

---

## Directory Structure

```
├── configuration
│   ├── classification_configuration.json
│   └── stable_diffusion_model_confguration.json
├── Data Pre-processing
│   ├── data_pre_processing.py
│   └── data_pre_processing_real_synthetic_datasets.py
├── Figures
├── evalaution_metrics_FID_KID.py
├── evaluation_metrics_for_focus_observation.py
├── evaluation_metrics.py
├── main_train_classification.py
├── main_train_diffsuion_model.py
├── models
│   ├── CheXagent.py
│   ├── CXR-CLIP_Res50.py
│   └── CXR-CLIP_Swin-T.py
├── README.md
├── requirements.yml
└── synthetic_dataset
    └── synthetic_dataset_creation.py
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Key Libraries

- `torch`, `torchvision`  
- `transformers`, `diffusers`, `peft`  
- `accelerate`, `tqdm`, `scikit-learn`  
- `pillow`, `matplotlib`  
- `pytorch-grad-cam`, `captum`  

---

## Data

Real images from (available online):
- CheXpert
- MIMIC-CXR
- ChestX-ray14

Optional: Synthetic images generated via diffusion using the synthetic_dataset_creation file.

**Format:**

```
path,text
similar to CheXpert dataset path "
CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
```

---

## 1. Diffusion Training

Fine-tunes Stable Diffusion using LoRA and medical captions.

```bash
python main_train_diffusion_model.py
```

### Configuration

- **Model**: `runwayml/stable-diffusion-v1-5`
- **Text embedding length**: 77
- **Image size**: 512×512
- **LoRA Config**: r=4, alpha=16, dropout=0.1
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5
- **Epochs**: 200
- **Weight Decay**: 1e-4

---

## 2. Classification Training

Trains on either real or synthetic or mixed CXR images using the required model and features.

```bash
python main_train_classification.py
```

### Configuration

- **Backbone**: ResNet-50 (CXR-CLIP pretrained) (can be changed accordingly)
- **Loss**: BCEWithLogitsLoss
- **Optimizer**: Adam
- **Learning Rate**: 5e-5
- **Scheduler**: Cosine Annealing
- **Weight Decay**: 1e-4
- **Early Stopping**: min. 50 epochs, max. 100
- **Metrics**: AUC, FPR, TPR, Precision, BCE, ECE, Error

---

## Evaluation

Evaluation metrics are implemented in `evaluation_metrics.py`: