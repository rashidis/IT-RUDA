# IT-RUDA (Information Theory Assisted Robust Unsupervised Domain Adaptation)

## Overview

This repository implements domain adaptation methods using information theory and alpha divergence. 

The project supports multiple domain adaptation scenarios including:

- Closed Set (CS)
- Partial Domain Adaptation (PDA) 
- Open Set (OS)

## Directory Structure

The repository is organized as follows:

- **`data/`:** Contains dataset files and preprocessing scripts for Office-31, Office-Home and Image-CLEF
- **`src/`:** Core implementation files including:
  - Network architectures (ResNet, VGG, AlexNet)
  - Training loops
  - Loss functions
  - Data loading utilities
- **`logs/`:** Training logs and saved models
- **`LICENSE`:** MIT License

## Getting Started

1. **Clone the Repository**:
```bash
git clone <repository-url>
```

2. **Set Environment**:

```bash
source set_env.sh
```

3. **Install Dependencies**:
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib

## Training

The repository supports training on multiple datasets and adaptation scenarios:
```bash
python src/train_image.py
```