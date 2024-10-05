# AI/ML Project Template Repository

## Overview

Welcome to the AI/ML Project Template Repository! This repository is designed to serve as an initial structured template for AI and machine learning projects. Whether you're a beginner or an experienced practitioner, this template provides a solid starting point for organizing your project files, documentation, and workflows.

# IT-RUDA (Information Theory Assisted Robust Unsupervised Domain Adaptation)
Maximum Density Divergence for Domain Adaptation published on IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

Authors: Jingjing Li, Erpeng Chen, Zhengming Ding, Lei Zhu, Ke Lu and Heng Tao Shen

PDF on Arxiv: https://arxiv.org/abs/2004.12615  , on IEEE: https://ieeexplore.ieee.org/abstract/document/9080115



# IT-RUDA implemented in PyTorch

## Getting Started
1. **Clone the Repository**: Clone the created repository to your local machine using Git.

   ```bash
   git clone https://github.com/your-username/<your repo name>.git
2. **Navigate to the Project Directory**: Enter the project directory in your terminal or command prompt.
3. **Install Dependencies**: Create the conda environment with dependencies installed:

   ```bash
   conda env create -f environment.yml
4. **Activate the conda environment**:

   ```bash
   conda activate income-prediction-env

## Training
Please use the following commands for different tasks. 

You can find more detailed commands samples in the *train.sh* file
```
SVHN->MNIST
python train_svhnmnist.py --mdd_weight 0.01 --epochs 50

USPS->MNIST
python train_uspsmnist.py --mdd_weight 0.01 --epochs 50 --task USPS2MNIST

MNIST->USPS
python train_uspsmnist.py --mdd_weight 0.01 --epochs 50 --task MNIST2USPS
```
```
Office-31

python train_image.py  --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
```
```
Office-Home

python train_image.py  --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```

```
Image-clef

python train_image.py  --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/b_list.txt --t_dset_path ../data/image-clef/i_list.txt
```

The adversarial learning part is inspired by CDAN.


## Directory Structure

The repository is organized as follows:

- **`data/`:** Contains the dataset used for training and evaluation.
- **`models/`:** Directory for storing trained models.
- **`notebooks/`:** Jupyter notebooks detailing the data exploration, preprocessing, and model training processes.
- **`src/`:** Python scripts for modularized code, including data preprocessing, feature engineering, and model training.
- **`tests/`:** Stores the test files such as data integration tests, model integration tests, responsible AI tests, 
- **`results/`:** Stores the results of the predictive models.
- **`README.md`:** Project overview and usage
- **`LICENSE`:** License file

## Contribution Guidelines

We welcome contributions from the community to improve this template repository. If you have suggestions, bug fixes, or additional features to add, please follow these guidelines:

- Fork the repository and create a new branch for your contribution.
- Make your changes, ensuring they adhere to the project's coding style and conventions.
- Test your changes thoroughly.
- Update documentation if necessary.
- Submit a pull request, providing a detailed description of your changes.

## License

This project is licensed under the [MIT License](License). Feel free to use, modify, and distribute this template for your AI/ML projects.