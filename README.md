# README for Appendix Source Code of BrightVO

## Introduction

This repository contains the source code provided as supplementary material for the paper **BRIGHT-VO: Brightness-Guided Hybrid Transformer for Visual Odometry with Multi-modality Refinement Module**, submitted to **IJCAI Main Track**. The code is intended to demonstrate the implementation details and facilitate the reproducibility of the experiments presented in the paper.

## Requirements

The following software and libraries are required to run the code:

- Python >= 3.9
- PyTorch >= 2.5.0
- Cudatoolkit >= 11.8.0

You can install the dependencies using the following command:

```bash

conda create --name brightvo python=3.9
pip install -r requirements.txt

```

## Architecture

The repository is organized as follows:
├── evaluate_kitti.py       # Entry point for running the experiments
├── checkpoint/             # Path to save checkpoints
├── models/                 # Implementation of the proposed model (e.g., BrightVO)
    ├── ViOT.py             # ViT-based Visual Odometry                   
│   ├── RefinementModule.py # Back-end refinement module based on pose graph optimization
│   ├── imu_integrator.py   # IMU-Integrator to obtain IMU poses
│   └── build_model.py      # Load pre-trained model and util BrightVO
├── datasets/               # Scripts for preprocessing and loading datasets
│   ├── transformation.py   # Data preprocessing pipeline
│   └── kitti.py            # Data loader implementation
├── config/                 # Configuration files for experiments
│   └── cfg.yaml            # Example configuration
├── results/                # Directory to save results and logs
└── README.md               # This file

## Run the code
1. Prepare the Dataset: Download KITTI datasets (including raw-data and ground-truth files) on the official website: https://www.cvlibs.net/datasets/kitti/ (ps: KiC4R datasets will be released upon acceptance of the paper)
2. You can download our pre-trained model here: https://drive.google.com/file/d/1DCLOL8jaiS1P2T59Sfeg5Tp9XNrK9YdO/view?usp=sharing
3. Use the following command to execute the experiments
```bash
python evaluate_kitti.py
```

## Visualize the results

We have provided our experimental results on the KITTI dataset in the "results" folder

```bash

pip install evo

evo_ape kitti BrightVO/results/00.txt /Path/to/groundtruth/00.txt --vas --plot --plot_mode xz                    # (unit: m) Command to visualize trajectory and show error in Absolute Trajectory Error (ATE)
evo_rpe kitti BrightVO/results/00.txt /Path/to/groundtruth/00.txt --vas --delta=100 --delta_unit=m -r trans_part # (unit: %) Command to illustrate trans-part Relative Pose Error (RPE) 
evo_rpe kitti BrightVO/results/00.txt /Path/to/groundtruth/00.txt --vas --delta=100 --delta_unit=m -r angle_deg  # (unit: deg/100m) Command to illustrate rotation-part Relative Pose Error (RPE) 

```