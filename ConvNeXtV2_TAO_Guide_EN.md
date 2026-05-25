# ConvNeXtV2 Large (Pre-trained Model) Guide
## Training and Inference with NVIDIA TAO Toolkit (Beginner-Friendly)

This document provides step-by-step instructions for beginners to perform image classification training and inference using the ConvNeXtV2 Large pre-trained model from NVIDIA NGC.

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Model Download](#4-model-download)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Training (Fine-tuning)](#6-training-fine-tuning)
7. [Running Inference](#7-running-inference)
8. [Model Export](#8-model-export)
9. [Troubleshooting](#9-troubleshooting)
10. [Reference Links](#reference-links)

---

## 1. Overview

**ConvNeXtV2** is a state-of-the-art CNN (Convolutional Neural Network) architecture that can be used as a feature extraction backbone for image classification, object detection, semantic segmentation, and other computer vision tasks.

### Key Features

| Item | Description |
|------|-------------|
| License | Commercial use allowed (NVIDIA Open Model License) |
| Input Image Size | 224 x 224 x 3 (RGB) |
| Model Size | ~2.21GB (compressed) |
| Supported TAO Version | TAO 5.5.0 and later |
| Output | 2D Float Tensor (Batch size x 1000) |

---

## 2. Requirements

### Hardware Requirements

#### Minimum Configuration

| Item | Minimum Requirement |
|------|---------------------|
| System RAM | 8GB |
| GPU VRAM | 4GB |
| CPU | 8 cores |
| GPU | At least 1 NVIDIA GPU (Pascal generation or later) |
| Storage | 100GB SSD |

#### Recommended Configuration

| Item | Recommended |
|------|-------------|
| System RAM | 32GB |
| GPU VRAM | 32GB |
| CPU | 8 cores |
| GPU | At least 1 NVIDIA GPU |
| Storage | 100GB SSD |

### Supported GPUs

- NVIDIA Pascal
- NVIDIA Volta
- NVIDIA Turing
- NVIDIA Ampere
- NVIDIA Lovelace
- NVIDIA Hopper
- NVIDIA Blackwell

### Software Requirements

| Software | Version |
|----------|---------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 or higher |
| Docker CE | 19.03.5 or higher |
| NVIDIA Driver | 550 or higher |
| NVIDIA Container Toolkit | 1.3.0 or higher |

---

## 3. Environment Setup

### Step 3.1: Install Docker CE
```bash
# Remove old Docker installations
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install required packages
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker CE
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Allow running Docker without sudo (Important!)
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3.2: Install NVIDIA Container Toolkit
```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Step 3.3: Set Up Python Environment (Using Miniconda)
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Restart terminal or run
source ~/.bashrc

# Create a new environment
conda create -n tao python=3.10 -y

# Activate the environment
conda activate tao

# Add Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name tao --display-name "tao"
```

### Step 3.4: Obtain NGC Account and Generate API Key

1. Visit [NVIDIA NGC](https://ngc.nvidia.com/)
2. Click "Create Account" (free)
3. After logging in, click on your profile icon in the top right
4. Select "Setup" → "API Key"
5. Click "Generate API Key"
6. **Save the generated API key securely** (it's only shown once)

### Step 3.5: Install and Configure NGC CLI
```bash
# Download NGC CLI
wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_linux.zip -O ngccli_linux.zip

# Extract and set up
unzip ngccli_linux.zip
chmod u+x ngc-cli/ngc

# Add to PATH
echo 'export PATH="$PATH:$HOME/ngc-cli"' >> ~/.bashrc
source ~/.bashrc

# Configure NGC CLI (enter your API key)
ngc config set
```

When prompted, enter:
- **API Key**: Your generated API key
- **CLI output format**: ascii (default)
- **org**: nvidia (or your organization)

### Step 3.6: Login to Docker Registry and Install TAO Launcher
```bash
# Login to NGC Docker registry
docker login nvcr.io
```

Enter the following credentials:
- **Username**: `$oauthtoken` (enter literally)
- **Password**: Your NGC API key
```bash
# Clone the tutorial repository
git clone https://github.com/NVIDIA/tao_tutorials.git
cd tao_tutorials

# Install TAO Launcher
bash setup/quickstart_launcher.sh --install

# Verify installation
tao --help
```

---

## 4. Model Download

### 4.1 Create Working Directories
```bash
# Create working directories
mkdir -p ~/tao_experiments/{pretrained_models,dataset,results,specs}
cd ~/tao_experiments
```

### 4.2 Download Model Using NGC CLI
```bash
# Download ConvNeXtV2 Large model
ngc registry model download-version nvidia/tao/pretrained_convnextv2:convnextv2_large_v1.0 --dest ./pretrained_models/
```

After download, the model will be saved at:
```
~/tao_experiments/pretrained_models/pretrained_convnextv2_vconvnextv2_large_v1.0/
```

### 4.3 Verify Download
```bash
ls -la ~/tao_experiments/pretrained_models/pretrained_convnextv2_vconvnextv2_large_v1.0/
```

---

## 5. Dataset Preparation

### 5.1 Required Directory Structure

TAO image classification requires the following directory structure:
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image001.jpg
│   │   └── ...
│   └── classN/
│       └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── classN/
├── test/  (optional)
│   ├── class1/
│   ├── class2/
│   └── classN/
└── classes.txt
```

### 5.2 Directory Creation Example (Cat and Dog Classification)
```bash
# Create dataset directories
mkdir -p ~/tao_experiments/dataset/{train,val}/{cat,dog}

# Create classes.txt (one class name per line)
cat > ~/tao_experiments/dataset/classes.txt << EOF
cat
dog
EOF
```

### 5.3 Place Your Images

- Place training images in `dataset/train/classname/`
- Place validation images in `dataset/val/classname/`
- Recommended: at least 100 images per class

---

## 6. Training (Fine-tuning)

### 6.1 Create Configuration File

Create `~/tao_experiments/specs/classification_experiment.yaml`:
```yaml
model:
  backbone:
    type: "convnextv2_large"
    pretrained_backbone_path: "/workspace/pretrained_models/pretrained_convnextv2_vconvnextv2_large_v1.0/convnextv2_large_trainable_v1.0.pth"
    freeze_backbone: False
  head:
    type: "TAOLinearClsHead"
    in_channels: 1536
    binary: False
    topk: [1, 5]
  loss:
    type: CrossEntropyLoss

dataset:
  dataset: "CLDataset"
  root_dir: /workspace/dataset
  batch_size: 16
  workers: 4
  num_classes: 2
  img_size: 224
  augmentation:
    mixup_cutmix: False
    random_flip:
      hflip_probability: 0.5
      vflip_probability: 0
      enable: True
    random_aug:
      enable: True
    random_erase:
      enable: True
  train_dataset:
    images_dir: /workspace/dataset/train
  val_dataset:
    images_dir: /workspace/dataset/val

train:
  num_epochs: 30
  checkpoint_interval: 5
  validation_interval: 1
  num_gpus: 1
  gpu_ids: [0]
  results_dir: /workspace/results/train
  optim:
    optim: adamw
    lr: 0.0001
    weight_decay: 0.01
    policy: cosine
    warmup_epochs: 5
  tensorboard:
    enabled: True
```

### 6.2 Create Mount Configuration File

Create `~/.tao_mounts.json`:
```json
{
    "Mounts": [
        {
            "source": "/home/YOUR_USERNAME/tao_experiments/dataset",
            "destination": "/workspace/dataset"
        },
        {
            "source": "/home/YOUR_USERNAME/tao_experiments/pretrained_models",
            "destination": "/workspace/pretrained_models"
        },
        {
            "source": "/home/YOUR_USERNAME/tao_experiments/results",
            "destination": "/workspace/results"
        },
        {
            "source": "/home/YOUR_USERNAME/tao_experiments/specs",
            "destination": "/workspace/specs"
        }
    ]
}
```

> **Note**: Replace `YOUR_USERNAME` with your actual username.

### 6.3 Run Training
```bash
# Create results directory
mkdir -p ~/tao_experiments/results/train

# Start training
tao model classification_pyt train \\
  -e /workspace/specs/classification_experiment.yaml \\
  results_dir=/workspace/results/train
```

### 6.4 Monitor Training (TensorBoard)

In a separate terminal:
```bash
pip install tensorboard
tensorboard --logdir ~/tao_experiments/results/train --host 0.0.0.0 --port 6006
```

Access `http://localhost:6006` in your browser

---

## 7. Running Inference

### 7.1 Add Inference Configuration

Add the following to `classification_experiment.yaml`:
```yaml
inference:
  checkpoint: /workspace/results/train/model_latest.pth
  batch_size: 8
  num_gpus: 1
  gpu_ids: [0]
  results_dir: /workspace/results/inference
```

### 7.2 Run Inference
```bash
# Create inference directory
mkdir -p ~/tao_experiments/results/inference

# Run inference
tao model classification_pyt inference \\
  -e /workspace/specs/classification_experiment.yaml \\
  inference.checkpoint=/workspace/results/train/model_latest.pth \\
  results_dir=/workspace/results/inference
```

### 7.3 View Results
```bash
# View inference results
cat ~/tao_experiments/results/inference/result.csv
```

Output format:
```
image_path,predicted_class,confidence
/path/to/image1.jpg,cat,0.98
/path/to/image2.jpg,dog,0.95
```

---

## 8. Model Export

### 8.1 Export to ONNX Format
```bash
# Create export directory
mkdir -p ~/tao_experiments/results/export

# Export model to ONNX format
tao model classification_pyt export \\
  -e /workspace/specs/classification_experiment.yaml \\
  export.checkpoint=/workspace/results/train/model_latest.pth \\
  export.onnx_file=/workspace/results/export/model.onnx \\
  export.input_width=224 \\
  export.input_height=224 \\
  export.opset_version=17
```

### 8.2 Verify Export Results
```bash
ls -la ~/tao_experiments/results/export/
```

The exported `.onnx` file can be converted to a TensorRT engine for deployment.

---

## 9. Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `tao` command not found | PATH not set | Run `export PATH=$PATH:~/.local/bin` and add to `~/.bashrc` |
| Docker permission error | User not in docker group | Run `sudo usermod -aG docker $USER` and re-login |
| GPU not recognized | Driver or Container Toolkit issue | Check with `nvidia-smi`, reinstall Container Toolkit |
| Out of Memory (OOM) | Insufficient GPU memory | Reduce `batch_size` (16→8→4) |
| Segmentation fault with multi-GPU | Thread contention | Set `export OMP_NUM_THREADS=1` |
| Model not found | Incorrect path | Verify paths in `~/.tao_mounts.json` |

### Checking Logs
```bash
# View training logs
cat ~/tao_experiments/results/train/train.log

# View Docker container logs
docker logs $(docker ps -lq)
```

---

## 📚 Reference Links

| Resource | URL |
|----------|-----|
| NGC ConvNeXtV2 Model Page | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_convnextv2 |
| TAO Toolkit Documentation | https://docs.nvidia.com/tao/tao-toolkit/ |
| TAO Tutorials (GitHub) | https://github.com/NVIDIA/tao_tutorials |
| TAO Quick Start Guide | https://docs.nvidia.com/tao/tao-toolkit/latest/text/quick_start_guide/index.html |
| ConvNeXtV2 Paper | https://arxiv.org/abs/2301.00808 |

---

## ⚠️ Important Notice

> **TAO Launcher Deprecation Notice**
> 
> Starting with TAO 7.x, the TAO Launcher will be officially deprecated. For long-term projects, please consider migrating to **FTMS (Fine-Tuning Micro-Services)**.
> 
> For FTMS details, refer to the [TAO FTMS Documentation](https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_toolkit_api/index.html).

---

*This document is based on NVIDIA TAO Toolkit 5.5.0.*