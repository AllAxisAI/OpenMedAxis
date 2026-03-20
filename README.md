# [OpenMedAxis](https://openmedaxis.allaxisai.com/)


**PyTorch Lightning for Generative Medical Imaging.**

OpenMedAxis is an open-source framework designed to make **generative medical imaging research reproducible, modular, and easy to experiment with**. It provides clean implementations of datasets, generative models, training pipelines, and evaluation tools commonly used in medical imaging research.

The goal is to create a **researcher-first framework** that removes boilerplate and fragmented codebases while keeping the system **low-level, flexible, and hackable**.

---

# Why OpenMedAxis?

Research in generative medical imaging often suffers from the same problems:

* Dataset preprocessing scripts are scattered across different repositories
* Training pipelines are inconsistent and difficult to reproduce
* Implementations of GANs or diffusion models vary widely
* Evaluation pipelines are rarely standardized

OpenMedAxis aims to solve this by providing a **clean, unified research toolkit** for the community.

Instead of every paper reinventing the same infrastructure, OpenMedAxis provides:

* standardized dataset pipelines
* reference implementations of generative models
* modular training components
* reproducible experiment structure

---

# Key Features

### Medical Imaging Dataset Tools

Utilities and standardized loaders for common datasets.

Planned initial support:

* IXI
* BraTS
* fastMRI

Each dataset module will include:

* download utilities
* preprocessing scripts
* standardized dataset interfaces

---

### Generative Model Zoo

Reference implementations of commonly used generative models in medical imaging.

Planned models:

**GAN-based models**

* DCGAN
* pix2pix
* CycleGAN
* SAGAN

**Diffusion-based models**

* DDPM
* DDIM
* Latent Diffusion

Future additions may include:

* Flow Matching


---

### Modular Training Framework

A lightweight and extensible training framework inspired by modern deep learning libraries.

Features include:

* experiment configuration
* modular trainer
* logging and checkpointing
* reproducible experiments

Researchers can easily swap components such as:

* models
* optimizers
* schedulers
* loss functions

---

### Evaluation and Metrics

OpenMedAxis will provide standardized evaluation tools for generative medical imaging.

Planned metrics include:

* PSNR
* SSIM
* MS-SSIM
* FID
* LPIPS

Future work will include domain-specific metrics for medical image realism and structure preservation.

---

### Research Utilities

Tools designed to reduce repetitive boilerplate code.

Examples include:

* preprocessing pipelines
* visualization tools
* slice viewers
* experiment templates
* training utilities

---

# Installation

Currently the project is in **pre-alpha development**.

Clone the repository:

```bash
git clone https://github.com/AllAxisAI/openmedaxis.git
cd openmedaxis
```

Install in editable mode:

```bash
pip install -e .
```

---

# Repository Structure

```
openmedaxis/

├── README.md
├── LICENSE
├── pyproject.toml
│
├── docs/
├── examples/
├── configs/
├── scripts/
├── tests/
│
└── openmedaxis/

    ├── datasets/
    ├── models/
    │   ├── gan/
    │   └── diffusion/
    │
    ├── training/
    ├── evaluation/
    ├── transforms/
    ├── visualization/
    └── utils/
```

---

# Example Vision

The framework aims to make research experiments simple and reproducible.

Example usage (future API):

```python
from openmedaxis.datasets import IXI
from openmedaxis.models.diffusion import DDPM
from openmedaxis.training import Trainer


dataset = IXI(root="data/", modality="T2")
model = DDPM()

trainer = Trainer(model=model, dataset=dataset)
trainer.fit()
```

---

# Roadmap

### Phase 1 — Foundation

* project architecture
* dataset interfaces
* IXI dataset loader
* BraTS dataset loader
* fastMRI dataset loader
* DCGAN baseline
* pix2pix baseline
* DDPM baseline
* basic evaluation metrics

### Phase 2 — Expansion

* CycleGAN and SAGAN
* improved evaluation tools
* visualization utilities
* configuration system

### Phase 3 — Advanced Generative Methods

* latent diffusion
* 3D generative models
* efficient training pipelines
* community benchmark suite

---

# Contributing

OpenMedAxis is at an early stage and contributions are welcome.

Areas where contributions are especially valuable:

* dataset integrations
* generative model implementations
* evaluation metrics
* documentation and examples

If you are interested in contributing, please open an issue to discuss ideas or improvements.

---

# License

This project will be released under an open-source license (MIT or Apache-2.0).

---

# Organization

OpenMedAxis is developed under **AllAxisAI**.

---

# Status

🚧 Pre-alpha — initial framework design and core modules under development.
