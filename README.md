# DiffCRISP

**DiffCRISP** is a research implementation for **single-cell drug perturbation response prediction** built on top of the CRISP framework.  
The main change is to replace direct perturbation-state decoding with a **conditional latent diffusion model**, so that post-perturbation cellular states are generated in a learned low-dimensional gene-expression space.

> **Status:** research code under active development. The current repository mainly supports experiments on the NeurIPS single-cell perturbation dataset, with SciPlex3-related configuration and code also included.

## Overview

CRISP uses single-cell foundation-model (scFM) representations of control cells together with drug information to predict perturbation responses. DiffCRISP keeps this general idea, but models the perturbation process as **conditional generation in latent space**.

```text
Control-cell scFM embedding ──> projection ──┐
                                            ├──> semantic condition z_sem
Drug embedding + dosage ──────> projection ─┘
                                                │
Perturbed expression ──> gene encoder ──> z0 ──┤
                                                ↓
                                     Conditional Latent Diffusion
                                                ↓
                                      generated latent state
                                                ↓
                                         gene decoder
                                                ↓
                                predicted perturbed expression
```

## Main Differences from CRISP

### 1. Latent diffusion for perturbation modeling

Instead of directly decoding a concatenated control/drug latent representation into the perturbed expression profile, DiffCRISP learns a **conditional diffusion process** in the gene-expression latent space.

### 2. Two-stage training

**Stage I — Gene autoencoder**

- Encode the observed perturbed gene-expression profile into a low-dimensional latent representation.
- Reconstruct gene expression with the gene decoder.
- The current reconstruction objective combines MSE, L1, and AFMSE losses.
- The current implementation uses a 40-epoch autoencoder warm-up.

**Stage II — Conditional latent diffusion**

- Freeze the gene encoder and decoder.
- Encode the target perturbed state into latent space.
- Train the diffusion model to predict the noise added to the latent representation.
- Optimize the diffusion network together with the cellular-context and drug-condition projection layers.

### 3. scFM-conditioned cellular context

Control-cell representations from a single-cell foundation model are projected into the latent conditioning space.

The AnnData object should contain the requested scFM embedding in `adata.obsm`, for example:

```text
X_scGPT
X_scGPT_human
```

The exact key is controlled by `--FM_key`.

### 4. Drug and dosage conditioning

Drug embeddings are combined with dosage information before projection into the conditioning space. Dosage is transformed with `log1p`, and the projected drug representation is concatenated with the projected control-cell representation:

```text
z_sem = [z_control, z_drug]
```

### 5. DMSO-based guidance

During inference, the conditional branch represents the target treatment while the reference branch uses the DMSO control condition. The generated perturbation signal is guided using the difference between these two branches.

### 6. DDIM inference

The model supports DDPM-style generation and accelerated **DDIM sampling**. The current prediction path uses DDIM sampling to reduce the reverse process from 1000 diffusion steps to a much smaller number of sampling steps.

## Repository Structure

```text
CRISP-diffusion/
├── DIFFCRISP/
│   ├── ConditionalDiffusion.py   # conditional diffusion and DDIM/DDPM sampling
│   ├── model.py                  # gene autoencoder, condition encoders, DiffCRISP model
│   ├── trainer.py                # training loop
│   ├── train.py                  # single-run training entry point
│   ├── run_all.py                # batch experiments over splits/seeds
│   ├── data.py                   # dataset loading utilities
│   ├── embedding.py              # drug embedding utilities
│   ├── scFM.py                   # single-cell foundation-model utilities
│   ├── losses.py                 # training losses
│   ├── eval.py                   # evaluation utilities
│   ├── test.py
│   └── test_all.py
├── data/
├── experiments/
│   ├── configs/
│   │   ├── nips.yaml
│   │   └── sci.yaml
│   ├── scripts/
│   └── results/
├── expdata/
├── model/
├── requirement.txt
└── setup.py
```

## Installation

```bash
git clone https://github.com/windy-hy/CRISP-diffusion.git
cd CRISP-diffusion

conda create -n diffcrisp python=3.9
conda activate diffcrisp

pip install -r requirement.txt
pip install -e .
```

Make sure the installed PyTorch version is compatible with your CUDA environment.

## Data Preparation

Large `.h5ad` files and model checkpoints are not stored in this repository.

For the current NeurIPS workflow, the training code expects files equivalent to:

```text
data/nips/nips_pp_scFM_resplit.h5ad
data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet
experiments/configs/nips.yaml
```

The AnnData object must contain the scFM embedding specified by `--FM_key`.

## Training

Because the current default paths are relative to the `DIFFCRISP/` directory, the safest way to run the training entry point is:

```bash
cd DIFFCRISP
```

Example:

```bash
python train.py \
    --config ../experiments/configs/nips.yaml \
    --split split \
    --seed 0 \
    --savedir ../model/nips/example/split/seed0 \
    --FM_key X_scGPT_human \
    --drug_emb ../data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet \
    --data_path ../data/nips/nips_pp_scFM_resplit.h5ad \
    --mmd_co 0.0001
```

For multiple splits and random seeds:

```bash
python run_all.py
```

Before running batch experiments, check the paths, `FM_key`, split list, seeds, and output directory defined in `run_all.py`.

## Current Model Configuration

| Component | Current setting |
|---|---|
| Gene latent dimension | 128 |
| Gene encoder hidden dims | 2048 → 1024 → 512 |
| Gene decoder hidden dims | 512 → 1024 → 2048 |
| Diffusion steps | 1000 |
| Diffusion backbone | MLP |
| Diffusion hidden dimension | 4 × latent dimension |
| Semantic condition | control-cell latent + drug/dose latent |
| Autoencoder optimizer | AdamW |
| Diffusion optimizer | AdamW |
| Autoencoder learning rate | 5e-4 |
| Diffusion learning rate | 1e-4 |
| AE warm-up | 40 epochs |
| Default prediction guidance scale | 1.5 |
| Default DDIM sampling | 20 steps |

## Evaluation

Evaluation and experiment-analysis scripts are provided under:

```text
DIFFCRISP/eval.py
DIFFCRISP/test.py
DIFFCRISP/test_all.py
experiments/
expdata/
```

The repository is currently being reorganized, so experiment-specific paths may need to be adjusted before reproduction.

## Relationship to CRISP

DiffCRISP is developed from the open-source **CRISP** framework:

- Original repository: https://github.com/ml4bio/CRISP
- Paper: *Predicting drug responses of unseen cell types through transfer learning with foundation models*
- DOI: https://doi.org/10.1038/s43588-025-00887-6

CRISP provides the foundation-model-assisted framework for cell-type-aware perturbation prediction. DiffCRISP explores a conditional latent diffusion formulation for modeling the distribution of post-perturbation cellular states while retaining the unseen-cell-type / unseen-drug generalization setting.

## Citation

If you use the original CRISP framework, please cite:

```bibtex
@article{wang2025crisp,
  title   = {Predicting drug responses of unseen cell types through transfer learning with foundation models},
  author  = {Wang, Yixuan and Liu, Xinyuan and Fan, Yimin and others},
  journal = {Nature Computational Science},
  year    = {2025},
  doi     = {10.1038/s43588-025-00887-6}
}
```

A citation for DiffCRISP can be added when the corresponding work is publicly available.

## License

This repository retains the license included in the project. Please also follow the licensing and attribution requirements of CRISP, external datasets, and pretrained models used by the project.

## Notes

DiffCRISP is currently research-oriented rather than a polished software package. Some dataset-specific constants and paths remain in the experimental code. Reproducibility scripts, pretrained checkpoints, and a more complete data-download guide can be added as the project is cleaned up.
