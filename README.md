# DiffCRISP

**Distribution-aware single-cell drug perturbation prediction with conditional latent diffusion**

DiffCRISP is a research implementation for predicting drug-induced single-cell transcriptional responses. It combines **single-cell foundation-model (scFM) representations of matched untreated cells**, **molecular drug features and dose**, and a **conditional latent diffusion model** to generate plausible post-treatment gene-expression profiles.

DiffCRISP builds on the [CRISP](https://github.com/ml4bio/CRISP) framework. In addition to the diffusion generator, this repository contains deterministic and fixed-variance Gaussian latent predictors for controlled generator comparisons.

> **Reproducibility status.** The repository contains research code and experiment configurations, but it is **not yet a fully self-contained release** of all processed datasets, pretrained checkpoints, or figure-generation pipelines. The included launcher scripts currently target particular experiments and must not be assumed to reproduce every figure in the manuscript without checking their settings. See [Reproducibility notes and known issues](#reproducibility-notes-and-known-issues).

## Overview

DiffCRISP uses two stages:

1. **Stage I — gene-expression autoencoder.** Train an encoder and decoder on gene-expression reconstruction for the first 40 epochs, with a loss combining L1, all-gene MSE, and DEG-focused MSE (`AFMSELoss`).
2. **Stage II — conditional latent generation.** Freeze the expression encoder and decoder. Train a generator to predict treated-cell latents from the paired-control scFM context and drug–dose condition. In the diffusion variant, the generator learns to denoise treated-cell latents corrupted by Gaussian noise.

```text
Matched control cells ---> scFM embeddings ---> control projector ---+
                                                                 |
Drug molecular features + log1p(dose) ---> drug projector --------+--> semantic condition
                                                                       |
Observed treated expression ---> expression encoder ---> treated latent z0
                                                                       |
                  Gaussian forward diffusion (training only)           |
                                                                       v
                                             conditional latent denoiser
                                                                       |
Random Gaussian latent --> DDIM reverse sampling --> generated latent
                                                                       |
                                                  frozen gene decoder
                                                                       |
                                             predicted treated cells
```

The conditional diffusion backbone uses five residual AdaLN-style MLP blocks. The code uses 1,000 training diffusion timesteps and 20 DDIM inference steps in the main prediction path.

### DMSO-referenced guidance

At inference, diffusion guidance compares the target drug–dose condition against a **DMSO reference with the same cellular context**. The current implementation identifies observations whose drug name is `dmso` (case-insensitive), then obtains the corresponding drug index and reference dose from the dataset's existing drug mapping.

- `DIFFCRISP/model.py`: `configure_dmso()` resolves the index and dose, while `_get_dmso_semantic_condition()` builds the reference condition.
- `DIFFCRISP/trainer.py`: calls `configure_dmso()` after model initialization and after loading a checkpoint.
- The current implementation requires exactly **one unique `(DMSO index, dose)` pair** across the dataset subsets supplied to it. Missing or ambiguous DMSO conditions raise an error instead of selecting a hard-coded index.

This is **dataset-derived DMSO configuration**, not a fixed NeurIPS or SciPlex3 drug index. When working with different processed datasets or drug mappings, confirm the DMSO condition printed at initialization.

## Generator variants

`DIFFCRISP/model.py` provides three Stage-II alternatives selected through the module-level `GENERATOR_TYPE` setting:

| `GENERATOR_TYPE` | Model | Latent output |
| --- | --- | --- |
| `"diffusion"` | Conditional latent diffusion (DiffCRISP) | DDIM-sampled latent |
| `"deterministic"` | Deterministic latent predictor | One predicted treated latent |
| `"gaussian"` | Fixed-variance Gaussian predictor | Predicted mean plus Gaussian noise |

The variants reuse the same Stage-I gene encoder/decoder and cell–drug conditioning design. The Gaussian branch estimates a conditional latent mean using MSE and adds fixed isotropic Gaussian noise during sampling; **its variance is not learned**.

**Important:** The checked-in model constructor currently sets `fixed_sigma=0.05` for the Gaussian variant. This is a **code default**, not a verified statement of the sigma used to produce every manuscript result. Verify the exact value against the archived Figure 3 experiment before treating a new run as a reproduction.

For controlled generator-only comparisons, use consistent datasets, splits, seeds, conditioning inputs, autoencoder settings, and evaluation configurations. In particular, note that DMSO guidance may change the comparison; use the documented guidance setting from the corresponding experiment.

## Repository structure

```text
CRISP-diffusion/
├── DIFFCRISP/
│   ├── ConditionalDiffusion.py
│   ├── DeterministicPredictor.py
│   ├── GaussianPredictor.py
│   ├── model.py
│   ├── trainer.py
│   ├── data.py
│   ├── embedding.py
│   ├── scFM.py
│   ├── losses.py
│   ├── eval.py
│   ├── train.py
│   ├── test.py
│   ├── run_all.py
│   └── test_all.py
├── experiments/
│   └── configs/
│       ├── nips.yaml
│       └── sci.yaml
├── expdata/                 # available experiment/figure materials
├── requirement.txt
├── setup.py
├── LICENSE.txt
└── README.md
```

Local `data/` and `model/` directories are used for inputs and generated checkpoints. They are **not** prerequisites for cloning the repository; create or populate them locally as needed. The exact set of experiment artifacts included in `expdata/` may change as the project is prepared for release.

## Environment and installation

```bash
git clone https://github.com/windy-hy/CRISP-diffusion.git
cd CRISP-diffusion

conda create -n diffcrisp python=3.9
conda activate diffcrisp

pip install -r requirement.txt
pip install -e .
```

The pinned dependencies include PyTorch, Scanpy, AnnData, RDKit, optimal-transport-related libraries, and scientific Python tools. Use a PyTorch installation compatible with your local CUDA runtime. The current data collation code contains CUDA-specific tensor placement and is **not documented as CPU-only compatible**.

## Data inputs

Training requires **processed AnnData** (`.h5ad`) files with predefined data splits and **precomputed drug embeddings** (`.parquet`). The scripts do not automatically download or reconstruct these inputs from the original raw studies.

The current script conventions include the following local paths:

| Dataset | Processed AnnData | Drug embeddings |
| --- | --- | --- |
| NeurIPS chemical perturbation benchmark | `data/nips/nips_pp_scFM_resplit.h5ad` | `data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet` |
| SciPlex3 | `data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad` | `data/drug_embeddings/fcfp4_1024_embedding_lincs_sciplex3.parquet` |

These are **expected local filenames**, not download URLs. Inspect the respective `experiments/configs/nips.yaml` and `experiments/configs/sci.yaml` for dataset-specific AnnData column names.

The processed AnnData must include, as appropriate:

- Gene-expression matrix `adata.X` and gene identifiers.
- Drug, dose, cell-type, DMSO/control, pairing, and split metadata in `adata.obs`.
- The selected scFM embedding in `adata.obsm`, e.g., `X_scGPT` or `X_scGPT_human`.
- Differential-expression annotations under the configured `adata.uns` key.
- Split annotations matching the value supplied through `--split` (for example, `split`, `split2`, `split3`, or the drug-holdout split names).

The `--FM_key` argument must match an existing key in `adata.obsm`, and its feature dimension must be compatible with the model configuration and checkpoints.

## Training

The training script uses paths relative to the `DIFFCRISP/` directory. From the repository root:

```bash
cd DIFFCRISP
```

### Single training run (NeurIPS example)

The following command illustrates the public training entry point, assuming the stated local input files exist:

```bash
python train.py \
  --config ../experiments/configs/nips.yaml \
  --split split \
  --seed 0 \
  --savedir ../model/nips/example/split/seed0 \
  --FM_key X_scGPT \
  --drug_emb ../data/drug_embeddings/fcfp4_1024_embedding_lincs_nips.parquet \
  --data_path ../data/nips/nips_pp_scFM_resplit.h5ad \
  --mmd_co 0.0
```

The script saves resolved arguments in `config.yaml`, writes training logs to `log.txt`, and saves checkpoint files into the output directory according to the trainer's checkpoint logic.

**MMD note:** In the current `model.py`, the implemented Stage-I optimization objective is

\[
\mathcal{L}_{AE}=0.5\mathcal{L}_{1}+0.5\mathcal{L}_{MSE}+0.2\mathcal{L}_{AFMSE}.
\]

The MMD term is not part of the optimized loss in this code revision. The `--mmd_co` flag remains in the CLI for compatibility; setting it does **not** enable MMD regularization in this implementation.

### Batch training

```bash
python run_all.py
```

**Check the script before use.** At the inspected revision, `run_all.py` enables **SciPlex3**, the `split_drugs`, `split_drugs2`, and `split_drugs3` columns, and training seeds **`0`, `42`, and `123`**. That is nine intended runs for the one enabled dataset, **not** the standard unseen-cell-type `split`/`split2`/`split3` grid. The launcher has experiment-specific output paths and an `FM_key` setting. Edit them to match the particular experiment being reproduced.

### Latent dimension

The checked-in `model.py` sets **64 as the default `lat_dim`** if no explicit model hyperparameter override is supplied. The research manuscript discusses experiments using a **128-dimensional** latent space. Accordingly, the default command above is an **execution example**, not a validated reproduction of the manuscript's 128-dimensional configuration. Use the archived run configuration/checkpoint to determine the setting used for each reported result. Do not assume that changing a filename or output directory changes the model dimension.

## Evaluation

For a saved checkpoint directory, the public `test.py` entry point supports the following command:

```bash
python test.py \
  --config ../experiments/configs/nips.yaml \
  --split split \
  --seed 0 \
  --FM_key X_scGPT \
  --data_path ../data/nips/nips_pp_scFM_resplit.h5ad \
  --model_dir ../model/nips/example/split/seed0
```

The script iterates over `model_*.pt` checkpoints in the supplied directory and writes a text file containing the DEG-focused R², perturbation-specific Pearson correlation, and Sinkhorn distance. Evaluation requires a compatible pretrained checkpoint, processed dataset, split, and scFM representation.

> **Batch evaluation caveat:** The checked-in `test_all.py` passes a `--ddim` argument to `test.py`, whereas the current `test.py` argument parser does **not** accept that option. Therefore, `python test_all.py` is **not a verified working batch-evaluation command in this revision**. Reconcile these two scripts before running batch evaluation; do not interpret the launcher's completion message as proof that all experiments succeeded.

## Reproducing the manuscript

The manuscript includes analyses of unseen cell types, unseen drugs, controlled generator ablations, distribution fidelity, responder proportions, cross-platform response-direction transfer, pathway enrichment, and hyperparameter sensitivity. The relevant scripts and output data are not currently organized into a tested, one-command workflow for each manuscript figure.

For reproducibility, retain the following for each experiment: code commit ID, processed dataset version, molecular embedding file, split name, random seed, model hyperparameters, guidance scale, Gaussian noise level (where applicable), checkpoint, number of generated cells, and evaluation configuration.

Record the code revision with:

```bash
git rev-parse HEAD
```

## Reproducibility notes and known issues

1. **Paper/configuration consistency:** Confirm the actual latent dimension used in the reported experiments. Current code defaults to 64; the manuscript describes a 128-dimensional model.
2. **Gaussian baseline consistency:** The checked-in Gaussian constructor uses `fixed_sigma=0.05`. Confirm the sigma used to generate the reported controlled-ablation results, which may differ from this code default.
3. **DMSO matching:** DMSO index and dose are now looked up from the loaded dataset rather than specified as fixed integers. The code assumes exactly one unique DMSO drug–dose pair across the supplied subsets and will fail if it is missing or non-unique. Validate this assumption for each dataset.
4. **Batch evaluation:** The `--ddim` argument mismatch between `test_all.py` and `test.py` must be corrected before batch evaluation works as advertised.
5. **Experiment-specific launchers:** `run_all.py` and `test_all.py` have hard-coded experiment choices and output paths. Review these for the experiment you intend to run.
6. **Checkpoint compatibility:** Checkpoints encode the model variant and some initialization settings. Use the corresponding compatible code and dataset/drug mapping when loading historical models.
7. **Package metadata:** `setup.py` currently retains the upstream package name `CRISP` rather than `DiffCRISP`; this is a naming/packaging detail to review before a formal software release.
8. **Artifacts:** The repository does not currently include a verified complete pipeline for downloading/preprocessing every dataset and reconstructing every paper figure.

These notes document the **inspected public implementation**. They are not claims that every manuscript experiment used exactly the present default settings.

## Acknowledgments and citation

This repository builds on the original [CRISP project](https://github.com/ml4bio/CRISP). If you use DiffCRISP or build on its CRISP components, please cite the corresponding original work and follow the license terms of the upstream implementation and external resources.

- **CRISP:** *Predicting drug responses of unseen cell types through transfer learning with foundation models*, *Nature Computational Science*. [DOI: 10.1038/s43588-025-00887-6](https://doi.org/10.1038/s43588-025-00887-6).
- **DiffCRISP:** Citation information will be added when the corresponding manuscript has a public bibliographic record.

## License

See [`LICENSE.txt`](LICENSE.txt). Third-party pretrained models, data, and molecular representations may have separate licensing conditions.
