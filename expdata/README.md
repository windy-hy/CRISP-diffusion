# DiffCRISP — Figure Reproduction Guide

This directory (`expdata/`) contains analysis materials, plotting scripts, intermediate tables, and/or figure exports associated with the **10 main figures** in the DiffCRISP manuscript:

> **DiffCRISP: Distribution-aware single-cell drug perturbation prediction with conditional latent diffusion**

**Scope.** This README explains the scientific purpose of Figures 1–10, the experimental inputs and measurements required to reproduce their panels, and how the figure folders should be used. It follows the **current 10-figure manuscript**, not historical plot filenames. For model installation, training, checkpoint evaluation, and environment setup, see the [repository README](../README.md).

> **Important reproducibility distinction:** Figure descriptions below are grounded in the manuscript; they do **not** certify that every required preprocessing script, intermediate data file, model checkpoint, or plotting script is already committed. `expdata/` is a collection of figure materials, not yet a verified end-to-end, one-command reproduction pipeline. **Do not claim full reproducibility until the checklist at the end has been verified.**

## 1. Layout and naming conventions

The main figure directory structure is organized by the current manuscript number:

```text
expdata/
├── README.md                  # This guide
├── fig1/                      # Fig. 1: model architecture and workflow
├── fig2/                      # Fig. 2: unseen-cell-type benchmarks
├── fig3/                      # Fig. 3: controlled generator comparison
├── fig4/                      # Fig. 4: distribution recovery / sampling
├── fig5/                      # Fig. 5: drug-responsive subpopulations
├── fig6/                      # Fig. 6: molecular / component ablations
├── fig7/                      # Fig. 7: unseen-drug / joint zero-shot tests
├── fig8/                      # Fig. 8: cross-platform transfer
├── fig9/                      # Fig. 9: breast-cancer pathway prioritization
└── fig10/                     # Fig. 10: hyperparameter sensitivity
```

Each `figN/` may contain lettered panel directories (`a/`, `b/`, etc.), scripts (often `.py`), source tables (`.csv`, `.xlsx`), or exported graphics (`.pdf`, `.png`). **The exact contents are not assumed to be identical across figures.**

**Figure renumbering note.** The `expdata` folders were recently reorganized: material that had been under the earlier `fig5/` was split, and later directories were renumbered to support all ten main manuscript figures. Some files may retain *historical* names (for example, a graphic name referring to an earlier figure number). Treat the **current manuscript panel descriptions below** as authoritative. Rename legacy exports or update plotting paths only after checking their content; never infer the panel solely from a legacy filename.

## 2. Shared requirements and workflow

### 2.1 Environment and model

Set up the repository as described in [`../README.md`](../README.md). Training and inference are implemented mainly in `DIFFCRISP/`. Relevant model implementations include:

- `DIFFCRISP/model.py` — two-stage model and generator selection.
- `DIFFCRISP/ConditionalDiffusion.py` — conditional diffusion and DDIM sampling.
- `DIFFCRISP/DeterministicPredictor.py` — deterministic latent baseline.
- `DIFFCRISP/GaussianPredictor.py` — fixed-variance Gaussian latent baseline.
- `DIFFCRISP/train.py`, `DIFFCRISP/test.py`, `DIFFCRISP/trainer.py` — training, evaluation, and checkpoint loading.
- `experiments/configs/nips.yaml`, `experiments/configs/sci.yaml` — dataset-specific configurations.

The main model described by the manuscript uses **a 128-dimensional expression latent space**, a Stage-I autoencoder followed by a Stage-II conditional latent generator, and **20-step DDIM** inference with DMSO-referenced guidance where applicable. The current source code's *defaults* may not match every reported experiment; **use saved run configurations**, not undocumented defaults.

### 2.2 Datasets and experimental partitions

| Source | Role in the manuscript | Required artifacts |
| --- | --- | --- |
| NeurIPS chemical perturbation benchmark | Unseen-cell-type performance, controlled generator ablations, distribution / responder comparisons, molecular and component ablations; source for cross-platform transfer | Processed expression AnnData, matched control cells, scFM embeddings, drug representations, DEG annotations, split labels |
| SciPlex3 | Cell-line benchmarks, sampling variability, unseen-drug / joint zero-shot settings, MCF7 pathway prioritization | Processed expression AnnData, dose metadata, drug molecular representations, split annotations, observed controls and treated cells |
| PBMC-bench | Sequencing-platform transfer of learned response directions | Control cells with platform/batch and cell-type labels, compatible expression features / embeddings |

Example local input paths used elsewhere in the repository are `data/nips/nips_pp_scFM_resplit.h5ad` and `data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad`; these are **local processed files**, not included/downloaded automatically by this directory. Check the actual paths in the archived experiment configuration.

### 2.3 Recommended figure-reproduction sequence

1. **Fix the code revision:** record `git rev-parse HEAD`, environment, model implementation, dataset version, and figure revision.
2. **Load the intended processed data:** confirm matching gene order, `adata.obs` fields, scFM embedding key, dose conventions, drug mapping, and train/test/OOD split definition.
3. **Load or train the correct checkpoints:** retain the split, training seed, model variant, hyperparameters, molecular features, and DMSO condition used in the reported result.
4. **Generate predictions:** save sufficient model outputs to reconstruct each figure (cell-level profiles when needed; do not retain only a mean profile for distribution-related figures).
5. **Calculate figure-level metrics:** export tidy, inspectable tables with experimental condition identifiers, observed values, predicted values, random seed / replicate identifiers, and metrics.
6. **Run the plotting script:** use the script and input files for each panel; inspect the figure visually and compare numerical annotations with the manuscript.

For the standard **3-split × 3-training-seed** experiments, the manuscript uses `split`, `split2`, and `split3` with training seeds **0, 42, and 123** (nine split–seed evaluations), unless a specific figure describes a different experiment. **Sampling seeds** used to regenerate virtual cells (e.g. ten independent generated populations in Figure 4) are a separate concept from training seeds.

### 2.4 Find available scripts and avoid invalid commands

From the **repository root**, list available plotting scripts rather than assuming a particular `draw.py` is present in every panel:

```powershell
# PowerShell (Windows)
Get-ChildItem .\expdata -Recurse -File -Include *.py,*.ipynb |
  Sort-Object FullName | Select-Object -ExpandProperty FullName
```

```bash
# Bash / Linux / macOS
find expdata -type f \( -name '*.py' -o -name '*.ipynb' \) | sort
```

For a panel that **actually contains** `draw.py` and its required input files, execute from that panel directory if the script uses relative paths:

```powershell
cd expdata/fig5/a
python draw.py
```

This is a **usage pattern**, not a guarantee that all figure panels have a `draw.py` or that the example will run without its recorded inputs. A figure export alone does not establish a reproducible plotting pipeline.

---

## 3. Main figures — contents and reproduction requirements

### Figure 1 — DiffCRISP architecture and two-stage training

**Directory:** [`fig1/`](fig1/)

**Panels:**

- **a — Overall framework:** untreated cells and matched scFM context, drug molecular features and dosage, observed treated-cell targets, conditional latent diffusion, predicted treated populations, and downstream evaluations.
- **b — Training / inference workflow:** Stage I autoencoder learning; Stage II frozen encoder/decoder with a conditioned diffusion target; inference from Gaussian noise using 20 DDIM reverse steps and the frozen gene decoder.

**Source material needed:** the final model architecture, layer dimensions, training-stage separation, guidance design, and a diagram source (e.g. an editable vector/drawing file) if available. This is a **method schematic**, not a numerical benchmark. Its reproduction may consist of regenerating/exporting the editable figure rather than running a model evaluation.

**Check:** schematic must agree with `DIFFCRISP/model.py` and `ConditionalDiffusion.py`, especially 128-dimensional latent, 256-dimensional semantic condition, five residual AdaLN-MLP blocks, and the DDIM inference path described in the manuscript.

### Figure 2 — Benchmarking unseen-cell-type perturbation prediction

**Directory:** [`fig2/`](fig2/)

**Panels:**

- **a:** training / testing schematic for held-out cell types.
- **b:** NeurIPS benchmark versus CRISP, CellOT, scGen, BioLORD, chemCPA, and CPA: top-50-DE-gene perturbation Pearson correlation (`PrΔ`), `R²`, and Sinkhorn distance; nine split–seed evaluations for the reported boxplots.
- **c:** per-split `R²` and `PrΔ` for NeurIPS and SciPlex3.
- **d:** observed versus predicted log fold change for Palbociclib and Idelalisib; distinguish top-50 DE genes and other genes.
- **e:** drug-response direction agreement for six held-out immune cell types, with cell and perturbation counts.
- **f:** cell-level ZFAS1 and SNHG6 expression distributions for Dactolisib-treated CD4+ T cells, comparing control, observed, CRISP, and DiffCRISP populations.

**Source material needed:** matched OOD split/seed evaluation outputs for each baseline, observed and predicted cell-level expressions, gene identifiers / DEG lists, per-group control references, and metadata identifying the cited drug/cell examples. **The original competing-method evaluation outputs must be supplied separately if their inference scripts are absent.**

**Check:** evaluate all methods on the same experimental conditions and top-50-DE-gene definitions. The plot legend / count `m` and `n` must come from the actual evaluation data.

### Figure 3 — Controlled ablation of Stage-II latent generators

**Directory:** [`fig3/`](fig3/)

**Panels:** **a** top-50-DE `R²`; **b** `PrΔ`; **c** Sinkhorn distance; **d** centered Sinkhorn distance; **e** variance log-MAE; **f** responder-proportion error.

**Compared methods:** deterministic latent predictor, fixed-variance Gaussian latent predictor, conditional latent diffusion (DiffCRISP), plus the training-free mean-shift reference.

**Source material needed:** nine matched split–seed evaluations, model-variant label, per-run results for all six metrics, the mean-shift benchmark, and exact test inputs used in the paired Wilcoxon analyses. The experimental controls should share Stage-I autoencoder, cell/drug conditioning, latent dimension, gene decoder, and evaluated splits. The manuscript's controlled generator comparison uses **guidance scale = 1.0**.

**Critical consistency check:** the manuscript specifies **Gaussian sampling `σ = 0.1`**, while a previously inspected `model.py` default used **`fixed_sigma = 0.05`**. The reported Figure 3 must be traced to its actual run configuration and code version; do **not** silently change the manuscript label or claim reproducibility from a different sigma. Report non-significant paired comparisons faithfully.

### Figure 4 — Distribution recovery and sampling calibration

**Directory:** [`fig4/`](fig4/)

**Panels:**

- **a:** NeurIPS three-split comparisons of Sinkhorn distance, centered Sinkhorn distance, and variance log-MAE (mean ± SEM).
- **b:** perturbation-level paired CRISP versus DiffCRISP distribution errors, shown after `log(1+x)` transformation, with win rates and paired Wilcoxon tests.
- **c:** K562/SciPlex3 *between-sampling-seed* variation across perturbation conditions, using pairwise Sinkhorn distances from ten independently sampled predicted populations.
- **d:** relative sampling variation = generated-population between-replicate distance divided by the corresponding observed-treated-population resampling distance. The plotted distribution excludes 2.5th–97.5th-percentile outliers **for visualization only**.
- **e:** pairwise Sinkhorn-distance heatmaps for ten generated populations from K562 cells treated with Cerdulatinib at dose 1.0 (CRISP vs DiffCRISP).

**Source material needed:** cell-level predicted and observed treated populations (not only group means), condition/sample identifiers, Sinkhorn configuration, ten independent virtual-population draws, and observed-cell resampling replicates with matching population sizes.

**Check:** do not conflate different **training seeds** with independent **sampling seeds**; for ten populations there are `10 choose 2 = 45` distinct pairwise comparisons. Disclose any trimming used only in panel d and note distinct heatmap color scales in panel e.

### Figure 5 — Reference-defined drug-responsive subpopulations

**Directory:** [`fig5/`](fig5/)

**Panels:**

- **a:** absolute responder-fraction error across three data splits.
- **b:** CRISP versus DiffCRISP responder-fraction error for matched perturbation conditions.
- **c:** fraction of perturbations where DiffCRISP has the smaller responder-fraction error.
- **d:** response-score distributions for CD4+ T cells treated with Idelalisib or Palbociclib, with observed, CRISP, and DiffCRISP responder percentages and control thresholds.

**Evaluation definition:** form a response direction from the **top 50 genes with the largest observed treatment-associated changes**; project cells onto that direction; classify cells above the **95th percentile of matched-control scores** as responders. Report absolute error in the predicted fraction.

**Source material needed:** true treated and control cells, predicted populations from both methods, per-condition top-50 gene lists / direction, cell-level projected scores, thresholds, and paired statistical outputs.

**Check:** the reference direction uses observed test-treatment expression **only for post hoc evaluation**, not to condition training or inference. This analysis quantifies recovery of a reference-defined response fraction; it is not de novo responder discovery.

### Figure 6 — Molecular representations and component ablations

**Directory:** [`fig6/`](fig6/)

**Panels:**

- **a:** NeurIPS comparison of FCFP4, RDKit2D, and ChemBERTa drug representations using `PrΔ`, `R²` on DE genes, and Sinkhorn distance.
- **b:** full DiffCRISP versus ablations without the Stage-I L1 reconstruction loss and without classifier-free guidance (CFG), across three NeurIPS splits.

**Source material needed:** separate runs / archived evaluation outputs for each drug representation and model ablation, a shared split/seed protocol, representation dimensions and files, checkpoint identifiers, and exact no-L1/no-CFG settings.

**Check:** changing the molecular feature file alone may not be enough if dimensions differ; use the matching checkpoint and input projector configuration. An ablation must vary the intended component while keeping other settings fixed.

### Figure 7 — Unseen-drug prediction and joint zero-shot transfer

**Directory:** [`fig7/`](fig7/)

**Panels:**

- **a:** unseen-drug and unseen-cell-type setup with target-cell controls available at inference.
- **b:** comparison against CRISP and chemCPA on **36 matched held-out drug–dose combinations**. Average first over the **three training seeds within each split**, then over the **three held-out-cell-type splits**; use matched drug–dose units for the reported 95% confidence intervals and paired one-sided tests.
- **c:** stricter joint zero-shot protocol: target cell type absent from both control and perturbed *training populations* and evaluated drugs held out during training.
- **d:** A549, K562, and MCF7 results, displaying observed control–treated MMD magnitude and `PrΔ` comparisons (DiffCRISP versus CRISP).
- **e:** dose-specific zero-shot performance across four doses with reference observed perturbation magnitude.

**Source material needed:** exact unseen-drug split definitions (`split_drugs`-style fields where applicable), unseen-cell-type definitions, molecular features for held-out drugs, control/treated observations, per-condition predictions, MMD and `PrΔ` metrics.

**Check:** distinguish ordinary unseen-drug evaluation with control conditioning from the stricter joint zero-shot experiment. Do not treat the 36 drug–dose combinations as if they were independent training-seed replicates.

### Figure 8 — Transfer across sequencing technologies

**Directory:** [`fig8/`](fig8/)

**Panels:**

- **a:** model trained using NeurIPS perturbations, evaluated with PBMC-bench control cells across sequencing platforms (seen and unseen drugs).
- **b:** UMAP of original PBMC-bench expression representations and DiffCRISP latent/feature space, colored by cell type.
- **c:** seen-drug directional accuracy in NK cells across six sequencing batches.
- **d:** mean directional accuracy for B cells, NK cells, and CD4+ T cells for seen and unseen drugs (95% confidence intervals).
- **e:** unseen-drug NK directional accuracy across the six batches.
- **f:** predicted versus reference gene-level drug response for Dactolisib in NK cells profiled with **10x Chromium v3**, highlighting top-50 DE genes and reporting `PrΔ` and Spearman correlation.

**Source material needed:** NeurIPS-trained checkpoint(s), PBMC-bench platform-labeled untreated cells, compatible scFM representations / gene alignment, drug descriptors, source-reference perturbation responses, per-batch predictions, UMAP embeddings, and direction-accuracy tables.

**Interpretation note:** the documented evaluation assesses **transfer of predicted drug-response directions relative to the NeurIPS perturbation reference**. Do not claim that PBMC-bench contains an independent, measured treated-cell truth for every drug condition unless that specific data is supplied.

### Figure 9 — Pathway-guided therapeutic prioritization

**Directory:** [`fig9/`](fig9/)

**Panels:**

- **a:** normalized pathway enrichment scores (NES) for representative MCF7 predictions, including Toremifene, Fulvestrant, and Lapatinib.
- **b:** drug ranking by agreement with a **predefined signed breast-cancer pathway signature**.
- **c:** Precision@K of ranked candidates versus random ranking/reference prevalence.
- **d:** pathway-level NES and FDR summary for Fulvestrant-treated MCF7 cells.
- **e:** representative GSEA running-enrichment curves for the Breast Cancer Pathway and Cell Cycle programs.

**Source material needed:** MCF7 predicted treated-cell expression, matched controls, ranked gene-level drug-response signatures, exact pathway collections / versions (e.g. MSigDB or Reactome), signed scoring rule, candidate list, ground-truth therapeutic-label rule, enrichment outputs and random-baseline definition.

**Check:** keep **FDR** separate from nominal P value; positive and negative NES indicate opposite enrichment directions. This is retrospective, hypothesis-generating **computational prioritization**, not a clinical efficacy claim.

### Figure 10 — Latent dimension, sampling steps, and guidance sensitivity

**Directory:** [`fig10/`](fig10/)

**Panels:**

- **a:** latent dimensions **32, 64, 128, 256**.
- **b:** DDIM reverse-step counts **5, 10, 20, 30, 50, 100, 200**.
- **c:** classifier-free-guidance scales **1.0, 1.5, 2.0, 3.0**.

**Metrics:** `R²`, `PrΔ`, and Sinkhorn distance. Each plotted point is described in the manuscript as the **mean of three training seeds**.

**Source material needed:** hyperparameter-specific evaluations, the exact baseline experiment and split used for each sweep, checkpoint paths, fixed settings for unswept hyperparameters, and summary tables recording results for every setting and training seed.

**Check:** change **one factor at a time**. The manuscript's standard setting is 128 latent dimensions, 20 DDIM inference steps, and guidance scale 1.5; verify those settings from the saved run records. When comparing guidance, retain consistent sampling settings and the correct DMSO reference condition.

---

## 4. Metrics and statistical conventions

| Metric | Meaning / use | Better |
| --- | --- | --- |
| `R²` on top-50 DE genes | Agreement between predicted and observed treated expression | Higher |
| `PrΔ` | Pearson correlation between observed and predicted *control-subtracted* treatment effects | Higher |
| Sinkhorn distance | Population-level entropic optimal-transport discrepancy | Lower |
| Centered Sinkhorn | Sinkhorn discrepancy after centering each population, emphasizing distribution shape | Lower |
| Variance log-MAE | Mean absolute difference of log gene-wise variances | Lower |
| Responder-fraction error | Absolute difference between predicted and observed reference-defined responder fractions | Lower |
| Directional accuracy | Fraction of selected genes with matching signs of predicted and observed expression changes | Higher |
| Sampling-variation ratio | Predicted between-sampling-population variation relative to observed resampling variation | Closer to 1 |

The actual code should document **which genes, normalizations, sample sizes, Sinkhorn regularization, bootstrap / test implementation, and significance corrections** were used. Do not assume that labels alone define all numerical details.

The manuscript uses paired two-sided Wilcoxon signed-rank comparisons for Figures 3–5 (as specified for the respective panels) and paired one-sided drug–dose-level tests for Figure 7b. Preserve the **correct paired unit** and do not treat individual cells as statistically independent replicates for perturbation-level tests.

## 5. Reproducibility checklist (verify before submission)

For each `figN/`, check the following before describing the corresponding figure as *fully reproducible*:

- [ ] The folder matches the current manuscript's Figure N and all lettered panels.
- [ ] The plotting script or editable source file is present (or a deliberate schematic / static figure is explicitly marked).
- [ ] Required source tables are provided, or clear instructions show how to regenerate them from public data and released scripts.
- [ ] The actual executable command and **working directory** have been tested in a clean environment.
- [ ] Split, training seeds (0/42/123 where relevant), sampling seeds, model/checkpoint IDs, and key hyperparameters are documented.
- [ ] Files created by the command correspond to the final manuscript panel (axes, colors, N, error bars, annotations, and p-values).
- [ ] Any externally sourced baseline outputs or pathway databases are credited, versioned, and obtainable.
- [ ] Sensitive/private data, large accidental checkpoints, local absolute paths, and IDE caches are not committed.

**Project-wide outstanding checks:** reconcile the manuscript's `σ=0.1` for the Figure 3 Gaussian baseline with the currently inspected default `fixed_sigma=0.05`; verify `lat_dim=128` against archived configurations (the public default previously inspected was 64); and verify that batch test scripts accept the arguments they pass before presenting them as runnable end-to-end commands. These are **verification items**, not assertions that the reported paper results are incorrect.

## 6. How to cite and maintain this directory

Use the [main repository README](../README.md) for installation and attribution to CRISP. Associate any released figure table with the version of the associated manuscript, code commit hash, dataset version, and calculation script.

When reorganizing figures (such as splitting a former figure into separate figures), update together:

1. the `expdata/figN/` paths and plotting code's relative filenames;
2. the **main manuscript** figure caption and `\includegraphics` path;
3. this `expdata/README.md` figure-to-panel map;
4. the repository-level README, if it gives a figure index.

Avoid hard-coding file paths or inventing commands that are not present in the repository. If a figure can only be regenerated from intermediate data currently stored locally, explicitly list those missing artifacts rather than claiming one-command reproduction.
