"""
Memory-light replacement for CRISP/data.py.

Main changes:
1. Keep gene expression X sparse / numpy in Dataset.__init__ instead of converting full X to dense torch.Tensor.
2. Convert only one row to dense torch.Tensor inside __getitem__.
3. Store DEG masks as group-level masks, not cell-level n_cells x n_genes masks.
4. Store paired-control FM embeddings as group means, not a full n_cells x FM_dim copy.

Drop this file into CRISP/data.py after backing up the original.
"""

import logging
from typing import List, Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from sklearn.preprocessing import OneHotEncoder

from DIFFCRISP.utils import canonicalize_smiles, sample_neg


indx = lambda a, i: a[i] if a is not None else None


def get_group_idx(data, pert_group_key):
    group_name = data.obs[pert_group_key].values
    unique_group_name = np.unique(group_name)
    unique_group_name_to_idx = dict(zip(unique_group_name, range(len(unique_group_name))))
    group_idx = torch.tensor([unique_group_name_to_idx[i] for i in group_name], dtype=torch.long)
    return unique_group_name_to_idx, group_idx


def get_degs(data, pert_group_key, de_genes, var_names):
    """
    Original cell-level DEG mask builder. Kept for compatibility, but Dataset below
    uses build_deg_masks_by_group() to avoid n_cells x n_genes memory blow-up.
    """
    number_idx = np.array(range(len(data)))
    degs = torch.zeros((data.shape), dtype=torch.bool)
    for k, sub_degenes in de_genes.items():
        adata_sub_index = number_idx[data.obs[pert_group_key] == k]
        degs[adata_sub_index, :] = torch.tensor(var_names.isin(sub_degenes), dtype=torch.bool)
    return degs


def build_deg_masks_by_group(data, pert_group_key, de_genes, var_names):
    """
    Memory-light DEG mask:
    - old: n_cells x n_genes
    - new: n_groups x n_genes + group_idx per cell
    """
    group_names = np.array(data.obs[pert_group_key].values)
    unique_groups = np.array(sorted(set(group_names)))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    group_idxs = torch.tensor([group_to_idx[g] for g in group_names], dtype=torch.long)

    deg_masks = torch.zeros((len(unique_groups), len(var_names)), dtype=torch.bool)
    if de_genes is not None:
        for g, sub_degenes in de_genes.items():
            if g in group_to_idx:
                deg_masks[group_to_idx[g], :] = torch.tensor(var_names.isin(sub_degenes), dtype=torch.bool)

    return group_to_idx, group_idxs, deg_masks


def get_groups(obs_df, covnames: str, split: str, control_key):
    obs_df["pc_cov_split"] = obs_df[covnames].astype(str) + "_" + obs_df[split].astype(str)
    obs_control = obs_df[obs_df[control_key] == 1]
    grouped_df_control = obs_control.groupby("pc_cov_split")
    return grouped_df_control


def get_paired_mean(obs_df, data, control_key, pc_cov, split_key, calc="mean", keep_ctrl=True):
    """
    Original-style helper retained for compatibility. The Dataset below does not
    call this function for paired FM embeddings; it calls build_paired_fm_lookup().
    """
    grouped_df_control = get_groups(obs_df, pc_cov, split_key, control_key)
    index_to_num = dict(zip(obs_df.index, range(len(obs_df))))

    def get_mean(group):
        idxnum = [index_to_num[i] for i in group.index]
        sub_x = data[idxnum]
        if sp.issparse(sub_x):
            mean_val = np.asarray(sub_x.mean(axis=0)).squeeze()
            return torch.tensor(mean_val, dtype=torch.float32)
        if isinstance(sub_x, torch.Tensor):
            return torch.mean(sub_x, dim=0)
        return torch.tensor(np.asarray(sub_x).mean(axis=0), dtype=torch.float32)

    def get_std(group):
        idxnum = [index_to_num[i] for i in group.index]
        sub_x = data[idxnum]
        if sp.issparse(sub_x):
            mean_sq = np.asarray(sub_x.power(2).mean(axis=0)).squeeze()
            sq_mean = np.asarray(sub_x.mean(axis=0)).squeeze() ** 2
            std_val = np.sqrt(np.maximum(mean_sq - sq_mean, 0))
            return torch.tensor(std_val, dtype=torch.float32)
        if isinstance(sub_x, torch.Tensor):
            return torch.std(sub_x, dim=0)
        return torch.tensor(np.asarray(sub_x).std(axis=0), dtype=torch.float32)

    grouped_mean = grouped_df_control.apply(get_mean if calc == "mean" else get_std)
    group_dict = dict(grouped_mean)
    keys = group_dict.keys()

    if keep_ctrl:
        if sp.issparse(data):
            paired = np.empty(data.shape[0], dtype=object)
            valid_barcodes = [
                i for i in obs_df[obs_df[control_key] == 0].index
                if obs_df.loc[i, "pc_cov_split"] in keys
            ]
            treated_index = [index_to_num[i] for i in valid_barcodes]
            for barcode, idx in zip(valid_barcodes, treated_index):
                paired[idx] = group_dict[obs_df.loc[barcode, "pc_cov_split"]]
        else:
            if isinstance(data, torch.Tensor):
                paired = data.clone()
            else:
                paired = torch.tensor(np.asarray(data), dtype=torch.float32)
            valid_barcodes = [
                i for i in obs_df[obs_df[control_key] == 0].index
                if obs_df.loc[i, "pc_cov_split"] in keys
            ]
            treated_index = [index_to_num[i] for i in valid_barcodes]
            for barcode, idx in zip(valid_barcodes, treated_index):
                paired[idx, :] = group_dict[obs_df.loc[barcode, "pc_cov_split"]]
    else:
        valid_barcodes = [i for i in obs_df.index if obs_df.loc[i, "pc_cov_split"] in keys]
        choice_control_mean = [group_dict[obs_df.loc[i, "pc_cov_split"]] for i in valid_barcodes]
        paired = torch.tensor(np.stack(choice_control_mean, axis=0))

    return paired


def build_paired_fm_lookup(obs_df, fm_emb_np, control_key, pc_cov, split_key):
    """
    Build low-memory paired-control lookup.

    Original code creates a full n_cells x FM_dim paired_cell_embeddings matrix.
    This version stores only control-group means and returns the right vector in
    __getitem__.
    """
    obs_df = obs_df.copy()
    obs_df["pc_cov_split"] = obs_df[pc_cov].astype(str) + "_" + obs_df[split_key].astype(str)
    index_to_num = dict(zip(obs_df.index, range(len(obs_df))))

    group_mean = {}
    control_obs = obs_df[obs_df[control_key] == 1]
    for key, group in control_obs.groupby("pc_cov_split"):
        idx = [index_to_num[i] for i in group.index]
        group_mean[key] = torch.tensor(fm_emb_np[idx].mean(axis=0), dtype=torch.float32)

    pc_cov_split = np.array(obs_df["pc_cov_split"].values)
    is_control = np.array(obs_df[control_key].values) == 1
    return group_mean, pc_cov_split, is_control


def row_to_tensor(X, i, dtype=torch.float32):
    """Convert one expression row to dense torch.Tensor."""
    x = X[i]
    if sp.issparse(x):
        x = x.toarray()
    elif hasattr(x, "A"):
        x = x.A
    x = np.asarray(x).squeeze()
    return torch.tensor(x, dtype=dtype)




class LazyMatrix:
    """
    Small compatibility wrapper.
    It exposes .shape like a matrix but does not materialize the full matrix.
    Only indexed rows are converted on demand.
    """
    def __init__(self, dataset, indices, kind):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=int)
        self.kind = kind
        if kind == "gene":
            self.shape = (len(self.indices), dataset.num_genes)
        elif kind == "paired":
            self.shape = (len(self.indices), dataset.fm_dim)
        elif kind == "deg":
            self.shape = (len(self.indices), dataset.num_genes)
        else:
            raise ValueError(f"Unknown LazyMatrix kind: {kind}")

    def __len__(self):
        return len(self.indices)

    def _get_one(self, local_i):
        global_i = int(self.indices[int(local_i)])
        if self.kind == "gene":
            return self.dataset.get_gene(global_i)
        if self.kind == "paired":
            return self.dataset.get_paired_fm(global_i)
        if self.kind == "deg":
            return self.dataset.get_deg(global_i)
        raise ValueError(self.kind)

    def _normalize_rows(self, rows):
        """Normalize row indexers used by code such as genes[idx_all, :]."""
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().numpy()

        if isinstance(rows, slice):
            return list(range(*rows.indices(len(self.indices))))

        if isinstance(rows, (list, tuple, np.ndarray)):
            arr = np.asarray(rows)
            if arr.dtype == bool:
                arr = np.where(arr)[0]
            return arr.reshape(-1).astype(int).tolist()

        return int(rows)

    def __getitem__(self, i):
        # Support matrix-like indexing used in eval.py, e.g. genes_true[idx_all, :]
        col_index = None
        if isinstance(i, tuple):
            rows = i[0]
            col_index = i[1] if len(i) > 1 else None
        else:
            rows = i

        rows = self._normalize_rows(rows)

        if isinstance(rows, list):
            if len(rows) == 0:
                out = torch.empty((0, self.shape[1]), dtype=torch.float32)
            else:
                out = torch.stack([self._get_one(j) for j in rows], dim=0)
        else:
            out = self._get_one(rows)

        # Usually col_index is ':'; keep this for compatibility with [:, gene_idx].
        if col_index is not None:
            if isinstance(col_index, slice) and col_index == slice(None, None, None):
                return out
            if out.dim() == 1:
                return out[col_index]
            return out[:, col_index]

        return out


def get_cov_gpt(adata, data, control_key, cov_name, split_key, setting="train"):
    adata_control = adata[(adata.obs[control_key] == 1) & (adata.obs[split_key] == setting)]
    grouped_adata_control = adata_control.obs.groupby(cov_name)
    index_to_num = dict(zip(adata.obs.index, range(len(adata))))

    def get_mean(group):
        idxnum = [index_to_num[i] for i in group.index]
        sub_x = data[idxnum]
        if sp.issparse(sub_x):
            return torch.tensor(np.asarray(sub_x.mean(axis=0)).squeeze(), dtype=torch.float32)
        if isinstance(sub_x, torch.Tensor):
            return torch.mean(sub_x, axis=0)
        return torch.tensor(np.asarray(sub_x).mean(axis=0), dtype=torch.float32)

    grouped_mean = grouped_adata_control.apply(get_mean)
    group_mean_dict = dict(grouped_mean)
    return group_mean_dict


def drug_names_to_once_canon_smiles(
    drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
):
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in dataset.obs.groupby([perturbation_key, smiles_key]).groups.keys()
    }
    return [name_to_smiles_map[name] for name in drug_names]


def drug_to_idx(drugs_names):
    drugs_names_unique = set()
    for d in drugs_names:
        [drugs_names_unique.add(i) for i in d.split("+")]

    drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))
    _drugs_name_to_idx = {smiles: idx for idx, smiles in enumerate(drugs_names_unique_sorted)}
    drugs_idx = [_drugs_name_to_idx[drug] for drug in drugs_names]

    return drugs_idx, drugs_names_unique_sorted, _drugs_name_to_idx


class Dataset:
    covariate_keys: Optional[List[str]]
    drugs: torch.Tensor
    drugs_idx: torch.Tensor
    max_num_perturbations: int
    dosages: torch.Tensor
    drugs_names_unique_sorted: np.ndarray

    def __init__(
        self,
        data,
        perturbation_key=None,
        dose_key=None,
        celltype_key="cell_type",
        covariate_keys=None,
        smiles_key="SMILES",
        FM_key="X_scGPT",
        degs_key="rank_genes_groups_cov",
        pert_category="cov_drug_name",
        control_key="control",
        split_key="split",
        pc_cov="type_donor",
        seed=0,
        use_FM=True,
    ):
        logging.info(f"Starting to read in data: {data}\n...")
        if isinstance(data, AnnData):
            data = data
        else:
            data = sc.read(data)
        logging.info("Finished data loading.")

        # Expression matrix: do NOT convert whole matrix to dense torch.Tensor.
        if sp.issparse(data.X):
            logging.info("Detected sparse matrix. Keeping X as scipy CSR matrix.")
            self.genes_X = data.X.tocsr().astype(np.float32)
        else:
            logging.info("Detected dense matrix. Keeping X as numpy float32 array.")
            self.genes_X = np.asarray(data.X, dtype=np.float32)

        self.n_obs = data.shape[0]
        self.var_names = data.var_names
        self.num_genes = data.shape[1]

        # FM embedding is low-dimensional; keep as numpy to avoid extra torch copies.
        if use_FM:
            self.FM_emb_np = np.asarray(data.obsm[FM_key], dtype=np.float32)
        else:
            # Not recommended for large data, but kept for compatibility.
            if sp.issparse(data.X):
                self.FM_emb_np = data.X.toarray().astype(np.float32)
            else:
                self.FM_emb_np = np.asarray(data.X, dtype=np.float32)

        self.fm_dim = int(self.FM_emb_np.shape[1])

        self.control_key = control_key
        obs_df = data.obs.copy()

        if "cov_drug_name" not in data.obs.columns:
            data.obs["cov_drug_name"] = data.obs[celltype_key].astype(str) + "_" + data.obs[perturbation_key].astype(str)
            obs_df["cov_drug_name"] = data.obs["cov_drug_name"]

        # Store group-level paired-control FM means instead of n_cells x FM_dim matrix.
        self.paired_group_mean, self.pc_cov_split, self.is_control = build_paired_fm_lookup(
            obs_df, self.FM_emb_np, control_key, pc_cov, split_key
        )

        # Negative sample indices. Keep original behavior.
        self.neg_idx = sample_neg(data, split_key, "cov_drug_name", perturbation_key, seed)

        self.celltype = np.array(data.obs[celltype_key].values)
        self.unique_celltype = np.array(sorted(set(self.celltype)))
        self.celltype_to_idx_full = {ct: idx for idx, ct in enumerate(self.unique_celltype)}
        self.celltype_idx_full = torch.tensor(
            [self.celltype_to_idx_full[ct] for ct in self.celltype], dtype=torch.long
        )

        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys

        if perturbation_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key'({perturbation_key})."
                )
            self.pert_categories = np.array(data.obs[pert_category].values)
            self.de_genes = data.uns[degs_key]
            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            drugs_idx, self.drugs_names_unique_sorted, _ = drug_to_idx(self.drugs_names)
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), data, perturbation_key, smiles_key
            )

            self.drugs_idx = torch.tensor(drugs_idx, dtype=torch.long)
            self.dosages = torch.tensor([float(d) for d in self.dose_names], dtype=torch.float32)
        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.canon_smiles_unique_sorted = None
            self.drugs_idx = None
            self.dosages = None

        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self.covariates = []
            for cov in covariate_keys:
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names_unique = self.covariate_names_unique[cov]
                try:
                    encoder_cov = OneHotEncoder(sparse=False)
                except TypeError:
                    encoder_cov = OneHotEncoder(sparse_output=False)
                encoder_cov.fit(names_unique.reshape(-1, 1))

                names = self.covariate_names[cov]
                self.covariates.append(torch.tensor(encoder_cov.transform(names.reshape(-1, 1)), dtype=torch.float32))
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.covariates = None

        if self.covariates is not None:
            self.num_covariates = [len(names) for names in self.covariate_names_unique.values()]
        else:
            self.num_covariates = [0]

        self.num_drugs = len(self.drugs_names_unique_sorted) if self.drugs_names_unique_sorted is not None else 0

        # Group-level DEG masks, not n_cells x n_genes.
        self.unique_group_name_dict, self.group_idxs, self.deg_masks = build_deg_masks_by_group(
            data, pert_category, self.de_genes, self.var_names
        )

        # Compatibility with original trainer code. These objects expose .shape
        # but read rows lazily instead of materializing full matrices.
        self.genes = LazyMatrix(self, np.arange(self.n_obs), "gene")
        self.paired_cell_embeddings = LazyMatrix(self, np.arange(self.n_obs), "paired")
        self.degs = LazyMatrix(self, np.arange(self.n_obs), "deg")

        self.indices = {
            "all": list(range(self.n_obs)),
            "control": np.where(data.obs[control_key] == 1)[0].tolist(),
            "treated": np.where(data.obs[control_key] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

    def get_gene(self, i):
        return row_to_tensor(self.genes_X, i)

    def get_fm(self, i):
        return torch.tensor(self.FM_emb_np[i], dtype=torch.float32)

    def get_paired_fm(self, i):
        if self.is_control[i]:
            return self.get_fm(i)
        key = self.pc_cov_split[i]
        if key in self.paired_group_mean:
            return self.paired_group_mean[key]
        # Fallback: own FM embedding. Do NOT fallback to genes; dimensions may differ.
        return self.get_fm(i)

    def get_deg(self, i):
        return self.deg_masks[self.group_idxs[i]]

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx, split)

    def __getitem__(self, i):
        genes_i = self.get_gene(i)
        paired_emb_i = self.get_paired_fm(i)
        deg_i = self.get_deg(i)

        if self.covariates is None:
            return (
                genes_i,
                paired_emb_i,
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                deg_i,
                indx(self.celltype_idx_full, i),
                indx(self.group_idxs, i),
                None,
            )
        else:
            return (
                genes_i,
                paired_emb_i,
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                deg_i,
                indx(self.celltype_idx_full, i),
                indx(self.group_idxs, i),
                *[indx(cov, i) for cov in self.covariates],
            )

    def __len__(self):
        return self.n_obs


class SubDataset:
    """
    Memory-light SubDataset.
    It keeps only global indices and reads rows lazily from Dataset.
    """

    def __init__(self, dataset: Dataset, indices, split_set):
        self.dataset = dataset
        self.indices = np.array(indices, dtype=int)
        self.split_set = split_set

        # Compatibility with original trainer code: trainer may access
        # self.datasets["training"].paired_cell_embeddings.shape[1].
        # These wrappers do not allocate full matrices.
        self.genes = LazyMatrix(dataset, self.indices, "gene")
        self.paired_cell_embeddings = LazyMatrix(dataset, self.indices, "paired")
        self.degs = LazyMatrix(dataset, self.indices, "deg")

        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.canon_smiles_unique_sorted = dataset.canon_smiles_unique_sorted

        self.drugs_idx = indx(dataset.drugs_idx, self.indices)
        self.dosages = indx(dataset.dosages, self.indices)
        self.drugs_names = indx(dataset.drugs_names, self.indices)
        self.pert_categories = indx(dataset.pert_categories, self.indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

        self.group_idxs = indx(dataset.group_idxs, self.indices)

        self.celltype = indx(dataset.celltype, self.indices)
        self.unique_celltype = np.array(sorted(set(self.celltype)))
        self.num_celltypes = len(self.unique_celltype)
        self.celltype_to_idx = {ct: idx for idx, ct in enumerate(self.unique_celltype)}
        self.celltype_idx = torch.tensor([self.celltype_to_idx[ct] for ct in self.celltype], dtype=torch.long)

        if dataset.covariates is not None:
            self.covariates = [indx(cov, self.indices) for cov in dataset.covariates]
        else:
            self.covariates = None

        if self.covariate_keys is not None:
            self.covariate_names = {}
            for cov in self.covariate_keys:
                self.covariate_names[cov] = indx(dataset.covariate_names[cov], self.indices)
        else:
            self.covariate_names = None

        if split_set == "train":
            self.neg_idx = dataset.neg_idx
            if isinstance(self.neg_idx, torch.Tensor):
                self.neg_idx = self.neg_idx.cpu().numpy()
            self.neg_idx = np.asarray(self.neg_idx, dtype=int)
        else:
            self.neg_idx = None

    def __getitem__(self, i):
        dataset = self.dataset
        global_i = int(self.indices[i])

        genes_i = dataset.get_gene(global_i)
        paired_emb_i = dataset.get_paired_fm(global_i)
        deg_i = dataset.get_deg(global_i)
        celltype_i = indx(self.celltype_idx, i)

        if self.neg_idx is not None:
            neg_sub_i = int(self.neg_idx[i])
            neg_global_i = int(self.indices[neg_sub_i])

            neg_genes_i = dataset.get_gene(neg_global_i)
            neg_paired_emb_i = dataset.get_paired_fm(neg_global_i)
            neg_deg_i = dataset.get_deg(neg_global_i)
            neg_celltype_i = indx(self.celltype_idx, neg_sub_i)
            neg_drug_i = indx(self.drugs_idx, neg_sub_i)
            neg_dose_i = indx(self.dosages, neg_sub_i)
        else:
            neg_genes_i = None
            neg_paired_emb_i = None
            neg_deg_i = None
            neg_celltype_i = None
            neg_drug_i = None
            neg_dose_i = None

        if self.covariates is None:
            return (
                genes_i,
                paired_emb_i,
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                deg_i,
                celltype_i,
                neg_genes_i,
                neg_paired_emb_i,
                neg_drug_i,
                neg_dose_i,
                neg_deg_i,
                neg_celltype_i,
                None,
                None,
            )
        else:
            covs = [indx(cov, i) for cov in self.covariates]
            if self.neg_idx is not None:
                neg_covs = [indx(cov, int(self.neg_idx[i])) for cov in self.covariates]
            else:
                neg_covs = [None for _ in self.covariates]

            return (
                genes_i,
                paired_emb_i,
                indx(self.drugs_idx, i),
                indx(self.dosages, i),
                deg_i,
                celltype_i,
                neg_genes_i,
                neg_paired_emb_i,
                neg_drug_i,
                neg_dose_i,
                neg_deg_i,
                neg_celltype_i,
                *covs,
                *neg_covs,
            )

    def __len__(self):
        return len(self.indices)


def custom_collate(batch):
    transposed = zip(*batch)
    concat_batch = []
    for samples in transposed:
        if samples[0] is None:
            concat_batch.append(None)
        else:
            concat_batch.append(torch.stack(samples, 0).to("cuda"))
    return concat_batch
