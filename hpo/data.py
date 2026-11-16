"""Data loading, the (unchanged) dev/test partition, and the leak-free preprocessing pipeline.

Everything learned from data (imputation, feature filter, scaling, PCA) lives in ONE sklearn Pipeline that is
fitted per fold on `fit_idx` only (see `FoldPreprocessor`), then applied -- never refitted -- to validation/test rows.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_mf import c2db_features  # noqa: E402  (same C2DB inputs as the current model)

TEST_FRAC, SPLIT_SEED = 0.15, 0   # identical to train_mf.run(): split(len(df), 0.15, default_rng(0))
N_RAW, NC = 35, 4                 # raw structural descriptors; C2DB columns appended after them


@dataclass
class Data:
    ids: np.ndarray
    X: np.ndarray        # (n, 35 + 4): raw descriptors | C2DB [pbe_min, pbe_max, has, metal]
    yP: np.ndarray       # PBE gap, NaN if unlabeled
    yG: np.ndarray       # GW gap, NaN if unlabeled
    dev: np.ndarray      # indices (into rows) of the development set
    test: np.ndarray     # held-out test indices -- only touched by final_eval.py

    @property
    def dev_gw(self): return self.dev[~np.isnan(self.yG[self.dev])]
    @property
    def dev_pbe(self): return self.dev[~np.isnan(self.yP[self.dev])]


def load_data():
    lab = pd.read_csv(ROOT / "features.csv", usecols=["id", "PBE_gap", "GW_gap"])
    raw = pd.read_csv(ROOT / "raw_features_cache.csv", index_col=0).loc[lab.id]  # built by correlation_analysis.py
    assert raw.shape[1] == N_RAW
    cg = pd.read_csv(ROOT / "c2db_gaps.csv").set_index("id")
    recs = lambda r: [{"gap": r[f"gap{j}"]} if r[f"uid{j}"] == r[f"uid{j}"] else None for j in (1, 2)]
    c2 = np.array([c2db_features(recs(cg.loc[i])) if i in cg.index else [0.0] * NC for i in lab.id])
    n = len(lab)
    p = np.random.default_rng(SPLIT_SEED).permutation(n); k = int(n * TEST_FRAC)
    D = Data(lab.id.values, np.hstack([raw.values.astype(float), c2]), lab.PBE_gap.values, lab.GW_gap.values, p[k:], p[:k])
    # the partition must be the one that produced metrics.json (test: 234 PBE / 30 GW)
    assert (~np.isnan(D.yP[D.test])).sum() == 234 and (~np.isnan(D.yG[D.test])).sum() == 30, "test split changed!"
    return D


class CorrFilter(BaseEstimator, TransformerMixin):
    """featurize.py's feature selection (drop near-constant, then |r| > thr duplicates) as a *fitted* transformer."""
    def __init__(self, corr_thr=0.95):
        self.corr_thr = corr_thr

    def fit(self, X, y=None):
        X = np.asarray(X, float); sd, mu = X.std(0), X.mean(0)
        keep = np.where((sd > 1e-3 * (np.abs(mu) + 1e-9)) & (sd > 1e-8))[0]
        corr = np.abs(np.corrcoef(X[:, keep], rowvar=False)); drop = set()
        for j in range(len(keep)):
            if any(corr[i, j] > self.corr_thr and keep[i] not in drop for i in range(j)): drop.add(keep[j])
        self.keep_ = np.array([c for c in keep if c not in drop]); return self

    def transform(self, X): return np.asarray(X, float)[:, self.keep_]


def build_pipeline(cfg):
    """[35 raw | 4 C2DB] -> [structural block (imputer -> filter -> scaler -> optional PCA) | C2DB continuous | C2DB flags].
    The last two columns stay the raw has-PBE / metal flags because the model's prior reads x[:, -2:]."""
    struct = [("imp", SimpleImputer(strategy="median")), ("filt", CorrFilter(cfg["pp_corr_thr"])), ("sc", StandardScaler())]
    if cfg["pp_pca_mode"] == "pca":
        struct.append(("pca", PCA(n_components=cfg["pp_pca_var"], whiten=cfg["pp_whiten"], svd_solver="full")))
    cont = StandardScaler() if cfg["pp_scale_c2db"] else "passthrough"
    return Pipeline([("ct", ColumnTransformer([("s", Pipeline(struct), list(range(N_RAW))),
                                               ("c", cont, [N_RAW, N_RAW + 1]),
                                               ("f", "passthrough", [N_RAW + 2, N_RAW + 3])]))])


class FoldPreprocessor:
    """Fits the pipeline on `fit_idx` ONLY and refuses overlap with the forbidden (validation/test) indices.
    Rows outside `fit_idx` ∪ `apply_idx` come back as NaN, so touching test rows by mistake fails loudly."""
    def __init__(self, cfg, fit_idx, forbidden_idx):
        fit_idx = np.asarray(fit_idx)
        overlap = np.intersect1d(fit_idx, np.asarray(forbidden_idx))
        assert overlap.size == 0, f"preprocessing fit set overlaps validation/test rows: {overlap[:5]}"
        self.fit_idx, self.cfg, self.pipe = fit_idx, cfg, build_pipeline(cfg)

    def fit_transform(self, X, apply_idx):
        self.pipe.fit(X[self.fit_idx])
        rows = np.union1d(self.fit_idx, apply_idx)
        T = self.pipe.transform(X[rows])
        out = np.full((len(X), T.shape[1]), np.nan); out[rows] = T
        return out.astype(np.float32)


def make_folds(D, k, seeds):
    """Repeated k-fold over the dev GW structures. Returns [(repeat, fold, fit_idx, val_idx)].
    `val_idx` = HF validation structures. `fit_idx` = ALL dev structures except those (so neither the preprocessing nor
    the LF/PBE training pool nor the HF training set can see a validation structure, even through its PBE label)."""
    from sklearn.model_selection import KFold
    out = []
    for r, s in enumerate(seeds):
        for f, (_, v) in enumerate(KFold(k, shuffle=True, random_state=s).split(D.dev_gw)):
            val = D.dev_gw[v]; out.append((r, f, np.setdiff1d(D.dev, val), val))
    return out
