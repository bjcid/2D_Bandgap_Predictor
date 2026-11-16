"""Featurize 2D-material CONTCARs -> scaled + PCA features. Run: python featurize.py [--soap]"""
import hashlib, json, pickle, re, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_masses, covalent_radii
from ase.io import read
from ase.neighborlist import neighbor_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
PROPS = ["Z", "chi", "radius", "valence"]
TEST_FRAC, SPLIT_SEED = 0.15, 0   # the partition of train_mf.py / hpo/data.py: split(n, 0.15, default_rng(0)); learned preprocessing sees ONLY the dev part


def dev_rows(n):
    """Positional indices (into the features table) of the development structures; the other 15 % is the held-out test set."""
    return np.random.default_rng(SPLIT_SEED).permutation(n)[int(n * TEST_FRAC):]


def ids_sha1(ids):
    return hashlib.sha1(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()


@lru_cache(None)
def element_props(z):
    """(Z, Pauling electronegativity, covalent radius [A], valence electrons) for element Z."""
    from pymatgen.core import Element
    e = Element.from_Z(int(z))
    g = e.group
    val = g if g <= 12 else g - 10  # s/d-block: group number; p-block: group-10
    chi = e.X if e.X == e.X else np.nan  # NaN for noble gases
    return float(z), float(chi), float(covalent_radii[int(z)]), float(val)


def read_structure(src):
    atoms = read(src, format="vasp")
    if len(atoms) == 0:
        raise ValueError("Empty structure")
    return atoms


def raw_features(atoms, soap_species=None):
    """Fixed-length feature dict for one 2D structure (vacuum along c)."""
    f = {}
    P = np.array([element_props(z) for z in atoms.numbers])
    for i, p in enumerate(PROPS):
        c = P[:, i]
        f.update({f"{p}_mean": np.nanmean(c) if not np.all(np.isnan(c)) else 0.0,
                  f"{p}_std": np.nanstd(c) if not np.all(np.isnan(c)) else 0.0,
                  f"{p}_min": np.nanmin(c) if not np.all(np.isnan(c)) else 0.0,
                  f"{p}_max": np.nanmax(c) if not np.all(np.isnan(c)) else 0.0})
    cell = atoms.cell
    a, b, c = cell.lengths()
    al, be, ga = cell.angles()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    height = atoms.get_volume() / area
    fz = np.sort(atoms.get_scaled_positions()[:, 2] % 1.0)
    gaps = np.diff(np.concatenate([fz, [fz[0] + 1]]))
    thick = max((1 - gaps.max()) * height, 0.0)  # slab thickness across periodic wrap
    mass = atomic_masses[atoms.numbers].sum()
    f.update(n_atoms=len(atoms), n_species=len(set(atoms.numbers)), a=a, b=b, c=c,
             alpha=al, beta=be, gamma=ga, area=area, area_per_atom=area / len(atoms),
             thickness=thick, density_areal=mass / area, density_slab=mass / (area * max(thick, 1.0)) * 1.66054)
    i, d = neighbor_list("id", atoms, 6.0)
    nn = np.array([d[i == k].min() if np.any(i == k) else 6.0 for k in range(len(atoms))])
    cn = np.array([np.sum(d[i == k] < 1.25 * nn[k]) for k in range(len(atoms))])
    f.update(nn_mean=nn.mean(), nn_std=nn.std(), nn_min=nn.min(), nn_max=nn.max(),
             cn_mean=cn.mean(), cn_std=cn.std())
    if soap_species:
        from dscribe.descriptors import SOAP
        s = SOAP(species=soap_species, r_cut=5.0, n_max=3, l_max=2, average="inner", periodic=True)
        v = s.create(atoms)
        f.update({f"soap_{k}": x for k, x in enumerate(np.ravel(v))})
    return f


def transform(raw_df, scaler, pca, kept):
    """Raw feature DataFrame -> PCA features (same pipeline as training)."""
    return pca.transform(scaler.transform(raw_df[kept].values))


def load_artifacts(d=ROOT):
    d = Path(d)
    with open(d / "scaler.pkl", "rb") as fh: scaler = pickle.load(fh)
    with open(d / "pca.pkl", "rb") as fh: pca = pickle.load(fh)
    meta = json.load(open(d / "feature_names.json"))
    return scaler, pca, meta


def featurize_atoms(atoms, scaler, pca, meta):
    raw = pd.DataFrame([raw_features(atoms, meta.get("soap_species"))])
    return transform(raw, scaler, pca, meta["kept"]).astype(np.float32)


def main(use_soap=False):
    df = pd.read_csv(ROOT / "DFT_data.csv")
    files = {int(m.group(1)): p for p in (ROOT / "CONTCAR_files").iterdir() if (m := re.match(r"(\d+)_?CONTCAR$", p.name))}
    ids = sorted(files)
    atoms_all, bad = {}, []
    for k in ids:
        try: atoms_all[k] = read_structure(files[k])
        except Exception as e: bad.append((k, str(e)))
    species = sorted({int(z) for a in atoms_all.values() for z in a.numbers}) if use_soap else None
    rows = {k: raw_features(a, species) for k, a in atoms_all.items()}
    raw = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    # Everything learned from data (imputation medians, feature filter, scaler, PCA) is fitted on the DEVELOPMENT structures only
    # and merely applied to the held-out test structures (no leakage of the test set into the preprocessing).
    dev = dev_rows(len(raw)); fit_ids = raw.index[dev]
    raw = raw.fillna(raw.iloc[dev].median()); rf = raw.iloc[dev]
    # drop near-constant, then highly correlated (|r|>0.95) features
    keep = [c for c in rf if rf[c].std() > 1e-3 * (abs(rf[c].mean()) + 1e-9) and rf[c].std() > 1e-8]
    corr = rf[keep].corr().abs().values
    drop = set()
    for j in range(len(keep)):
        if any(corr[i, j] > 0.95 and keep[i] not in drop for i in range(j)): drop.add(keep[j])
    kept = [c for c in keep if c not in drop]
    scaler = StandardScaler().fit(rf[kept].values)
    pca = PCA(n_components=0.95, svd_solver="full").fit(scaler.transform(rf[kept].values))
    X = pca.transform(scaler.transform(raw[kept].values))
    out = pd.DataFrame(X, index=raw.index, columns=[f"PC{i+1}" for i in range(X.shape[1])])
    out.insert(0, "id", raw.index)
    lab = df.set_index("file_name")[["PBE_Bandgap", "GW_Bandgap"]].rename(columns={"PBE_Bandgap": "PBE_gap", "GW_Bandgap": "GW_gap"})
    out = out.join(lab, on="id")
    out.to_csv(ROOT / "features.csv", index=False)
    pickle.dump(scaler, open(ROOT / "scaler.pkl", "wb")); pickle.dump(pca, open(ROOT / "pca.pkl", "wb"))
    json.dump({"raw": list(raw.columns), "kept": kept, "dropped": sorted(set(raw.columns) - set(kept)),
               "pca": list(out.columns[1:-2]), "soap_species": species,
               "explained_variance": float(pca.explained_variance_ratio_.sum()),
               "fit_on": "development split only", "n_fit": int(len(fit_ids)), "fit_ids_sha1": ids_sha1(fit_ids)},
              open(ROOT / "feature_names.json", "w"), indent=1)
    print(f"{len(ids)} CONTCARs ({len(bad)} unreadable), {len(raw.columns)} raw -> {len(kept)} kept -> {X.shape[1]} PCs; "
          f"PBE labels {out.PBE_gap.notna().sum()}, GW labels {out.GW_gap.notna().sum()}")
    for k, e in bad: print("skipped", k, e)


if __name__ == "__main__":
    main("--soap" in sys.argv)
