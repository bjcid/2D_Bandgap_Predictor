"""Inference for the final model (hpo/export_to_app.py + hpo/export_pbe_to_app.py).

  GW  : tuned multi-fidelity ensemble   model_v2.pt + preproc_v2.joblib + model_v2_meta.json
  PBE : PBE-only ensemble (the PBE head of the joint GW model is only a by-product of the GW search: RMSE ~0.36 vs ~0.22)
        pbe_v2.pt + preproc_pbe_v2.joblib
  The two are joined by `apply_metal_rule`.

The tuned model needs the C2DB monolayer inputs: without them it is much less accurate than the original model
(see HPO_REPORT.md), so app.py uses it only when C2DB layers were found and falls back to model.pt otherwise.
"""
import json, sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "hpo"))   # network classes (mf_model.py) and data.CorrFilter (needed to unpickle the preprocessing)
import mf_model as M                    # noqa: E402
import data as _hpo_data                # noqa: E402,F401
from featurize import raw_features      # noqa: E402
from train_mf import c2db_features      # noqa: E402

FILES = ("model_v2.pt", "preproc_v2.joblib", "model_v2_meta.json", "pbe_v2.pt", "preproc_pbe_v2.joblib")


def available(root=ROOT):
    return all((Path(root) / f).exists() for f in FILES)


def apply_metal_rule(pbe, gw):
    """A structure predicted metallic (gap exactly 0) at one level is metallic at both: PBE == 0 -> GW = 0 (GW >= PBE for the
    metals in the data: all 7 PBE-metals with a GW label have GW <= 0.11 eV) and GW == 0 -> PBE = 0. Works on floats and arrays."""
    pbe, gw = np.asarray(pbe, float), np.asarray(gw, float)
    gw = np.where(pbe == 0, 0.0, gw); pbe = np.where(gw == 0, 0.0, pbe)
    return pbe, gw


def load_v2(root=ROOT):
    root = Path(root)
    ck = torch.load(root / "model_v2.pt", map_location="cpu", weights_only=False)
    cfg = {**M.DEFAULT, **ck["cfg"]}
    models = []
    for s in ck["states"]:
        m = M.HighFid(ck["in_dim"], M.LowFid(ck["in_dim"], cfg), cfg); m.load_state_dict(s); models.append(m.eval())
    pk = torch.load(root / "pbe_v2.pt", map_location="cpu", weights_only=False); pcfg = {**M.DEFAULT, **pk["cfg"]}
    pbe_nets = []
    for s in pk["states"]:
        m = M.LowFid(pk["in_dim"], pcfg); m.load_state_dict(s); pbe_nets.append(m.eval())
    return dict(models=models, pipe=joblib.load(root / "preproc_v2.joblib"), pbe_nets=pbe_nets, pbe_pipe=joblib.load(root / "preproc_pbe_v2.joblib"),
                meta=json.load(open(root / "model_v2_meta.json", encoding="utf-8")))


def design_row(atoms, layers, meta):
    """[35 raw structural descriptors | 4 C2DB inputs] in the column order the preprocessing was fitted with."""
    raw = raw_features(atoms)
    return np.array([[raw[c] for c in meta["raw_columns"]] + list(c2db_features(layers or [None, None]))], float)


def predict_v2(atoms, layers, bundle):
    """-> (PBE gap, GW gap, P(metal)) in eV. P(metal) is the larger of the two models' metal probabilities when the rule makes the gap 0."""
    row = design_row(atoms, layers, bundle["meta"])
    gw, p_gw = M.predict(bundle["models"], torch.tensor(bundle["pipe"].transform(row).astype(np.float32)))
    pbe, p_pbe = M.predict(bundle["pbe_nets"], torch.tensor(bundle["pbe_pipe"].transform(row).astype(np.float32)))
    pbe, gw = apply_metal_rule(pbe[0], gw[0])
    return float(pbe), float(gw), float(max(p_gw[0], p_pbe[0]) if gw == 0 else p_gw[0])
