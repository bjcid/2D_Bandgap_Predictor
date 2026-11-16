"""Multi-fidelity NN: PBE (low) -> GW (high) with a metal/non-metal hurdle head, C2DB inputs and ensembling.
Run: python train_mf.py"""
import copy, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
SEED, K, HID, ENS = 0, 5, 64, 5
THR_L, THR_H = 0.01, 0.02  # gaps below this (eV) count as metallic -> predicted as exactly 0
NC = 4                      # number of C2DB input columns (appended after the PCA features)
torch.manual_seed(SEED); np.random.seed(SEED)


def mlp(i, o, h=HID, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, o))


def prior(x, w):
    """Learnable prior from C2DB: monolayer PBE gap == 0 (flag) strongly suggests a metal; has-PBE masks missing data."""
    has, flag = x[:, -NC + 2], x[:, -NC + 3]
    return w * has * (2 * flag - 1)


class LowFid(nn.Module):
    """f_L(x) -> (metal logit, gap value). Also used as the GW-only baseline."""
    def __init__(self, d, thr=THR_L, w0=0.0):
        super().__init__(); self.net = mlp(d, 2); self.thr = thr; self.wf = nn.Parameter(torch.tensor(w0))
    def forward(self, x):
        o = self.net(x); return o[:, 0] + prior(x, self.wf), F.softplus(o[:, 1], beta=5)


class HighFid(nn.Module):
    """f_H(x, f_L) : value = softplus(rho*f_L + delta), metal logit = logit_L + delta_logit."""
    def __init__(self, d, low, thr=THR_H, w0=0.0):
        super().__init__(); self.low, self.thr = low, thr; self.rho = nn.Parameter(torch.tensor(1.0)); self.delta = mlp(d + 2, 2); self.wf = nn.Parameter(torch.tensor(w0))
    def forward(self, x):
        lg, v = self.low(x); o = self.delta(torch.cat([x, v[:, None], lg[:, None]], 1))
        return lg + o[:, 0] + prior(x, self.wf), F.softplus(self.rho * v + o[:, 1], beta=5)


def crit(model, out, y):
    lg, v = out; m = (y < model.thr).float()
    return F.binary_cross_entropy_with_logits(lg, m) + (((v - y) ** 2) * (1 - m)).sum() / (1 - m).sum().clamp(min=1)


def fit(model, X, y, Xv, yv, lr=3e-3, wd=1e-4, epochs=1500, pat=80, params=None, nc=0):
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = 1e9, copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(X)).split(128):
            xb = X[b].clone()
            if nc: xb[torch.rand(len(b)) < 0.15, -nc:] = 0  # hide C2DB inputs sometimes -> robust when C2DB is unavailable
            opt.zero_grad(); crit(model, model(xb), y[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = crit(model, model(Xv), yv).item()
        if v < best - 1e-5: best, state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= pat: break
    model.load_state_dict(state); return model


def predict(models, X):
    """Ensemble prediction -> (gap, p_metal); gap is exactly 0 where p_metal > 0.5."""
    with torch.no_grad():
        outs = [(m.eval(), m(X))[1] for m in models]
    p = torch.stack([torch.sigmoid(o[0]) for o in outs]).mean(0); v = torch.stack([o[1] for o in outs]).mean(0)
    return torch.where(p > 0.5, torch.zeros_like(v), v).numpy(), p.numpy()


def metrics(y, p, thr=None):
    e = p - y
    r = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), n=int(len(y)))
    if thr is not None and (y < thr).any():
        met = y < thr  # how well are metals (gap = 0) handled?
        r.update(metal_n=int(met.sum()), metal_MAE=float(np.abs(e[met]).mean()), metal_recall=float((p[met] < 0.05).mean()),
                 false_metal=int(((p < 0.05) & ~met).sum()), nonmetal_MAE=float(np.abs(e[~met]).mean()))
    return r


def split(n, frac, rng):
    p = rng.permutation(n); k = int(n * frac); return p[k:], p[:k]


def T(a): return torch.tensor(np.asarray(a), dtype=torch.float32)


def train_low(X, y, rng, d, nc=0, n=ENS):
    out = []
    for _ in range(n):
        tr, va = split(len(X), 0.15, rng); out.append(fit(LowFid(d), X[tr], y[tr], X[va], y[va], nc=nc))
    return out


def train_high(X, y, lows, rng, d, nc=0):
    """Stage 2a: f_L frozen, train rho+delta. Stage 2b: unfreeze, fine-tune at low LR. One model per f_L of the ensemble."""
    out = []
    for low in lows:
        tr, va = split(len(X), 0.15, rng); m = HighFid(d, copy.deepcopy(low), w0=4.0 if nc else 0.0)
        for p in m.low.parameters(): p.requires_grad = False
        m = fit(m, X[tr], y[tr], X[va], y[va], params=[p for p in m.parameters() if p.requires_grad], wd=1e-3, nc=nc)
        for p in m.low.parameters(): p.requires_grad = True
        out.append(fit(m, X[tr], y[tr], X[va], y[va], lr=1e-4, wd=1e-3, epochs=400, pat=40, nc=nc))
    return out


def train_gw_only(X, y, rng, d, nc=0):
    out = []
    for _ in range(ENS):
        tr, va = split(len(X), 0.15, rng); out.append(fit(LowFid(d, THR_H, w0=4.0 if nc else 0.0), X[tr], y[tr], X[va], y[va], wd=1e-3, nc=nc))
    return out


def c2db_features(layers):
    """C2DB inputs (only the computed PBE gap of the two monolayers; HSE/GW there are predictions):
    [PBE min, PBE max, has PBE, C2DB-metal flag (PBE gap ~ 0)] (zeros when missing)."""
    pbe = [l["gap"] for l in layers if l and l.get("gap") is not None and l["gap"] == l["gap"]]
    return [min(pbe, default=0.0), max(pbe, default=0.0), float(bool(pbe)), float(bool(pbe) and max(pbe) <= 0.05)]


def parity(ax, y, preds, title):
    lim = [0, max(y.max(), *(p.max() for p in preds.values())) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    for (k, p), c in zip(preds.items(), ["#2a6fdb", "#e0782b", "#888888"]):
        ax.scatter(y, p, s=18, alpha=.75, c=c, label=f"{k} (MAE {np.abs(p - y).mean():.2f})")
    ax.set(xlabel="DFT gap (eV)", ylabel="Predicted gap (eV)", title=title, xlim=lim, ylim=lim); ax.legend(fontsize=8)


def run(use_c2db, save=True, ckpt=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    df = pd.read_csv(ROOT / "features.csv"); pc = [c for c in df if c.startswith("PC")]
    nc = 0
    if use_c2db:  # third data source: C2DB monolayer PBE gaps of the two layers
        cg = pd.read_csv(ROOT / "c2db_gaps.csv").set_index("id")
        recs = lambda r: [{"gap": r[f"gap{j}"]} if r[f"uid{j}"] == r[f"uid{j}"] else None for j in (1, 2)]
        cn = ["C_pbe_min", "C_pbe_max", "C_has_pbe", "C_metal"]
        df[cn] = [c2db_features(recs(cg.loc[i])) if i in cg.index else [0.0] * NC for i in df.id]
        pc, nc = pc + cn, NC
    d = len(pc); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dev_i, te_i = split(len(df), 0.15, rng)  # held-out test (shared by both fidelities)
    from featurize import ids_sha1   # the scaler/PCA in features.csv must have been fitted on exactly these development structures
    assert json.load(open(ROOT / "feature_names.json")).get("fit_ids_sha1") == ids_sha1(df.id.values[dev_i]),         "features.csv was built with a PCA fitted on other structures than this dev split: rerun `python featurize.py`"
    dev, te = df.iloc[dev_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)
    P, G = dev[dev.PBE_gap.notna()], dev[dev.GW_gap.notna()].reset_index(drop=True)  # skip unlabeled
    Pt, Gt = te[te.PBE_gap.notna()], te[te.GW_gap.notna()]
    print(f"[c2db={use_c2db}] dev PBE {len(P)} / GW {len(G)}; test PBE {len(Pt)} / GW {len(Gt)}", flush=True)
    XP, yP = T(P[pc].values), T(P.PBE_gap.values)
    out = {"n_features": d, "use_c2db": use_c2db, "pbe": {}, "gw": {}}

    # ---- Stage 1: PBE ----
    oof = np.zeros(len(P))
    for a, b in KFold(K, shuffle=True, random_state=SEED).split(XP):
        oof[b] = predict(train_low(XP[a], yP[a], rng, d, nc), XP[b])[0]
    out["pbe"]["cv"] = metrics(P.PBE_gap.values, oof, THR_L)
    lows = train_low(XP, yP, rng, d, nc)
    pb_te = predict(lows, T(Pt[pc].values))[0]; out["pbe"]["test"] = metrics(Pt.PBE_gap.values, pb_te, THR_L)
    print("PBE", out["pbe"], flush=True)

    # ---- Stage 2: GW (multi-fidelity vs GW-only vs linear PBE->GW) ----
    XG, yG = T(G[pc].values), T(G.GW_gap.values)
    pbe_hat_G, pbe_true_G, yg = predict(lows, XG)[0], G.PBE_gap.values, G.GW_gap.values
    cvp = {k: np.zeros(len(G)) for k in ["Multi-fidelity", "GW-only NN", "Linear PBE->GW", "Linear predPBE->GW"]}
    for a, b in KFold(K, shuffle=True, random_state=SEED).split(XG):
        cvp["Multi-fidelity"][b] = predict(train_high(XG[a], yG[a], lows, rng, d, nc), XG[b])[0]
        cvp["GW-only NN"][b] = predict(train_gw_only(XG[a], yG[a], rng, d, nc), XG[b])[0]
        cvp["Linear PBE->GW"][b] = LinearRegression().fit(pbe_true_G[a, None], yg[a]).predict(pbe_true_G[b, None]).clip(0)
        cvp["Linear predPBE->GW"][b] = LinearRegression().fit(pbe_hat_G[a, None], yg[a]).predict(pbe_hat_G[b, None]).clip(0)
    out["gw"]["cv"] = {k: metrics(yg, v, THR_H) for k, v in cvp.items()}

    mf, gwo = train_high(XG, yG, lows, rng, d, nc), train_gw_only(XG, yG, rng, d, nc)
    XGt, ytG = T(Gt[pc].values), Gt.GW_gap.values
    lin_true = LinearRegression().fit(pbe_true_G[:, None], yg); lin_hat = LinearRegression().fit(pbe_hat_G[:, None], yg)
    tp = {"Multi-fidelity": predict(mf, XGt)[0], "GW-only NN": predict(gwo, XGt)[0],
          "Linear PBE->GW": lin_true.predict(Gt.PBE_gap.values[:, None]).clip(0),
          "Linear predPBE->GW": lin_hat.predict(predict(lows, XGt)[0][:, None]).clip(0)}
    out["gw"]["test"] = {k: metrics(ytG, v, THR_H) for k, v in tp.items()}
    print("GW CV", json.dumps(out["gw"]["cv"]["Multi-fidelity"]), "\nGW test", json.dumps(out["gw"]["test"]["Multi-fidelity"]), flush=True)
    if ckpt is not None:  # structure-only model (the app's answer when no C2DB monolayer is found)
        torch.save({"mf_states": [m.state_dict() for m in mf], "in_dim": d, "use_c2db": use_c2db, "nc": nc, "metrics": out}, ckpt)
    if not save: return out

    out["ablation_no_c2db"] = run(False, save=False, ckpt=ROOT / "model_noc2db.pt")
    json.dump(out, open(ROOT / "metrics.json", "w"), indent=1)
    torch.save({"mf_states": [m.state_dict() for m in mf], "in_dim": d, "use_c2db": use_c2db, "nc": nc, "metrics": out}, ROOT / "model.pt")
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), dpi=130)
    parity(ax[0], Pt.PBE_gap.values, {"Low-fidelity NN": pb_te}, "PBE (test)")
    parity(ax[1], ytG, {k: tp[k] for k in ["Multi-fidelity", "GW-only NN", "Linear PBE->GW"]}, "GW (test)")
    fig.tight_layout(); fig.savefig(ROOT / "parity_plots.png")
    fig, a = plt.subplots(figsize=(5.5, 5), dpi=130); parity(a, yg, {k: cvp[k] for k in ["Multi-fidelity", "GW-only NN", "Linear PBE->GW"]}, f"GW ({K}-fold CV)")
    fig.tight_layout(); fig.savefig(ROOT / "parity_gw_cv.png")
    return out


def load_ensemble(ck):
    """Rebuild the GW ensemble (each member contains its own PBE head) from a model.pt checkpoint."""
    ms = []
    for s in ck["mf_states"]:
        m = HighFid(ck["in_dim"], LowFid(ck["in_dim"])); m.load_state_dict(s); ms.append(m.eval())
    return ms


def main():
    import sys
    if "--fallback-only" in sys.argv:  # retrain only the structure-only model (model_noc2db.pt) and refresh its entry in metrics.json
        o = run(False, save=False, ckpt=ROOT / "model_noc2db.pt"); m = json.load(open(ROOT / "metrics.json")); m["ablation_no_c2db"] = o
        json.dump(m, open(ROOT / "metrics.json", "w"), indent=1); return
    run(True)


if __name__ == "__main__":
    main()
