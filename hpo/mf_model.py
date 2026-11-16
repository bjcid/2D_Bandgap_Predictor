"""Configurable multi-fidelity network: a generalisation of train_mf.py whose DEFAULT config reproduces the current code.

  f_L(x)            -> (metal logit, PBE value)                       [LowFid]
  f_H(x, f_L(x))    -> (metal logit, GW value) = lin(v) + delta(x, v, lg)  [HighFid]
     lin  : scalar rho | rho*v + beta | rho(x)*v + beta(x)  (small "linear-correlation" subnet)
     delta: non-linear subnet (inputs x [+ v, lg])
All hyper-parameters come from one flat dict (see DEFAULT); prefixes: lf_ (low fid.), hf_ (high fid.), ft_ (fine-tune),
ho_ (HF-only baseline), pp_ (preprocessing, used in data.py).
"""
import copy, hashlib, json, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC, THR_L, THR_H = 4, 0.01, 0.02

DEFAULT = dict(
    # --- preprocessing (data.py)
    pp_pca_mode="pca", pp_pca_var=0.95, pp_whiten=False, pp_scale_c2db=False, pp_corr_thr=0.95,
    # --- shared
    ens=5, val_frac=0.15, softplus_beta=5.0, strategy="sequential",
    # --- low-fidelity net + its training
    lf_depth=2, lf_w1=64, lf_w2=64, lf_w3=64, lf_w4=64, lf_act="silu", lf_dropout=0.1, lf_residual=False, lf_init="default",
    lf_lr=3e-3, lf_wd=1e-4, lf_bs=128, lf_sched="none", lf_epochs=1500, lf_patience=80, lf_w_bce=1.0, lf_w_mse=1.0,
    lf_mask_p=0.15, lf_prior_w0=0.0,
    # --- high-fidelity structure
    coupling="both", rho_mode="scalar", lin_depth=1, lin_w1=16, lin_w2=16, lin_act="tanh",
    delta_inputs="x+v+lg", hf_depth=2, hf_w1=64, hf_w2=64, hf_w3=64, hf_w4=64, hf_act="silu", hf_dropout=0.1,
    hf_residual=False, hf_init="default", hf_zero_init_delta=False, hf_prior_w0=4.0,
    # --- stage 2a (f_L frozen) training
    hf_lr=3e-3, hf_wd=1e-3, hf_lin_wd=None, scalars_wd=None, hf_bs=128, hf_sched="none", hf_epochs=1500, hf_patience=80,
    hf_w_bce=1.0, hf_w_mse=1.0, hf_mask_p=0.15, hf_low_eval=False,
    # --- stage 2b (fine-tune) training
    ft_mode="freeze_then_ft", ft_optim="adamw", ft_lr=1e-4, ft_lr_low=1e-4, ft_wd=1e-3, ft_wd_low=1e-3,
    ft_epochs=400, ft_patience=40,
    # --- joint strategy
    joint_lambda_lf=1.0,
    # --- HF-only baseline (current: LowFid(thr_H) with wd 1e-3 and prior w0=4)
    ho_depth=2, ho_w1=64, ho_w2=64, ho_w3=64, ho_w4=64, ho_act="silu", ho_dropout=0.1, ho_residual=False, ho_init="default",
    ho_lr=3e-3, ho_wd=1e-3, ho_bs=128, ho_sched="none", ho_epochs=1500, ho_patience=80, ho_mask_p=0.15, ho_prior_w0=4.0,
    ho_w_bce=1.0, ho_w_mse=1.0,
)
LF_KEYS = lambda k: k.startswith(("pp_", "lf_")) or k in ("ens", "val_frac", "softplus_beta", "strategy")

ACT = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def widths(cfg, p):
    return [int(cfg[f"{p}_w{k}"]) for k in range(1, int(cfg[f"{p}_depth"]) + 1)]


def init_(m, mode, act):
    if mode == "default": return
    gain = nn.init.calculate_gain("tanh" if act == "tanh" else "relu")
    for l in m.modules():
        if isinstance(l, nn.Linear):
            {"kaiming": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
             "xavier": lambda w: nn.init.xavier_uniform_(w, gain=gain),
             "orthogonal": lambda w: nn.init.orthogonal_(w, gain=gain)}[mode](l.weight)
            nn.init.zeros_(l.bias)


class MLP(nn.Module):
    """Linear -> act -> dropout (x depth) -> Linear. With widths=[64, 64], act=silu, residual=False this is exactly mlp()."""
    def __init__(self, i, o, ws, act="silu", p=0.1, residual=False, init="default", zero_last=False):
        super().__init__()
        dims = [i] + list(ws)
        self.lins = nn.ModuleList(nn.Linear(dims[k], dims[k + 1]) for k in range(len(ws)))
        self.out = nn.Linear(dims[-1], o); self.act, self.drop = ACT[act](), nn.Dropout(p)
        self.res = [bool(residual) and k > 0 and dims[k] == dims[k + 1] for k in range(len(ws))]
        init_(self, init, act)
        if zero_last: nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = x
        for lin, r in zip(self.lins, self.res):
            z = self.drop(self.act(lin(h))); h = h + z if r else z
        return self.out(h)


def prior(x, w):
    has, flag = x[:, -NC + 2], x[:, -NC + 3]
    return w * has * (2 * flag - 1)


class LowFid(nn.Module):
    """Also used for the HF-only baseline (prefix 'ho')."""
    def __init__(self, d, cfg, prefix="lf", thr=THR_L):
        super().__init__()
        self.net = MLP(d, 2, widths(cfg, prefix), cfg[f"{prefix}_act"], cfg[f"{prefix}_dropout"], cfg[f"{prefix}_residual"], cfg[f"{prefix}_init"])
        self.wf = nn.Parameter(torch.tensor(float(cfg[f"{prefix}_prior_w0"]))); self.thr, self.bt = thr, cfg["softplus_beta"]

    def forward(self, x):
        o = self.net(x); return o[:, 0] + prior(x, self.wf), F.softplus(o[:, 1], beta=self.bt)


class HighFid(nn.Module):
    def __init__(self, d, low, cfg):
        super().__init__()
        self.low, self.thr, self.bt = low, THR_H, cfg["softplus_beta"]
        self.coupling, self.rho_mode, self.din = cfg["coupling"], cfg["rho_mode"], cfg["delta_inputs"]
        self.rho, self.beta0 = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0))
        self.wf = nn.Parameter(torch.tensor(float(cfg["hf_prior_w0"])))
        self.lin = MLP(d, 2, widths(cfg, "lin"), cfg["lin_act"], 0.0, False, cfg["hf_init"], zero_last=True) \
            if self.rho_mode == "net" and self.coupling != "nonlinear" else None
        di = d + {"x+v+lg": 2, "x+v": 1, "x": 0}[self.din]
        self.delta = MLP(di, 2, widths(cfg, "hf"), cfg["hf_act"], cfg["hf_dropout"], cfg["hf_residual"], cfg["hf_init"],
                         zero_last=cfg["hf_zero_init_delta"]) if self.coupling != "linear" else None

    def lin_part(self, x, v):
        if self.rho_mode == "scalar": return self.rho * v
        if self.rho_mode == "affine": return self.rho * v + self.beta0
        o = self.lin(x); return (self.rho + o[:, 0]) * v + self.beta0 + o[:, 1]

    def forward(self, x):
        lg, v = self.low(x); pr = prior(x, self.wf)
        if self.coupling == "linear": return lg + pr, F.softplus(self.lin_part(x, v), beta=self.bt)
        inp = {"x+v+lg": [x, v[:, None], lg[:, None]], "x+v": [x, v[:, None]], "x": [x]}[self.din]
        o = self.delta(torch.cat(inp, 1))
        val = o[:, 1] if self.coupling == "nonlinear" else self.lin_part(x, v) + o[:, 1]
        return lg + o[:, 0] + pr, F.softplus(val, beta=self.bt)


def crit(model, out, y, w_bce=1.0, w_mse=1.0):
    lg, v = out; m = (y < model.thr).float()
    return w_bce * F.binary_cross_entropy_with_logits(lg, m) + w_mse * (((v - y) ** 2) * (1 - m)).sum() / (1 - m).sum().clamp(min=1)


def _scheduler(opt, kind, pat):
    if kind == "cosine": return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    if kind == "plateau": return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(5, pat // 4), min_lr=1e-6)
    return None


def fit(model, X, y, Xv, yv, groups, bs, epochs, pat, sched="none", w=(1.0, 1.0), mask_p=0.0, eval_mod=None):
    """Mini-batch AdamW with early stopping on the validation loss (same loop as train_mf.fit, plus options)."""
    opt = torch.optim.AdamW([g for g in groups if g["params"]]); sch = _scheduler(opt, sched, pat)
    best, state, bad = 1e9, copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        if eval_mod is not None: eval_mod.eval()  # e.g. keep a frozen f_L deterministic (dropout off)
        for b in torch.randperm(len(X)).split(bs):
            xb = X[b].clone()
            if mask_p: xb[torch.rand(len(b)) < mask_p, -NC:] = 0  # hide C2DB inputs sometimes
            opt.zero_grad(); crit(model, model(xb), y[b], *w).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = crit(model, model(Xv), yv, *w).item()
        if not np.isfinite(v): break
        if sch is not None: sch.step(v) if sched == "plateau" else sch.step()
        if v < best - 1e-5: best, state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= pat: break
    model.load_state_dict(state); return model


def fit_lbfgs(model, X, y, Xv, yv, params, wd, steps, pat, w=(1.0, 1.0), lr=1.0):
    """Full-batch L-BFGS fine-tuning (deterministic: dropout off). Explicit L2 on `params`; early stopping on validation."""
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.LBFGS(params, lr=lr, max_iter=10, history_size=20, line_search_fn="strong_wolfe")
    best, state, bad = 1e9, copy.deepcopy(model.state_dict()), 0
    model.eval()
    with torch.no_grad(): best = crit(model, model(Xv), yv, *w).item()
    for _ in range(steps):
        def closure():
            opt.zero_grad(); l = crit(model, model(X), y, *w) + 0.5 * wd * sum((p ** 2).sum() for p in params); l.backward(); return l
        try: opt.step(closure)
        except Exception: break
        with torch.no_grad(): v = crit(model, model(Xv), yv, *w).item()
        if not np.isfinite(v): break
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


def T(a): return torch.tensor(np.asarray(a), dtype=torch.float32)


def split(n, frac, rng):
    p = rng.permutation(n); k = int(n * frac); return p[k:], p[:k]


def _wd(cfg, base): return base if cfg["scalars_wd"] is None else cfg["scalars_wd"]


# ------------------------------------------------------------------ training stages
def train_low(X, y, rng, d, cfg, n=None):
    out = []
    for _ in range(n or cfg["ens"]):
        tr, va = split(len(X), cfg["val_frac"], rng); m = LowFid(d, cfg, "lf", THR_L)
        g = [dict(params=list(m.net.parameters()), lr=cfg["lf_lr"], weight_decay=cfg["lf_wd"]),
             dict(params=[m.wf], lr=cfg["lf_lr"], weight_decay=_wd(cfg, cfg["lf_wd"]))]
        out.append(fit(m, X[tr], y[tr], X[va], y[va], g, cfg["lf_bs"], cfg["lf_epochs"], cfg["lf_patience"], cfg["lf_sched"],
                       (cfg["lf_w_bce"], cfg["lf_w_mse"]), cfg["lf_mask_p"]))
    return out


def _hf_groups(m, cfg, stage):
    """Parameter groups (lr / weight decay per subnet). stage: 'a' (f_L frozen), 'a_unfrozen', 'b' (fine-tune)."""
    scal = [m.rho, m.beta0, m.wf]
    if stage == "b":
        lr_h, wd_h, lr_l, wd_l = cfg["ft_lr"], cfg["ft_wd"], cfg["ft_lr_low"], cfg["ft_wd_low"]; lin_wd = wd_h
    else:
        lr_h, wd_h, lr_l, wd_l = cfg["hf_lr"], cfg["hf_wd"], cfg["ft_lr_low"], cfg["ft_wd_low"]
        lin_wd = wd_h if cfg["hf_lin_wd"] is None else cfg["hf_lin_wd"]
    g = [dict(params=list(m.delta.parameters()) if m.delta is not None else [], lr=lr_h, weight_decay=wd_h),
         dict(params=list(m.lin.parameters()) if m.lin is not None else [], lr=lr_h, weight_decay=lin_wd),
         dict(params=scal, lr=lr_h, weight_decay=_wd(cfg, wd_h))]
    if stage != "a": g.append(dict(params=list(m.low.parameters()), lr=lr_l, weight_decay=wd_l))
    return g


def train_high(X, y, lows, rng, d, cfg):
    """Stage 2a: f_L frozen, train lin+delta. Stage 2b: unfreeze, fine-tune (AdamW or L-BFGS). One model per f_L member."""
    out, w = [], (cfg["hf_w_bce"], cfg["hf_w_mse"]); mode = cfg["ft_mode"]
    for low in lows:
        tr, va = split(len(X), cfg["val_frac"], rng); m = HighFid(d, copy.deepcopy(low), cfg)
        if mode == "unfrozen_only":
            m = fit(m, X[tr], y[tr], X[va], y[va], _hf_groups(m, cfg, "a_unfrozen"), cfg["hf_bs"], cfg["hf_epochs"], cfg["hf_patience"], cfg["hf_sched"], w, cfg["hf_mask_p"])
            out.append(m); continue
        for p in m.low.parameters(): p.requires_grad = False
        # hf_low_eval=False reproduces train_mf.py, where dropout stays active inside the "frozen" f_L during stage 2a
        m = fit(m, X[tr], y[tr], X[va], y[va], _hf_groups(m, cfg, "a"), cfg["hf_bs"], cfg["hf_epochs"], cfg["hf_patience"], cfg["hf_sched"], w, cfg["hf_mask_p"],
                eval_mod=m.low if cfg["hf_low_eval"] else None)
        if mode == "freeze_then_ft":
            for p in m.low.parameters(): p.requires_grad = True
            if cfg["ft_optim"] == "lbfgs":
                m = fit_lbfgs(m, X[tr], y[tr], X[va], y[va], list(m.parameters()), cfg["ft_wd"], cfg["ft_epochs"], cfg["ft_patience"], w)
            else:
                m = fit(m, X[tr], y[tr], X[va], y[va], _hf_groups(m, cfg, "b"), cfg["hf_bs"], cfg["ft_epochs"], cfg["ft_patience"], "none", w, cfg["hf_mask_p"])
        out.append(m)
    return out


def train_joint(XP, yP, XG, yG, rng, d, cfg):
    """Joint strategy: each ensemble member trains f_L and f_H together; loss = lambda_L*L_LF(PBE batch) + L_HF(GW batch)."""
    out, lam = [], cfg["joint_lambda_lf"]
    for _ in range(cfg["ens"]):
        trP, vaP = split(len(XP), cfg["val_frac"], rng); trG, vaG = split(len(XG), cfg["val_frac"], rng)
        low = LowFid(d, cfg, "lf", THR_L); m = HighFid(d, low, cfg)
        groups = [dict(params=list(low.net.parameters()), lr=cfg["lf_lr"], weight_decay=cfg["lf_wd"])] + \
                 [g for g in _hf_groups(m, cfg, "a")] + [dict(params=[low.wf], lr=cfg["lf_lr"], weight_decay=_wd(cfg, cfg["lf_wd"]))]
        opt = torch.optim.AdamW([g for g in groups if g["params"]]); sch = _scheduler(opt, cfg["lf_sched"], cfg["lf_patience"])
        wl, wh = (cfg["lf_w_bce"], cfg["lf_w_mse"]), (cfg["hf_w_bce"], cfg["hf_w_mse"])
        best, state, bad = 1e9, copy.deepcopy(m.state_dict()), 0
        for _e in range(cfg["lf_epochs"]):
            m.train()
            for b in torch.randperm(len(trP)).split(cfg["lf_bs"]):
                xp = XP[trP[b]].clone(); g = torch.randint(len(trG), (min(cfg["hf_bs"], len(trG)),))
                xg = XG[trG[g]].clone()
                if cfg["lf_mask_p"]: xp[torch.rand(len(xp)) < cfg["lf_mask_p"], -NC:] = 0
                if cfg["hf_mask_p"]: xg[torch.rand(len(xg)) < cfg["hf_mask_p"], -NC:] = 0
                opt.zero_grad()
                (lam * crit(low, low(xp), yP[trP[b]], *wl) + crit(m, m(xg), yG[trG[g]], *wh)).backward(); opt.step()
            m.eval()
            with torch.no_grad(): v = (lam * crit(low, low(XP[vaP]), yP[vaP], *wl) + crit(m, m(XG[vaG]), yG[vaG], *wh)).item()
            if not np.isfinite(v): break
            if sch is not None: sch.step(v) if cfg["lf_sched"] == "plateau" else sch.step()
            if v < best - 1e-5: best, state, bad = v, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg["lf_patience"]: break
        m.load_state_dict(state); out.append(m)
    return out


def train_hf_only(X, y, rng, d, cfg):
    """Baseline trained on HF (GW) data only: a single net with the HF metal threshold (== train_gw_only)."""
    out = []
    for _ in range(cfg["ens"]):
        tr, va = split(len(X), cfg["val_frac"], rng); m = LowFid(d, cfg, "ho", THR_H)
        g = [dict(params=list(m.net.parameters()), lr=cfg["ho_lr"], weight_decay=cfg["ho_wd"]),
             dict(params=[m.wf], lr=cfg["ho_lr"], weight_decay=_wd(cfg, cfg["ho_wd"]))]
        out.append(fit(m, X[tr], y[tr], X[va], y[va], g, cfg["ho_bs"], cfg["ho_epochs"], cfg["ho_patience"], cfg["ho_sched"],
                       (cfg["ho_w_bce"], cfg["ho_w_mse"]), cfg["ho_mask_p"]))
    return out


def lf_key(cfg):
    """Hash of everything that determines the trained f_L ensemble (for the stage-2 cache)."""
    sub = {k: v for k, v in sorted(cfg.items()) if LF_KEYS(k) or k == "ens"}
    return hashlib.md5(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:10]
