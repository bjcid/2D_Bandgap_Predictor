"""Streamlit app: predict PBE and GW band gaps of 2D materials from a CONTCAR."""
import io, json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from ase.io import write

from featurize import featurize_atoms, load_artifacts, read_structure
import c2db
import mf_v2
from train_mf import c2db_features, load_ensemble, predict as ens_predict

ROOT = Path(__file__).parent
st.set_page_config(page_title="2D Bandgap Predictor", page_icon="🔬", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#1f3c88 0%,#4f46e5 55%,#7c3aed 100%);padding:2rem 2.2rem;border-radius:16px;color:#fff;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.1rem;color:#fff}.hero p{margin:.4rem 0 0;opacity:.9;font-size:1.02rem}
div[data-testid="stMetric"]{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,.06)}
div[data-testid="stMetricValue"]{font-size:2rem;font-weight:700}
.stTabs [data-baseweb="tab"]{font-weight:600}
.footer{opacity:.6;font-size:.85rem;margin-top:2rem;text-align:center}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    ck = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)
    model = load_ensemble(ck)
    return model, load_artifacts(ROOT), ck["metrics"], ck.get("use_c2db", False)


@st.cache_resource(show_spinner="Loading structure-only model…")
def load_structure_only():
    """Model trained WITHOUT C2DB inputs (train_mf.py --fallback-only): the answer when no C2DB monolayer is found."""
    f = ROOT / "model_noc2db.pt"
    if not f.exists(): return None
    ck = torch.load(f, map_location="cpu", weights_only=False)
    return load_ensemble(ck), ck["metrics"]


@st.cache_resource(show_spinner="Loading tuned model…")
def load_tuned():
    return mf_v2.load_v2(ROOT) if mf_v2.available(ROOT) else None


@st.cache_data(show_spinner="Querying C2DB for the monolayers…", ttl=86400)
def lookup_c2db(numbers, positions, cell):
    from ase import Atoms
    try:
        recs = c2db.layer_gaps(Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True))
        return [{k: v for k, v in r.items() if k != "err"} if r else None for r in recs]
    except Exception:
        return [None, None]


def predict(atoms, model, art, layers=None, use_c2db=None):
    x = featurize_atoms(atoms, *art)
    if USE_C2DB if use_c2db is None else use_c2db: x = np.hstack([x, np.array([c2db_features(layers or [None, None])], dtype=np.float32)])
    x = torch.tensor(x)
    pbe = ens_predict([m.low for m in model], x)[0][0]
    gw, p_metal = ens_predict(model, x)
    return float(pbe), float(gw[0]), float(p_metal[0])


def layer_group(atoms, symprec=1e-2):
    """(Hermann-Mauguin symbol, number) of the layer group (periodic in a,b; vacuum along c)."""
    try:
        import spglib
        r = spglib.get_symmetry_layerdataset((atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers), aperiodic_dir=2, symprec=symprec)
        return r.international, str(r.number)
    except Exception:
        return "n/a", "n/a"


def view_3d(atoms, rep=(2, 2, 1)):
    import py3Dmol
    buf = io.StringIO(); write(buf, atoms.repeat(rep), format="xyz")
    v = py3Dmol.view(width=620, height=460)
    v.addModel(buf.getvalue(), "xyz"); v.setStyle({"stick": {"radius": .15}, "sphere": {"scale": .28}})
    v.zoomTo(); return v._make_html()


try:
    model, art, M, USE_C2DB = load_model()
except Exception as e:
    st.error(f"Model files not found or unreadable ({e}). Run `python featurize.py` and `python train_mf.py` first."); st.stop()

try:
    V2 = load_tuned()   # tuned model (HPO); used only when C2DB monolayers are found, the original model is the fallback
except Exception as e:
    V2 = None; st.warning(f"Tuned model could not be loaded ({e}); using the original model.")

try:
    FB = load_structure_only()   # structure-only model (no C2DB inputs); used when no C2DB monolayer is found
except Exception as e:
    FB = None; st.warning(f"Structure-only model could not be loaded ({e}); the original model with empty C2DB inputs is used instead.")

st.markdown('<div class="hero"><h1>2D Material Bandgap Predictor</h1>'
            '<p>Upload a VASP <b>CONTCAR</b> or <b>POSCAR</b> file to estimate the PBE and GW (quasiparticle) bandgaps in seconds.</p></div>',
            unsafe_allow_html=True)

up = st.file_uploader("Upload a CONTCAR / POSCAR file", help="VASP 5 format, vacuum along the c axis.")
if up is None:
    st.info("Upload a structure to get started. Predictions are most reliable for monolayers similar to the training set.")
    st.stop()

try:
    atoms = read_structure(io.StringIO(up.getvalue().decode("utf-8", errors="replace")))
    layers = lookup_c2db(atoms.numbers, atoms.positions, atoms.cell.array) if USE_C2DB else None
    found = bool(layers) and any(layers)
    use_v2 = V2 is not None and found
    use_fb = not use_v2 and not found and FB is not None   # no C2DB monolayer: structure-only model
    if use_v2: pbe, gw, p_metal = mf_v2.predict_v2(atoms, layers, V2)
    elif use_fb: pbe, gw, p_metal = predict(atoms, FB[0], art, None, use_c2db=False)
    else: pbe, gw, p_metal = predict(atoms, model, art, layers)
    pbe, gw = (float(v) for v in mf_v2.apply_metal_rule(pbe, gw))  # metal at one level (gap exactly 0) -> metal at both
except Exception as e:
    st.error(f"Could not process this file as a valid CONTCAR: {e}"); st.stop()

st.subheader(atoms.get_chemical_formula())
m1, m2, m3 = st.columns(3)
m1.metric("PBE bandgap", f"{pbe:.2f} eV"); m2.metric("GW bandgap", f"{gw:.2f} eV")
m3.metric("GW − PBE correction", f"{gw - pbe:+.2f} eV", delta=f"{(gw / pbe - 1) * 100:.0f}%" if pbe > 0.05 and gw > 0 else None)
if gw == 0:
    st.info(f"Predicted **metallic** at the GW level (gap = 0 eV, metal probability {p_metal:.0%}).")
if use_v2:
    st.caption(f"Tuned model v2. Typical error: ±{V2['meta']['metrics'].get('test_pbe_rmse', float('nan')):.2f} eV (PBE, test RMSE), "
               f"±{V2['meta']['metrics']['cv_gw_rmse_tuned'][0]:.2f} eV (GW, cross-validated RMSE).")
elif use_fb:
    st.caption(f"Structure-only model (no C2DB inputs). Typical test error (RMSE): ±{FB[1]['pbe']['test']['RMSE']:.2f} eV (PBE), ±{FB[1]['gw']['test']['Multi-fidelity']['RMSE']:.2f} eV (GW).")
else:
    st.caption(f"Original model. Typical test error (RMSE): ±{M['pbe']['test']['RMSE']:.2f} eV (PBE), ±{M['gw']['test']['Multi-fidelity']['RMSE']:.2f} eV (GW).")

if USE_C2DB:
    if any(layers):
        st.success("C2DB monolayer PBE bandgaps used as extra inputs: " + " · ".join(
            f"layer {i+1}: [{l['uid']}]({c2db.BASE}/material/{l['uid']}) = {l['gap']:.2f} eV" if l else f"layer {i+1}: not found"
            for i, l in enumerate(layers)))
    else:
        st.warning("No matching monolayer found in C2DB (or C2DB is unreachable). Prediction uses structure features only "
                   "(a model trained without C2DB inputs; the tuned model needs them), so it is less accurate.")

t1, t2, t3 = st.tabs(["🧊 Structure", "⚛️ Composition", "📐 Lattice"])
with t1:
    components.html(view_3d(atoms), height=480)
    st.caption("2×2 supercell shown. Drag to rotate, scroll to zoom.")
with t2:
    cnt = Counter(atoms.get_chemical_symbols())
    st.dataframe(pd.DataFrame({"Element": cnt.keys(), "Atoms": cnt.values()}), hide_index=True)
with t3:
    a, b, c = atoms.cell.lengths(); al, be, ga = atoms.cell.angles()
    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    lg = layer_group(atoms)
    st.dataframe(pd.DataFrame({"Parameter": ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)", "Area (Å²)", "Atoms", "Density (g/cm³, cell)",
                                             "Layer group", "Layer group number"],
                               "Value": [f"{v:.3f}" for v in [a, b, c, al, be, ga, area]] + [str(len(atoms)),
                                         f"{atoms.get_masses().sum() * 1.66054 / atoms.get_volume():.3f}", lg[0], lg[1]]}),
                 hide_index=True)

st.markdown('<div class="footer">Multi-fidelity model trained on PBE/GW DFT data · predictions are estimates, not substitutes for DFT.</div>',
            unsafe_allow_html=True)
