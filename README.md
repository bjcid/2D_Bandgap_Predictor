# 2D Material Bandgap Predictor (PBE + GW)

Streamlit app that estimates the PBE and GW (quasiparticle) bandgaps of 2D materials from a VASP CONTCAR or POSCAR file,
using a multi-fidelity neural network (PBE MLP -> GW head).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder's contents to a GitHub repository (files at the repo root).
2. share.streamlit.io -> New app -> select repo, branch, main file `app.py`.
3. In Advanced settings choose **Python 3.12**. The models were trained and pickled with Python 3.12.10, so keep the pinned
   `scikit-learn` and `torch` versions in `requirements.txt`.
