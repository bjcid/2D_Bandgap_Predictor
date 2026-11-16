"""C2DB connector: monolayer PBE gaps for the layers that make up a bilayer CONTCAR.
Usage: python c2db.py   (builds c2db_cache.json + c2db_gaps.csv for every CONTCAR)"""
import json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from ase import Atoms

ROOT = Path(__file__).parent
BASE, CACHE = "https://c2db.fysik.dtu.dk", ROOT / "c2db_cache.json"
MAX_AREA_ERR, MAX_THICK_ERR = 0.12, 0.6  # relative area/atom, thickness [A]
_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {"struct": {}}
_cache.setdefault("search2", {})
_http = requests.Session()


def save_cache():
    CACHE.write_text(json.dumps(_cache))


def split_layers(atoms):
    """Split a bilayer into two monolayers (cut at the vacuum, then at the interlayer gap)."""
    n, c = len(atoms), atoms.cell.lengths()[2]
    z = atoms.get_scaled_positions()[:, 2] % 1.0
    zs = np.sort(z); gaps = np.diff(np.r_[zs, zs[0] + 1]); start = zs[(np.argmax(gaps) + 1) % n]  # first atom above vacuum
    order = np.argsort((z - start) % 1.0); zo = ((z - start) % 1.0)[order]
    comp = lambda idx: sorted(atoms.numbers[idx])
    half = n // 2
    if n % 2 == 0 and comp(order[:half]) == comp(order[half:]): k = half  # homobilayer
    else: k = int(np.argmax(np.diff(zo))) + 1                              # heterobilayer: largest interior gap
    return [make_layer(atoms, order[:k]), make_layer(atoms, order[k:])]


def make_layer(atoms, idx):
    s = atoms[list(idx)].copy(); s.set_pbc([True, True, True]); return s


def descriptor(numbers, pos, cell):
    """(area per atom, thickness, composition key) of a 2D layer."""
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    zz = np.sort(pos[:, 2]); thick = float(zz[-1] - zz[0])
    if thick > 0.5 * cell[2][2]:  # wrapped across periodic z: unwrap by the largest gap
        zf = np.sort((pos[:, 2] / cell[2][2]) % 1.0); g = np.diff(np.r_[zf, zf[0] + 1]); thick = (1 - g.max()) * cell[2][2]
    return area / len(numbers), thick


def reduced(atoms):
    return atoms.get_chemical_formula(mode="hill", empirical=True)


_sid, _lock = None, threading.Lock()
FIELDS = {"Band gap (PBE)": "gap", "Band gap (HSE06)": "gap_hse", "Band gap (G": "gap_gw",
          "Energy above hull": "ehull", "Heat of formation": "hform"}


def _session_id():
    """Fresh C2DB table session with HSE06 and G0W0 columns enabled."""
    global _sid
    with _lock:
        if _sid is None:
            sid = re.search(r"sid=(\d+)", _http.get(BASE + "/", timeout=40).text).group(1)
            for t in ("gap_hse", "gap_gw"): _http.get(f"{BASE}/table", params={"sid": sid, "toggle": t}, timeout=40)
            _sid = sid
    return _sid


def _num(x):
    try: return float(x)
    except ValueError: return None


def search(formula):
    """C2DB rows [{uid, gap, gap_hse, gap_gw, ehull, hform}] for a (reduced) formula."""
    if formula in _cache["search2"]: return _cache["search2"][formula]
    html = _http.get(f"{BASE}/table", params={"sid": _session_id(), "filter": formula}, timeout=40).text
    clean = lambda c: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()
    head = [clean(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", html.split("</thead>")[0], re.S)]
    cols = {i: v for i, h in enumerate(head) for k, v in FIELDS.items() if h.startswith(k)}
    rows = []
    for tr in html.split("</thead>")[1].split("<tr")[1:]:
        uid = re.search(r"/material/([^ \"'>]+)", tr)
        cells = [clean(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", tr, re.S)]
        if uid and len(cells) == len(head): rows.append({"uid": uid.group(1), **{v: _num(cells[i]) for i, v in cols.items()}})
    _cache["search2"][formula] = rows; return rows


def structure(uid):
    if uid not in _cache["struct"]:
        d = _http.get(f"{BASE}/material/{uid}/download/json", timeout=40).json()["1"]
        num, pos, cell = np.array(d["numbers"]), np.array(d["positions"]), np.array(d["cell"])
        _cache["struct"][uid] = list(descriptor(num, pos, cell)) + [len(num)]
    return _cache["struct"][uid]


def match_layer(layer):
    """Best C2DB monolayer for a layer -> dict(uid, gap, gap_hse, gap_gw, ehull, hform, err) or None."""
    try:
        cands = [r for r in search(reduced(layer)) if r.get("gap") is not None]
        a, t = descriptor(layer.numbers, layer.positions, layer.cell.array)
        best = None
        for r in cands:
            ca, ct, _ = structure(r["uid"])
            err = abs(ca - a) / a + abs(ct - t) / 3.0
            if abs(ca - a) / a < MAX_AREA_ERR and abs(ct - t) < MAX_THICK_ERR and (best is None or err < best["err"]):
                best = {**r, "err": float(err)}
        return best
    except Exception:  # network / parse errors -> treated as "not found"
        return None


def layer_gaps(atoms):
    """[record dict or None] for the two monolayers of a bilayer."""
    out = []
    for l in split_layers(atoms):
        m = match_layer(l); out.append(m)
    return out


def main():
    from featurize import read_structure
    files = {int(m.group(1)): p for p in (ROOT / "CONTCAR_files").iterdir() if (m := re.match(r"(\d+)_?CONTCAR$", p.name))}
    ids = sorted(files)
    def work(k):
        try: return k, layer_gaps(read_structure(files[k]))
        except Exception as e: return k, [None, None]
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, (k, g) in enumerate(ex.map(work, ids)):
            rows.append({"id": k, **{f"{f}{j+1}": (x or {}).get(f) for j, x in enumerate(g) for f in ["uid", *FIELDS.values()]}})
            if i % 100 == 0: print(i, len(ids), flush=True); save_cache()
    save_cache()
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(ROOT / "c2db_gaps.csv", index=False)
    print(f"matched layer 1: {df.gap1.notna().sum()}/{len(df)}; HSE {df.gap_hse1.notna().sum()}, GW {df.gap_gw1.notna().sum()}")


if __name__ == "__main__":
    main()
