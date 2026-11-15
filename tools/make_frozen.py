"""Erzeugt `data/llz_frozen.json`: die Messreihen, die die App nicht live rechnet (Laufzeit einige Sekunden bis eine Minute).

Aufruf (Wurzelordner der Demo): _venvs/test/Scripts/python.exe -X utf8 tools/make_frozen.py
Inhalt (Modell wie in llz_dynamics.py: beta = 0, s = 0, ohne Klemmung, Bedarfsstoß e_0 = 1, Anfangszustand 0, Zähleinheit Periode):
  stability_grid   Stabilitätsgebiet = (0, alpha_krit)? Gitter von 8000 alpha in (0, 4), L = 1 bis 12: Zählung der Ausnahmen
  random_pairs     dieselbe Frage für 20000 Zufallspaare (L gleichverteilt 1 bis 60, alpha gleichverteilt in (0, 4)), Zufalls-Seed 15
  below_star       5000 Zufallspaare (L = 2 bis 40, alpha zufällig in (0, 0,99 alpha*)): wie viele haben einen Bestand über Soll (Überschwinger)?
  rho_monotone     Spektralradius fällt streng links und steigt streng rechts von alpha* (L = 2 bis 12, 2000 + 2000 Gitterpunkte)
  amplification    Gitter 20000 Punkte in (0, alpha_krit): erster Überschwinger und erste Verstärkung des Stoßes als Anteil an alpha_krit (L = 2 bis 12)
Die Zufallszahlen kommen aus dem SplitMix64 der Demo (llz_rng.py), nie aus Python-`hash()`."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llz_dynamics as K  # noqa: E402
from llz_rng import uniform_array  # noqa: E402

SEED = 15


def stability_grid() -> dict:
    n = 8000
    grid = np.linspace(0.0, 4.0, n + 2)[1:-1]
    below = above = bad_below = bad_above = 0
    for L in range(1, 13):
        ak = K.alpha_krit(L)
        rho = K.rho_grid(L, grid)
        lo = grid < ak
        below += int(lo.sum())
        above += int((~lo).sum())
        bad_below += int((rho[lo] >= 1.0).sum())
        bad_above += int((rho[~lo] <= 1.0).sum())
    return {"n_alpha": n, "L_max": 12, "below": below, "above": above, "exceptions_below": bad_below, "exceptions_above": bad_above}


def random_pairs() -> dict:
    n = 20000
    u = uniform_array(SEED, 2 * n)
    L = np.floor(u[:n] * 60).astype(int) + 1
    alpha = 4.0 * u[n:]
    below = above = bad = skipped = 0
    for ell in range(1, 61):
        sel = L == ell
        a = alpha[sel]
        ak = K.alpha_krit(ell)
        keep = np.abs(a / ak - 1.0) >= 1e-6
        skipped += int((~keep).sum())
        a = a[keep]
        rho = K.rho_grid(ell, a)
        lo = a < ak
        below += int(lo.sum())
        above += int((~lo).sum())
        bad += int((rho[lo] >= 1.0).sum()) + int((rho[~lo] <= 1.0).sum())
    return {"seed": SEED, "n_pairs": n, "L_max": 60, "below": below, "above": above, "skipped": skipped, "exceptions": bad}


def below_star() -> dict:
    n, T = 5000, 400
    u = uniform_array(SEED + 1, 2 * n)
    L = np.floor(u[:n] * 39).astype(int) + 2
    frac = u[n:]
    worst = 0.0
    positive = 0
    for ell in range(2, 41):
        sel = L == ell
        a = frac[sel] * 0.99 * K.alpha_star(ell)
        i = K.pulse_grid(ell, a, T)
        mx = i.max(axis=0)
        positive += int((mx > K.OVERSHOOT_TOL).sum())
        worst = max(worst, float(mx.max()))
    return {"seed": SEED + 1, "n_pairs": n, "L_min": 2, "L_max": 40, "T": T, "positive": positive, "largest_max_i": worst}


def rho_monotone() -> dict:
    out = {}
    for L in range(2, 13):
        ast = K.alpha_star(L)
        left = np.linspace(0.0, ast, 2002)[1:-1]
        right = np.linspace(ast, K.alpha_krit(L), 2002)[1:-1]
        dl = np.diff(K.rho_grid(L, left))
        dr = np.diff(K.rho_grid(L, right))
        out[str(L)] = {"left_largest_step": float(dl.max()), "right_smallest_step": float(dr.min())}
    return {"n_each_side": 2000, "L": out}


def amplification() -> dict:
    n = 20000
    out = {}
    for L in range(2, 13):
        ak = K.alpha_krit(L)
        grid = np.linspace(0.0, ak, n + 2)[1:-1]
        T = 30 * L + 300
        i = K.pulse_grid(L, grid, T)
        mx = i.max(axis=0)
        peak = np.abs(i).max(axis=0)
        over = np.nonzero(mx > K.OVERSHOOT_TOL)[0]
        amp = np.nonzero(peak > 1.0 + 1e-9)[0]
        out[str(L)] = {
            "first_overshoot_share": float(grid[over[0]] / ak) if over.size else None,
            "first_amplification_share": float(grid[amp[0]] / ak) if amp.size else None,
        }
    return {"n_grid": n, "per_L": out}


def main() -> None:
    result = {
        "stability_grid": stability_grid(),
        "random_pairs": random_pairs(),
        "below_star": below_star(),
        "rho_monotone": rho_monotone(),
        "amplification": amplification(),
    }
    out = ROOT / "data" / "llz_frozen.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
