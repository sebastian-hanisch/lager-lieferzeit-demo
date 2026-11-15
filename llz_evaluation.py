"""Auswertungen der Abschnitte der App und ihre Verdikte (reine Funktionen: (Art, Text)).

Jede Auswertung ruft nur den Kern `llz_dynamics` auf; die App legt sie hinter `st.cache_data`. Zahlen in den Verdikten kommen aus der Rechnung
oder aus `data/llz_frozen.json` (vorgerechnet von tools/make_frozen.py). Modell und Zeitlage: siehe llz_dynamics.py."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import llz_constants as C
import llz_dynamics as K

FROZEN_PATH = Path(__file__).resolve().parent / "data" / "llz_frozen.json"
CELL_SHARES = (0.11, 0.26, 0.52, 0.77, 0.91, 0.95, 0.99)   # alpha als Anteil an alpha_krit (keine Zweierpotenzen)
CELL_LS = (1, 2, 3, 6, 12)
TABLE_LS = (1, 2, 3, 6, 12)
SUM_SHARES = (0.21, 0.47, 0.83)
JUMP_ALPHAS = (0.1769, 0.1772)                              # L = 3, Schwelle 1 %: die Abklingzeit springt zwischen diesen beiden Werten
CONT_LS = (1, 2, 3, 4, 6, 8, 12, 24, 48, 100, 1000)
CONT_MAX = 200000                                           # größtes L der Dauerhaft-innerhalb-Prüfung


@dataclass(frozen=True)
class Settings:
    L: int = C.DEFAULT_L
    alpha: float = C.DEFAULT_ALPHA
    thr_pct: float = C.DEFAULT_THR

    @property
    def thr(self) -> float:
        return self.thr_pct / 100.0


# --- Formatierung ----------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit geschütztem Leerzeichen als Tausendertrenner (1479 -> 1 479)."""
    return f"{n:,}".replace(",", " ")


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


def thr_text(thr_pct: float) -> str:
    return de(thr_pct, 1 if thr_pct != int(thr_pct) else 0) + " %"


@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


# --- Abschnitt Stoßantwort -------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings) -> K.PulseAnalysis:
    return K.analyse_pulse(settings.L, settings.alpha, settings.thr)


def horizon_text(a: K.PulseAnalysis) -> str:
    return f"{fmt_int(a.horizon)} Perioden"


def verdict_pulse(a: K.PulseAnalysis) -> tuple[str, str]:
    """Verdikt der Stoßantwort als (Art, Text): rein, die App ruft st.<Art>(Text). Die Zweige hängen an der gemessenen Antwort, nicht an alpha < alpha*."""
    thr_pct = 100.0 * a.thr
    if a.regime == "unstable":
        return "error", (f"Kippt: ρ = {de(a.rho, 5)} > 1, α = {de(a.alpha, 3)} liegt über α_krit = {de(a.alpha_krit, 4)}. "
                         f"Die Auslenkung wächst je Periode um etwa den Faktor {de(a.rho, 4)}.")
    if a.regime == "boundary":
        return "warning", f"An der Grenze: ρ = 1 (auf 10⁻⁹), die Schwingung klingt nicht ab. Ihre Periode ist 4L − 2 = {K.period_krit(a.L)} Perioden."
    if a.decay is None:
        dec = f"ist nach {horizon_text(a)} noch nicht auf {thr_text(thr_pct)} der Spitze abgeklungen"
    else:
        dec = f"klingt in {fmt_int(a.decay)} Perioden auf {thr_text(thr_pct)} der Spitze ab"
    est = f" (Schätzung aus ρ: {de(a.estimate, 1)})" if a.estimate is not None and a.decay is not None else ""
    if a.peak > 1.0 + 1e-9:
        return "warning", f"Stabil, aber der Stoß wird verstärkt: max|i| = {de(a.peak, 2)} (der Stoß ist 1); die Abweichung {dec}{est}."
    if a.L == 1 and a.alpha > 1.0:
        return "info", (f"Überregeln: der Bestand springt über das Soll, das Vorzeichen wechselt je Periode (ρ = |1 − α| = {de(a.rho, 3)}); "
                        f"die Abweichung {dec}{est}.")
    if a.max_pos > K.OVERSHOOT_TOL:
        return "info", f"Stabil, schwingt: der Bestand steigt bis {de(a.max_pos, 4)} über das Soll; die Abweichung {dec}{est}."
    below = f" (α liegt unter α* = {de(a.alpha_star, 4)})" if a.alpha <= a.alpha_star * (1 + 1e-9) else ""
    return "success", f"Monoton{below}: kein Vorzeichenwechsel, der Bestand kehrt ohne Überschwinger zurück; die Abweichung {dec}{est}."


def sums_line(L: int, alpha: float) -> tuple[str, str, bool]:
    """(Σu, Σi, abgeklungen) als Text für die Kennzahlzeile; im instabilen Fall und an der Grenze „nicht definiert“."""
    s_i, s_u, done = K.pulse_sums(L, alpha)
    if s_i is None:
        return "nicht definiert", "nicht definiert", False
    return de(s_u, 4), de(s_i, 3), done


def verdict_shift(a: K.PulseAnalysis, e: float, i_end: float, T: int) -> tuple[str, str]:
    """Verdikt der dauerhaften Bedarfsverschiebung e: im stabilen Fall sitzt der Bestand bei −e/α (stationär: Bestellung = Bedarf)."""
    if a.regime != "stable":
        return "warning", "Dauerhaft verschobener Bedarf: hier gibt es keinen stationären Bestand, weil der Kreis nicht abklingt (ρ ≥ 1)."
    target = -e / a.alpha
    done = abs(i_end - target) <= 1e-3 * abs(target)
    tail = "" if done else " (noch nicht eingeschwungen, der Wert gilt zum Ende der Anzeige)"
    return "info", (f"Dauerhaft verschobener Bedarf e = {de(e, 2)} je Periode: der Bestand sitzt nach {fmt_int(T)} Perioden bei {de(i_end, 4)}, "
                    f"erwartet −e/α = {de(target, 4)}{tail}. Die Lieferzeit ändert diesen Endwert nicht, nur den Weg dorthin.")


# --- Abschnitt Grenze gegen L ----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def map_data() -> dict:
    """Farbfläche ρ(L, α) über L = 1 bis 40 und α (logarithmisch 0,01 bis 2,5), dazu α_krit aus der Formel und aus den Eigenwerten (Bisektion)."""
    Ls = list(range(C.L_MIN, C.L_MAX + 1))
    alphas = np.geomspace(0.01, C.ALPHA_MAX, C.MAP_ALPHAS)
    z = np.array([K.rho_grid(L, alphas) for L in Ls])
    formula = np.array([K.alpha_krit(L) for L in Ls])
    eigen = np.array([K.alpha_krit_from_eigenvalues(L) for L in Ls])
    return {"L": Ls, "alpha": alphas, "rho": z, "formula": formula, "eigen": eigen, "max_dev": float(np.abs(formula - eigen).max())}


def krit_strictly_falling(L_max: int = 60) -> bool:
    v = [K.alpha_krit(L) for L in range(1, L_max + 1)]
    return all(b < a for a, b in zip(v, v[1:]))


def verdict_limit(L: int, alpha: float, frozen: dict, falling: bool = True) -> tuple[str, str]:
    ak = K.alpha_krit(L)
    first = ", ".join(de(K.alpha_krit(k), 3 if k > 2 else 0) for k in range(1, 5))
    lage = "unter" if alpha < ak * (1 - 1e-9) else ("über" if alpha > ak * (1 + 1e-9) else "auf")
    parts = [f"α_krit = {de(ak, 3)} bei L = {L}; der gewählte Wert α = {de(alpha, 3)} liegt {lage} der Grenze ({pct(alpha / ak)} von α_krit)."]
    if falling:
        parts.append(f"Jede Periode mehr senkt die Grenze ({first} für L = 1 bis 4; gemessen streng fallend bis L = 60).")
    g, r = frozen["stability_grid"], frozen["random_pairs"]
    n_g = g["below"] + g["above"]
    exc = g["exceptions_below"] + g["exceptions_above"] + r["exceptions"]
    if exc == 0:
        parts.append(f"Vorgerechnet: stabil genau unterhalb der Grenze, 0 Ausnahmen in {fmt_int(n_g)} Gitterpunkten (L ≤ 12) und {fmt_int(r['n_pairs'])} Zufallspaaren "
                     f"(L ≤ 60, Zufalls-Seed {r['seed']}); für L > 60 nicht gemessen.")
    else:
        parts.append(f"Vorgerechnet: {exc} Ausnahmen von „stabil genau unterhalb der Grenze“ in {fmt_int(n_g + r['n_pairs'])} Paaren (L ≤ 12 Gitter, L ≤ 60 Zufall).")
    return ("error" if lage == "über" else "success" if lage == "unter" else "warning"), " ".join(parts)


# --- Abschnitt Periode und Wurzeln -----------------------------------------------------------------------------------------------------

def root_period(L: int) -> dict:
    """Wurzelwinkel und Periode knapp hinter der Grenze (alpha = alpha_krit (1 + 1e-6)): größter Wurzelbetrag, Winkel theta, Periode 2 pi / theta."""
    z = K.roots_p(L, K.alpha_krit(L) * (1.0 + 1e-6))
    w = z[int(np.argmax(np.abs(z)))]
    theta = abs(float(np.angle(w)))
    return {"L": L, "abs": float(abs(w)), "theta": theta, "period": 2.0 * math.pi / theta}


@lru_cache(maxsize=1)
def period_check(L_max: int = 40) -> dict:
    rows = [root_period(L) for L in range(2, L_max + 1)]
    dev = [abs(r["period"] / (4 * r["L"] - 2) - 1.0) for r in rows]
    return {"n": len(rows), "deviating": int(sum(d > 1e-3 for d in dev)), "max_rel_dev": float(max(dev)), "rows": rows}


@lru_cache(maxsize=1)
def residual_check(L_max: int = 40) -> float:
    """Größtes Residuum |P(e^{iθ})| bei alpha = alpha_krit und θ = π/(2L − 1) für L = 1 bis L_max."""
    worst = 0.0
    for L in range(1, L_max + 1):
        z = np.exp(1j * np.pi / (2 * L - 1))
        worst = max(worst, abs(z ** L - z ** (L - 1) + K.alpha_krit(L)))
    return float(worst)


def root_trace(L: int, n: int = 60, upto: float = 1.5):
    """Wurzeln von P für alpha von 0 bis upto · alpha_krit (n Werte): (Anteile, Feld der Form (n, L))."""
    shares = np.linspace(0.0, upto, n + 1)[1:]
    ak = K.alpha_krit(L)
    return shares, np.array([K.roots_p(L, s * ak) for s in shares])


def verdict_roots(L: int, alpha: float, check: dict) -> tuple[str, str]:
    ak = K.alpha_krit(L)
    P = K.period_krit(L)
    rho = K.spectral_radius(L, alpha)
    if L == 1:
        head = "Bei L = 1 gibt es nur die Wurzel z = 1 − α; sie erreicht bei α_krit = 2 den Wert −1 (Winkel π, Periode 2 = 4L − 2)."
    else:
        head = (f"Bei α_krit liegt ein Wurzelpaar auf dem Einheitskreis, Winkel π/(2L − 1) = {de(180.0 / (2 * L - 1), 1)}°, Periode 4L − 2 = {P} "
                f"(L = {L}, α_krit = {de(ak, 4)}).")
    tail = (f" Aus dem Wurzelwinkel gemessen weicht die Periode in {check['deviating']} von {check['n']} Fällen (L = 2 bis {check['n'] + 1}) um mehr als 0,1 % "
            f"von 4L − 2 ab (größte Abweichung {de(100 * check['max_rel_dev'], 4)} %). Beim gewählten α = {de(alpha, 3)} ist der größte Wurzelbetrag {de(rho, 5)}.")
    return ("success" if check["deviating"] == 0 else "warning"), head + tail


# --- Abschnitt Abklingzeit -------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def decay_cells() -> dict:
    """35 Zellen (L = 1, 2, 3, 6, 12 × 7 Anteile an alpha_krit), Schwelle 1 %: ρ, gemessene Abklingzeit, Schätzung ln(0,01)/ln(ρ)."""
    cells = []
    for L in CELL_LS:
        for s in CELL_SHARES:
            a = K.analyse_pulse(L, s * K.alpha_krit(L), 0.01)
            cells.append({"L": L, "share": s, "rho": a.rho, "measured": a.decay, "estimate": a.estimate,
                          "ratio": a.decay / a.estimate if a.decay is not None and a.estimate else None})
    ratios = [c["ratio"] for c in cells if c["ratio"] is not None]
    within = [c for c in cells if c["ratio"] is not None and abs(c["ratio"] - 1.0) <= 0.1]
    out = [c for c in cells if c not in within]
    out_ratios = [c["ratio"] for c in out]
    return {"cells": cells, "outside_ratio_min": min(out_ratios), "outside_near_star": sum(1 for c in out if c["share"] == 0.26), "n_outside": len(out), "n": len(cells), "within_10": len(within), "ratio_min": min(ratios), "ratio_max": max(ratios),
            "ratio_mean": float(np.mean(ratios)), "outside": [(c["L"], c["share"]) for c in out]}


@lru_cache(maxsize=1)
def jump_example() -> dict:
    """L = 3, Schwelle 1 %: zwei dicht benachbarte alpha mit sehr verschiedener Abklingzeit (die Nachschwingung kreuzt die Schwelle)."""
    out = {}
    for a in JUMP_ALPHAS:
        r = K.analyse_pulse(3, a, 0.01)
        out[a] = {"time": r.decay, "max_pos": r.max_pos}
    return out


@lru_cache(maxsize=32)
def decay_curve(L: int, thr: float) -> dict:
    """ρ(α), gemessene Abklingzeit und Schätzung über α in (0, 0,95 α_krit) (400 Punkte); Zeiten, die bis zum Horizont nicht abklingen, fehlen (None)."""
    ak = K.alpha_krit(L)
    grid = np.linspace(0.0, 0.95 * ak, C.STABLE_SAMPLE + 2)[1:-1]
    rho = K.rho_grid(L, grid)
    times = K.decay_times_grid(L, grid, thr, C.CURVE_HORIZON)
    est = np.array([K.decay_estimate(float(r), thr) or np.nan for r in rho])
    return {"alpha": grid, "rho": rho, "measured": np.where(times >= 0, times, np.nan), "estimate": est, "horizon": C.CURVE_HORIZON}


@lru_cache(maxsize=64)
def fastest(L: int, thr: float) -> dict:
    f = K.fastest_return(L, thr)
    f["share_lo"] = f["alpha_lo"] / f["alpha_star"]
    f["share_hi"] = f["alpha_hi"] / f["alpha_star"]
    f["near"] = K.analyse_pulse(L, 0.95 * K.alpha_krit(L), thr).decay
    return f


def verdict_decay(L: int, thr_pct: float, f: dict, jump: dict, cells: dict) -> tuple[str, str]:
    ast = K.alpha_star(L)
    rho_star = (L - 1.0) / L
    if L == 1:
        head = ("Bei L = 1 ist α* = 1 der Totbeat: Der Stoß ist nach einer Periode weg (ρ = 0). Die kürzeste Rückkehr im Gitter dauert "
                f"{f['t_min']} Periode ({de(f['alpha_lo'], 3)} bis {de(f['alpha_hi'], 3)}), die Rückkehr bei α* ebenso.")
    else:
        head = (f"ρ ist bei α* = {de(ast, 4)} am kleinsten (ρ = (L − 1)/L = {de(rho_star, 4)}), die Rückkehr dort aber nicht am kürzesten: "
                f"{f['t_star']} Perioden bei α*, {f['t_min']} Perioden im Gitter bei {de(f['share_lo'], 2)} bis {de(f['share_hi'], 2)} · α* "
                f"(Schwelle {thr_text(thr_pct)}, Gitter {fmt_int(f['n'])} Punkte).")
    a1, a2 = JUMP_ALPHAS
    j = (f" Die Zeit springt, wenn die Nachschwingung die Schwelle kreuzt (L = 3, Schwelle 1 %: α = {de(a1, 4)}: {jump[a1]['time']} Perioden, "
         f"α = {de(a2, 4)}: {jump[a2]['time']} Perioden).")
    near = (f" Nahe der Grenze klingt es kaum ab: bei 0,95 · α_krit dauert es für L = {L} {fmt_int(f['near'])} Perioden." if f["near"] is not None else "")
    est = (f" Die Schätzung ln(Schwelle)/ln ρ trifft in {cells['within_10']} von {cells['n']} Zellen (Schwelle 1 %, L = 1, 2, 3, 6, 12) auf ±10 %; "
           f"in den {cells['n_outside']} übrigen Zellen liegt die gemessene Zeit {de(100 * (cells['outside_ratio_min'] - 1), 0)} % bis "
           f"{de(100 * (cells['ratio_max'] - 1), 0)} % über der Schätzung ({cells['outside_near_star']} davon bei 0,26 · α_krit, nahe α*).")
    return "info", head + j + near + est


# --- Abschnitt Stetige Grenze ----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def continuous_summary() -> dict:
    Ls = np.arange(1, CONT_MAX + 1)
    t = K.continuous_table(Ls)
    big = (Ls >= 100) & (Ls <= 3200)
    slope = lambda rel: float(np.polyfit(np.log(Ls[big]), np.log(np.abs(rel[big])), 1)[0])  # noqa: E731
    return {
        "within_1": K.first_within(Ls, t["rel_L"], 0.01), "within_01": K.first_within(Ls, t["rel_L"], 0.001),
        "half_within_1": K.first_within(Ls, t["rel_L_half"], 0.01), "half_within_01": K.first_within(Ls, t["rel_L_half"], 0.001),
        "slope_L": slope(t["rel_L"]), "slope_half": slope(t["rel_L_half"]),
        "above_pi_half": bool((t["L"] > math.pi / 2).all()), "half_below_pi_half": bool((t["L_half"] < math.pi / 2).all()),
        "L_max": CONT_MAX,
    }


def continuous_rows(Ls=CONT_LS) -> list[dict]:
    t = K.continuous_table(np.array(Ls))
    return [{"L": int(Ls[k]), "alpha_krit": float(t["alpha_krit"][k]), "aL": float(t["L"][k]), "rel_L": float(t["rel_L"][k]),
             "a_half": float(t["L_half"][k]), "rel_half": float(t["rel_L_half"][k]), "a_m1": float(t["L_minus1"][k]),
             "rel_m1": float(t["rel_L_minus1"][k]), "inv": 1.0 / (2 * Ls[k] - 1)} for k in range(len(Ls))]


def verdict_continuous(s: dict) -> tuple[str, str]:
    return "info", (f"In diesem Modell (Zeitlage wie oben, geprüft bis L = {fmt_int(s['L_max'])}): α_krit · L liegt ab L = {s['within_1']} dauerhaft innerhalb 1 % von π/2 "
                    f"(ab L = {s['within_01']} innerhalb 0,1 %); der Abstand gleicht 1/(2L − 1) (Steigung {de(s['slope_L'], 2)} im Log-Log-Bild, L = 100 bis 3200). "
                    f"Mit L − ½ statt L liegt der Abstand schon ab L = {s['half_within_1']} innerhalb 1 % (ab L = {s['half_within_01']} innerhalb 0,1 %; Steigung {de(s['slope_half'], 2)}). "
                    f"α_krit · (L − 1) ist die Euler-Zahl 2m·sin(π/(4m + 2)) mit m = L − 1.")


# --- Ergebnistabelle und Gegenproben ---------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def results_table() -> list[dict]:
    """Bedarfsstoß, β = 0, s = 0; Abklingzeit auf 1 % der Spitze in Perioden (L = 1, 2, 3, 6, 12)."""
    rows = []
    for L in TABLE_LS:
        ak, ast = K.alpha_krit(L), K.alpha_star(L)
        f = K.fastest_return(L, 0.01)
        star = K.analyse_pulse(L, ast, 0.01)
        near = K.analyse_pulse(L, 0.95 * ak, 0.01)
        rows.append({"L": L, "alpha_krit": ak, "aL": ak * L, "period": K.period_krit(L), "alpha_star": ast, "rho_star": star.rho,
                     "t_star": star.decay, "t_min": f["t_min"], "t_near": near.decay})
    return rows


@lru_cache(maxsize=1)
def sums_check() -> dict:
    """Σu = 1 und Σi = −1/α nach einem Stoß: L = 1 bis 12, α = 0,21 / 0,47 / 0,83 · α_krit (36 Fälle)."""
    dev_u = dev_i = 0.0
    n = 0
    for L in range(1, 13):
        for s in SUM_SHARES:
            a = s * K.alpha_krit(L)
            s_i, s_u, done = K.pulse_sums(L, a)
            assert done
            dev_u = max(dev_u, abs(s_u - 1.0))
            dev_i = max(dev_i, abs(a * s_i + 1.0))
            n += 1
    return {"n": n, "dev_u": dev_u, "dev_i": dev_i}
