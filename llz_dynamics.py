"""Kern der Demo (nur numpy und Standardbibliothek): ein Lager mit der Bestellregel u = -alpha * Bestandsabweichung und Lieferzeit L.

Zeitlage (gilt für jede Zahl der Demo): i_{t+1} = i_t + u_{t+1-L} - e_t mit u_t = -alpha * i_t, also
    i_{t+1} = i_t - alpha * i_{t+1-L} - e_t.
Eine Bestellung der Periode t steht im Bestand der Periode t + L. L = 1 heißt „eine Periode Wirkverzug“, nicht „keine Lieferzeit“; L = 0 gibt es nicht.
Das charakteristische Polynom ist P(z) = z^L - z^(L-1) + alpha. Kein Leitungsgewicht, kein Meldeverzug, keine Klemmung (Stücke 2 und 3 der Linie).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Formeln          alpha_krit, period_krit, alpha_star, alpha_star_exact (Brüche)
  Eigenwerte       char_poly, state_matrix, roots_p, spectral_radius, rho_grid
  Antworten        simulate (beliebiger Bedarf), pulse, pulse_grid (vektorisiert über alpha)
  Abklingzeit      decay_time, decay_times_grid, analyse_pulse, fastest_return
  Summenregeln     pulse_sums
  Stetige Grenze   continuous_table

Der Euler-Bezug (i_{t+1} = i_t - alpha * i_{t-(L-1)} ist das Euler-Schema mit m = L - 1 Schritten je Verzögerung) steht als Verweis auf
`verzoegerung-hopf-demo` (Befund B4), nicht als zweite Messung."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

RHO_TOL = 1e-9          # |rho - 1| <= RHO_TOL gilt als „an der Grenze“ (nie mit == vergleichen: Rundung)
OVERSHOOT_TOL = 1e-12   # ein Bestand über Soll von höchstens 1e-12 gilt als kein Überschwinger
HORIZON_CAP = 40000     # längster Horizont der Abklingzeit-Messung (Perioden); sonst „noch nicht abgeklungen“
SUMS_CAP = 60000        # längster Horizont der Summenregel
BLOW_UP = 1e12          # ab diesem Betrag bricht die Simulation einer instabilen Antwort ab


# --- Formeln ---------------------------------------------------------------------------------------------------------------------------

def alpha_krit(L: int) -> float:
    """Stabilitätsgrenze 2 sin(pi / (4L - 2)): eigene Herleitung in diesem Modell (Wurzel auf dem Einheitskreis: |z - 1| = alpha, (L - 1/2) theta = pi/2)."""
    return 2.0 * math.sin(math.pi / (4 * L - 2))


def period_krit(L: int) -> int:
    """Periode der Schwingung an der Grenze: 2 pi / theta = 4L - 2 mit theta = pi / (2L - 1)."""
    return 4 * L - 2


def alpha_star(L: int) -> float:
    """Doppelwurzel von P bei z* = (L - 1)/L: alpha* = (1/L) ((L - 1)/L)^(L - 1). L = 1: alpha* = 1 (Totbeat)."""
    return (1.0 / L) * ((L - 1.0) / L) ** (L - 1)


def alpha_star_exact(L: int) -> Fraction:
    """alpha* als Bruch (für exakte Proben P(z*) = P'(z*) = 0)."""
    return Fraction(1, L) * Fraction(L - 1, L) ** (L - 1)


# --- Eigenwerte ------------------------------------------------------------------------------------------------------------------------

def char_poly(L: int, alpha: float) -> np.ndarray:
    """Koeffizienten von P(z) = z^L - z^(L-1) + alpha (höchste Potenz zuerst); L = 1: z - 1 + alpha."""
    c = np.zeros(L + 1)
    c[0] = 1.0
    c[1] += -1.0
    c[L] += alpha
    return c


def state_matrix(L: int, alpha: float) -> np.ndarray:
    """Zustandsmatrix für x_t = (i_t, i_{t-1}, ..., i_{t-L+1}); L = 1: das Skalar 1 - alpha."""
    m = np.zeros((L, L))
    m[0, 0] += 1.0
    m[0, L - 1] += -alpha
    for r in range(1, L):
        m[r, r - 1] = 1.0
    return m


def roots_p(L: int, alpha: float) -> np.ndarray:
    """Alle L Wurzeln von P (Eigenwerte der Zustandsmatrix)."""
    return np.linalg.eigvals(state_matrix(L, alpha))


def spectral_radius(L: int, alpha: float) -> float:
    return float(np.max(np.abs(roots_p(L, alpha))))


def _matrix_stack(L: int, alphas: np.ndarray) -> np.ndarray:
    m = np.zeros((alphas.size, L, L))
    m[:, 0, 0] += 1.0
    m[:, 0, L - 1] += -alphas
    for r in range(1, L):
        m[:, r, r - 1] = 1.0
    return m


def rho_grid(L: int, alphas) -> np.ndarray:
    """Spektralradius für viele alpha zugleich (Eigenwerte gestapelter Zustandsmatrizen)."""
    a = np.asarray(alphas, dtype=float)
    return np.max(np.abs(np.linalg.eigvals(_matrix_stack(L, a))), axis=1)


def alpha_krit_from_eigenvalues(L: int, lo: float = 0.0, hi: float = 4.0, steps: int = 60) -> float:
    """Gegenprobe der Formel: Bisektion auf rho(alpha) = 1 (rho < 1 darunter, rho > 1 darüber; das ist die gemessene Aussage M2)."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if spectral_radius(L, mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Antworten -------------------------------------------------------------------------------------------------------------------------

def simulate(L: int, alpha: float, e, blow_up: float = BLOW_UP) -> np.ndarray:
    """Bestandsabweichung i_0 ... i_T für die Bedarfsabweichung e_0 ... e_{T-1}; Anfangszustand 0 (i_t = 0 für t <= 0).
    Wächst der Betrag über `blow_up`, endet die Folge dort (kürzeres Feld)."""
    e = list(map(float, e))
    T = len(e)
    i = [0.0] * (T + 1)
    for t in range(T):
        k = t + 1 - L
        prev = i[k] if k >= 0 else 0.0
        v = i[t] - alpha * prev - e[t]
        i[t + 1] = v
        if abs(v) > blow_up:
            return np.array(i[: t + 2])
    return np.array(i)


def pulse(L: int, alpha: float, T: int, e0: float = 1.0, blow_up: float = BLOW_UP) -> np.ndarray:
    """Stoßantwort: Bedarfsstoß e_0 = e0 in Periode 0, danach 0. Also i_1 = -e0."""
    e = [e0] + [0.0] * (T - 1)
    return simulate(L, alpha, e, blow_up)


def pulse_grid(L: int, alphas, T: int) -> np.ndarray:
    """Stoßantwort i_0 ... i_T für viele alpha zugleich, Form (T + 1, n). Nur für alpha unterhalb der Grenze gedacht (sonst Überlauf)."""
    a = np.asarray(alphas, dtype=float)
    i = np.zeros((T + 1, a.size))
    if T >= 1:
        i[1] = -1.0
    with np.errstate(all="ignore"):
        for t in range(1, T):
            k = t + 1 - L
            i[t + 1] = i[t] - (a * i[k] if k >= 0 else 0.0)
    return i


# --- Abklingzeit -----------------------------------------------------------------------------------------------------------------------

def decay_time(i, thr: float) -> int:
    """Letzte Periode t mit |i_t| > thr * max|i| (Anteil thr, z. B. 0,01). Die Schwelle ist eine Wahl, keine Naturgröße."""
    a = np.abs(np.asarray(i, dtype=float))
    above = np.nonzero(a > thr * a.max())[0]
    return int(above.max()) if above.size else 0


def decay_estimate(rho: float, thr: float):
    """Schätzung ln(thr) / ln(rho); nur für 0 < rho < 1 definiert, sonst None. Sie ist nur eine Schätzung (siehe README)."""
    if not 0.0 < rho < 1.0:
        return None
    return math.log(thr) / math.log(rho)


def decay_times_grid(L: int, alphas, thr: float, T: int) -> np.ndarray:
    """Abklingzeit für viele alpha (Stoßantwort, Horizont T); -1 heißt „bis zum Horizont nicht abgeklungen“ (letzter Überschreitungspunkt im
    letzten Fünftel). Nur für alpha unterhalb der Grenze."""
    i = np.abs(pulse_grid(L, alphas, T))
    peak = i.max(axis=0)
    above = i > thr * peak
    last = (i.shape[0] - 1) - np.argmax(above[::-1], axis=0)
    last = np.where(above.any(axis=0), last, 0)
    return np.where(last < 0.8 * T, last, -1).astype(int)


@dataclass(frozen=True)
class PulseAnalysis:
    L: int
    alpha: float
    thr: float                 # Anteil (0,01 = 1 %)
    alpha_krit: float
    alpha_star: float
    share: float               # alpha / alpha_krit
    rho: float
    regime: str                # "stable", "boundary" oder "unstable"
    i: np.ndarray              # i_0 ... i_T (bei Instabilität bis zum Abbruch)
    peak: float                # max |i_t|
    max_pos: float             # größter Bestand über Soll (0 ohne Überschwinger)
    decay: int | None          # Abklingzeit in Perioden, None = bis zum Horizont nicht abgeklungen oder instabil
    estimate: float | None     # ln(thr)/ln(rho)
    horizon: int               # gerechnete Perioden


def classify(rho: float) -> str:
    if rho > 1.0 + RHO_TOL:
        return "unstable"
    if rho >= 1.0 - RHO_TOL:
        return "boundary"
    return "stable"


def _initial_horizon(L: int, rho: float, thr: float) -> int:
    est = decay_estimate(rho, thr)
    return int(2.5 * (est if est is not None else 0.0)) + 12 * L + 60


def analyse_pulse(L: int, alpha: float, thr: float, cap: int = HORIZON_CAP) -> PulseAnalysis:
    """Stoßantwort, Spektralradius und Abklingzeit. Der Horizont wächst, bis die Antwort sicher unter die Schwelle gefallen ist; bis zur Obergrenze
    `cap` nicht abgeklungen heißt decay = None (nie eine erfundene Zahl)."""
    ak, ast = alpha_krit(L), alpha_star(L)
    rho = spectral_radius(L, alpha)
    regime = classify(rho)
    if regime != "stable":   # kippt oder Schwingung an der Grenze: keine Abklingzeit, 200 Perioden zur Anzeige
        i = pulse(L, alpha, 200)
        return PulseAnalysis(L, alpha, thr, ak, ast, alpha / ak, rho, regime, i, float(np.abs(i).max()), float(max(i.max(), 0.0)), None, None, len(i) - 1)
    T = min(cap, _initial_horizon(L, rho, thr))
    while True:
        i = pulse(L, alpha, T)
        a = np.abs(i)
        peak = float(a.max())
        idx = int(np.nonzero(a > thr * peak)[0].max())
        if idx < 0.8 * T or T >= cap:
            break
        T = min(cap, 2 * T)
    decay = idx if idx < 0.8 * T else None
    return PulseAnalysis(L, alpha, thr, ak, ast, alpha / ak, rho, regime, i, peak, float(max(i.max(), 0.0)), decay, decay_estimate(rho, thr), T)


def fastest_return(L: int, thr: float, n: int = 4000) -> dict:
    """Kürzeste Abklingzeit über ein Gitter von n Punkten in (0, alpha_krit (1 - 1e-3)) und die Zeit bei alpha* (Stoßantwort, Schwelle thr).
    Die Zeit ist keine glatte Funktion von alpha: die Lage des Minimums gilt nur für dieses Gitter und diese Schwelle."""
    ak = alpha_krit(L)
    grid = np.linspace(0.0, ak * (1.0 - 1e-3), n + 2)[1:-1]
    est = decay_estimate((L - 1.0) / L, thr) if L > 1 else 0.0
    T = int(3.0 * est) + 100
    times = decay_times_grid(L, grid, thr, T)
    valid = times >= 0
    tmin = int(times[valid].min())
    at_min = grid[valid & (times == tmin)]
    ast = alpha_star(L)
    star = analyse_pulse(L, ast, thr)
    return {"L": L, "thr": thr, "n": n, "t_min": tmin, "alpha_lo": float(at_min.min()), "alpha_hi": float(at_min.max()),
            "alpha_star": ast, "t_star": star.decay, "formula": est if L > 1 else None}


# --- Summen und Dauerverschiebung ------------------------------------------------------------------------------------------------------

def pulse_sums(L: int, alpha: float, cap: int = SUMS_CAP):
    """(Summe i, Summe u, abgeklungen) nach einem Stoß e_0 = 1, summiert bis der Betrag unter 1e-15 liegt oder `cap` erreicht ist.
    Nur im stabilen Fall sinnvoll. Erwartung (eigene Herleitung): Summe u = 1, Summe i = -1/alpha, unabhängig von L."""
    rho = spectral_radius(L, alpha)
    if rho >= 1.0 - RHO_TOL:
        return None, None, False
    T = 20 * L + 100 if rho < 1e-12 else min(cap, int(1.8 * math.log(1e-15) / math.log(rho)) + 20 * L + 100)
    i = pulse(L, alpha, T)
    done = bool(np.abs(i[-max(1, 4 * L):]).max() < 1e-12)
    s_i = float(i.sum())
    return s_i, float(-alpha * s_i), done


def shift_response(L: int, alpha: float, e: float, T: int) -> np.ndarray:
    """Dauerhaft verschobener Bedarf e ab Periode 0: Bestand i_0 ... i_T. Im stabilen Fall sitzt er bei -e/alpha (stationär: u = e)."""
    return simulate(L, alpha, [e] * T)


# --- Stetige Grenze --------------------------------------------------------------------------------------------------------------------

def continuous_table(Ls) -> dict:
    """alpha_krit * L, alpha_krit * (L - 1/2) und alpha_krit * (L - 1) gegen pi/2 (stetige Gleichung x' = -a x(t - tau): a tau = pi/2), je L.
    Relative Abstände: Wert / (pi/2) - 1."""
    L = np.asarray(Ls, dtype=float)
    ak = 2.0 * np.sin(np.pi / (4.0 * L - 2.0))
    half = np.pi / 2.0
    out = {"L": L.astype(int), "alpha_krit": ak}
    for name, factor in (("L", L), ("L_half", L - 0.5), ("L_minus1", L - 1.0)):
        out[name] = ak * factor
        out["rel_" + name] = ak * factor / half - 1.0
    return out


def first_within(Ls, rel, tol: float) -> int | None:
    """Kleinstes L, ab dem |rel| für alle größeren (in der Liste) höchstens tol ist; None, wenn es keines gibt."""
    ok = np.abs(np.asarray(rel)) <= tol
    if ok.all():
        return int(np.asarray(Ls)[0])
    last_bad = int(np.nonzero(~ok)[0].max())
    return int(np.asarray(Ls)[last_bad + 1]) if last_bad + 1 < len(ok) else None
