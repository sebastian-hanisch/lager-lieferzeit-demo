"""Orakel-Tests lager-lieferzeit-demo (ORACLE-PLAYBOOK §5, §6): jede Kernroutine gegen einen anderen Rechenweg.

Kern (llz_dynamics.py)                       Orakel (anderer Rechenweg)
alpha_krit(L) = 2 sin(π/(4L − 2))            Eigenwerte der Zustandsmatrix (Bisektion auf ρ = 1) und `numpy.roots` von P; Residuum von P auf dem Kreis
pulse (Rekursion)                            `numpy.linalg.matrix_power` der Zustandsmatrix und `scipy.signal.lfilter` mit der Übertragungsfunktion
decay_time / analyse_pulse                   geschlossene Formeln (L = 1: |1 − α|^(t−1); L = 2, α = 1/4: Brüche)
alpha_star                                   Brüche (`fractions`): P(z*) = P'(z*) = 0; Minimum von ρ(α) gegen ein Gitter
Stabilitätsgebiet (0, α_krit)                Gitter L ≤ 12 und Zufallspaare L ≤ 60 (SplitMix64, andere Zufalls-Seeds als tools/make_frozen.py)
Summenregeln                                 Brüche (Dauerverschiebung) und Summation der Rekursion
Gitter-Abklingzeit (pulse_grid)              einzelne Rekursion je alpha

Eigenwerte am Doppelwurzelpunkt sind auf etwa 1e-8 genau (Jordan-Block): dort nur Bänder. Zahlen, die auf anderen Plattformen anders ausfallen können
(BLAS, lange Läufe), sind als Band oder mit Toleranz geprüft, nie bitgleich."""

import numpy as np
import pytest
from fractions import Fraction as F

import llz_dynamics as K
from llz_rng import uniform_array

scipy_signal = pytest.importorskip("scipy.signal")


def test_alpha_krit_matches_eigenvalues_and_polynomial_roots():
    for L in range(1, 41):
        ak = K.alpha_krit(L)
        assert K.alpha_krit_from_eigenvalues(L) == pytest.approx(ak, abs=1e-9)
        below = np.roots(K.char_poly(L, ak * (1 - 1e-6)))
        above = np.roots(K.char_poly(L, ak * (1 + 1e-6)))
        assert np.abs(below).max() < 1.0 < np.abs(above).max(), L
        z = np.exp(1j * np.pi / (2 * L - 1))
        assert abs(z ** L - z ** (L - 1) + ak) < 1e-13, L          # Residuum von P auf dem Kreis (Messung 3,6e-16)


def test_critical_period_from_the_root_angle():
    """Periode 2π/θ aus dem Winkel der größten Wurzel hinter der Grenze weicht von 4L − 2 um höchstens 0,1 % ab (L = 2 bis 40)."""
    for L in range(2, 41):
        z = np.roots(K.char_poly(L, K.alpha_krit(L) * (1 + 1e-6)))
        w = z[np.argmax(np.abs(z))]
        assert 2 * np.pi / abs(np.angle(w)) == pytest.approx(4 * L - 2, rel=1e-3), L


def test_state_matrix_and_polynomial_have_the_same_spectrum():
    for L in (1, 2, 3, 7, 12):
        for alpha in (0.07, 0.23, 0.61):
            a = np.sort_complex(np.linalg.eigvals(K.state_matrix(L, alpha)))
            b = np.sort_complex(np.roots(K.char_poly(L, alpha)))
            assert a == pytest.approx(b, abs=1e-6)


def test_pulse_matches_matrix_power_and_lfilter():
    """60 Fälle (L = 1 bis 12, 5 α, 300 Perioden): Rekursion gegen Zustandsmatrixpotenz und gegen lfilter mit i = −z⁻¹ E / (1 − z⁻¹ + α z⁻ᴸ)."""
    T = 300
    for L in range(1, 13):
        for share in (0.13, 0.31, 0.47, 0.73, 0.93):
            alpha = share * K.alpha_krit(L)
            i = K.pulse(L, alpha, T)
            M = K.state_matrix(L, alpha)
            x1 = np.zeros(L)
            x1[0] = -1.0                              # Zustand zur Periode 1: (i_1, i_0, ..., ) = (−1, 0, ...)
            via_matrix = [np.linalg.matrix_power(M, t - 1) @ x1 for t in range(1, T + 1)]
            assert np.array([v[0] for v in via_matrix]) == pytest.approx(i[1:], abs=1e-12)
            den = np.zeros(L + 1)
            den[0], den[1] = 1.0, -1.0
            den[L] += alpha
            e = np.zeros(T + 1)
            e[0] = 1.0
            via_filter = scipy_signal.lfilter(np.r_[0.0, -1.0], den, e)    # i = −z⁻¹ E / (1 − z⁻¹ + α z⁻ᴸ); die Rekursion hat i_t = 0 für t = 0
            assert via_filter[: T + 1] == pytest.approx(i, abs=1e-12)


def test_decay_time_against_closed_forms():
    """L = 1: |i_t| = |1 − α|^(t−1), die Spitze ist 1 (t = 1): Abklingzeit = größtes t mit |1 − α|^(t−1) > Schwelle. L = 2, α = 1/4: 11 (Brüche)."""
    for alpha in (0.37, 0.83, 1.13, 1.61):
        for thr in (0.005, 0.01, 0.05):
            q = abs(1 - alpha)
            expected = max(t for t in range(1, 400) if q ** (t - 1) > thr)
            assert K.analyse_pulse(1, alpha, thr).decay == expected, (alpha, thr)
    assert K.analyse_pulse(2, 0.25, 0.01).decay == 11


def test_double_root_minimises_the_spectral_radius():
    """ρ(α) ist bei α* am kleinsten, Wert (L − 1)/L: Lage auf zwei Gitterschritte, Wert auf 1e-3 (L = 2 bis 12, 3000 Punkte)."""
    for L in range(2, 13):
        ak, ast = K.alpha_krit(L), K.alpha_star(L)
        grid = np.linspace(0.0, ak, 3002)[1:-1]
        rho = K.rho_grid(L, grid)
        k = int(np.argmin(rho))
        step = grid[1] - grid[0]
        assert abs(grid[k] - ast) <= 2 * step, L
        assert rho[k] == pytest.approx((L - 1) / L, abs=1e-3), L
        assert K.spectral_radius(L, ast) == pytest.approx((L - 1) / L, abs=1e-6), L      # Band: Jordan-Block, gemessen bis 1,6e-8


def test_stability_region_is_exactly_below_alpha_krit_grid_and_random():
    """Gitter L = 1 bis 12 (800 α in (0, 4)) und 500 Zufallspaare (L bis 60, Zufalls-Seed 2026): keine Ausnahme von „ρ < 1 genau unterhalb“."""
    grid = np.linspace(0.0, 4.0, 802)[1:-1]
    for L in range(1, 13):
        rho = K.rho_grid(L, grid)
        below = grid < K.alpha_krit(L)
        assert (rho[below] < 1.0).all() and (rho[~below] > 1.0).all(), L
    n = 500
    u = uniform_array(2026, 2 * n)
    Ls = np.floor(u[:n] * 60).astype(int) + 1
    al = 4.0 * u[n:]
    for L, a in zip(Ls, al):
        ak = K.alpha_krit(int(L))
        if abs(a / ak - 1) < 1e-6:
            continue
        assert (K.spectral_radius(int(L), float(a)) < 1.0) == (a < ak), (L, a)


def test_no_overshoot_below_alpha_star_random_pairs():
    """500 Zufallspaare (L = 2 bis 40, α zufällig in (0, 0,99 α*), Zufalls-Seed 77): kein Bestand über Soll in 400 Perioden."""
    n = 500
    u = uniform_array(77, 2 * n)
    for L, f in zip(np.floor(u[:n] * 39).astype(int) + 2, u[n:]):
        i = K.pulse(int(L), float(f) * 0.99 * K.alpha_star(int(L)), 400)
        assert i.max() <= 1e-12, L


def test_grid_decay_times_equal_single_recursions():
    for L in (1, 2, 3, 6, 12):
        grid = np.linspace(0.02, 0.9, 7) * K.alpha_krit(L)
        got = K.decay_times_grid(L, grid, 0.01, 3000)
        for a, t in zip(grid, got):
            single = K.analyse_pulse(L, float(a), 0.01)
            expect = single.decay if (single.decay is not None and single.decay < 0.8 * 3000) else -1
            assert t == expect, (L, a)


def test_summation_rules_in_36_cases():
    """Σ u = 1 und Σ i = −1/α, L = 1 bis 12, α = 0,21 / 0,47 / 0,83 · α_krit (Messung 7,8e-16; hier Toleranz 1e-9)."""
    for L in range(1, 13):
        for s in (0.21, 0.47, 0.83):
            a = s * K.alpha_krit(L)
            s_i, s_u, done = K.pulse_sums(L, a)
            assert done and s_u == pytest.approx(1.0, abs=1e-9) and a * s_i == pytest.approx(-1.0, abs=1e-9), (L, s)


def test_boundary_cases_by_exact_fractions_and_neighbours():
    """Benannte exakte Fälle α_krit(1) = 2, α_krit(2) = 1 (Brüche) und bei α (1 ± 1e-3): darunter ρ < 1, darüber ρ > 1."""
    for L, exact in ((1, F(2)), (2, F(1))):
        assert K.alpha_krit(L) == pytest.approx(float(exact), abs=1e-14)
        assert K.spectral_radius(L, float(exact) * (1 - 1e-3)) < 1.0 < K.spectral_radius(L, float(exact) * (1 + 1e-3))


def test_roots_nearly_double_loses_digits_but_the_decay_time_does_not():
    """Am Doppelwurzelpunkt streut ρ um bis 1e-8 (Eigenwerte eines Jordan-Blocks); die Rekursion bleibt davon unberührt: Abklingzeit 18 bei L = 3."""
    assert K.analyse_pulse(3, K.alpha_star(3), 0.01).decay == 18
    for L in (4, 7, 12):
        assert abs(K.spectral_radius(L, K.alpha_star(L)) - (L - 1) / L) < 1e-6
