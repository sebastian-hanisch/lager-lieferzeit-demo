"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die für das lineare Lager gelten müssen, ohne dass man das Ergebnis kennt.

Das Modell ist linear und zeitinvariant (β = 0, s = 0, ohne Klemmung): Skalierung, Überlagerung, Verschiebung und Spiegelung des Bedarfs wirken genau so auf den
Bestand. Die Instanzen kommen aus dem SplitMix64 der Demo (feste Zufalls-Seeds). Voraussetzung jeder Aussage: dieses Modell, Zeitlage wie in llz_dynamics.py."""

import numpy as np
import pytest

import llz_dynamics as K
from llz_rng import uniform_array


def _cases(n, seed):
    """n Paare (L, α) mit L = 1 bis 12 und α = Anteil in (0,05, 0,95) von α_krit (keine Zweierpotenzen: nicht auf einem Gitter, das Dualbrüche trifft)."""
    u = uniform_array(seed, 2 * n)
    L = np.floor(u[:n] * 12).astype(int) + 1
    s = 0.05 + 0.9 * u[n:]
    return [(int(k), float(x * K.alpha_krit(int(k)))) for k, x in zip(L, s)]


def test_scaling_the_demand_scales_the_stock():
    for L, a in _cases(30, 11):
        e = np.zeros(200)
        e[0] = 1.0
        base = K.simulate(L, a, e)
        assert K.simulate(L, a, 3.7 * e) == pytest.approx(3.7 * base, abs=1e-9)


def test_superposition_of_two_shocks():
    for L, a in _cases(30, 12):
        e1 = np.zeros(150)
        e1[0] = 1.0
        e2 = np.zeros(150)
        e2[7] = -0.6
        both = K.simulate(L, a, e1 + e2)
        assert both == pytest.approx(K.simulate(L, a, e1) + K.simulate(L, a, e2), abs=1e-9)


def test_time_invariance_a_later_shock_shifts_the_response():
    for L, a in _cases(30, 13):
        k = 9
        e = np.zeros(150)
        e[k] = 1.0
        late = K.simulate(L, a, e)
        early = K.pulse(L, a, 150 - k)
        assert late[: k + 1] == pytest.approx(np.zeros(k + 1), abs=1e-15)
        assert late[k:] == pytest.approx(early, abs=1e-9)


def test_mirroring_the_demand_mirrors_the_stock():
    for L, a in _cases(30, 14):
        assert K.pulse(L, a, 120, e0=-1.0) == pytest.approx(-K.pulse(L, a, 120), abs=1e-12)


def test_summation_rules_hold_for_every_stable_pair_independent_of_L():
    for L, a in _cases(40, 15):
        s_i, s_u, done = K.pulse_sums(L, a)
        assert done and a * s_i == pytest.approx(-1.0, abs=1e-8) and s_u == pytest.approx(1.0, abs=1e-8), (L, a)


def test_constant_shift_settles_at_minus_e_over_alpha_independent_of_L():
    for L, a in _cases(15, 16):
        e = 0.35
        T = K.analyse_pulse(L, a, 0.01).horizon + 200
        assert K.shift_response(L, a, e, T)[-1] == pytest.approx(-e / a, rel=1e-6), (L, a)


def test_alpha_to_zero_leaves_the_shock_in_the_stock():
    for L in (1, 2, 3, 6, 12):
        i = K.pulse(L, 1e-9, 60)
        assert i[1:] == pytest.approx(-np.ones(60), abs=1e-6)


def test_alpha_krit_falls_strictly_with_L_but_alpha_krit_times_L_only_weakly():
    """α_krit fällt streng mit L (1 bis 60). α_krit · L fällt nur nicht streng: α_krit(1) · 1 = α_krit(2) · 2 = 2 exakt (Rundung ≤ 1e-15)."""
    v = [K.alpha_krit(L) for L in range(1, 61)]
    assert all(b < a for a, b in zip(v, v[1:]))
    w = [K.alpha_krit(L) * L for L in range(1, 61)]
    assert abs(w[0] - w[1]) < 1e-15
    assert all(b < a for a, b in zip(w[1:], w[2:]))


def test_spectral_radius_falls_left_and_rises_right_of_alpha_star():
    """Streng, soweit die Eigenwerte reichen: Verletzungen unter 1e-7 zählen nicht (Jordan-Block, gemessen bis 1,6e-8). L = 2 bis 12, je 300 Punkte."""
    for L in range(2, 13):
        ast, ak = K.alpha_star(L), K.alpha_krit(L)
        left = K.rho_grid(L, np.linspace(0.0, ast, 302)[1:-1])
        right = K.rho_grid(L, np.linspace(ast, ak, 302)[1:-1])
        assert np.diff(left).max() < 1e-7, L
        assert np.diff(right).min() > -1e-7, L


def test_same_inputs_give_the_same_output():
    a, b = K.analyse_pulse(3, 0.6, 0.01), K.analyse_pulse(3, 0.6, 0.01)
    assert (a.i == b.i).all() and a.decay == b.decay and a.rho == b.rho
