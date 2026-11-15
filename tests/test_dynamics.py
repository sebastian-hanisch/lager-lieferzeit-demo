"""Kern gegen Handrechnung (Mini-Instanzen von Hand, Brüche) und Randfälle (DEMO-PLAYBOOK §1b).

Jede Handrechnung steht als Zahl im Test, nicht als Ausgabe des Codes. Die benannten Zweierpotenz-Fälle (α_krit(1) = 2, α_krit(2) = 1, α*(1) = 1,
α*(2) = 1/4) sind exakt in Brüchen gerechnet und zusätzlich bei α (1 ± 1e-3) geprüft; sonst kommt keine Zweierpotenz als α vor."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import llz_dynamics as K
from conftest import exact_pulse


def test_l1_closed_form_and_sign_change():
    """L = 1: i_{t+1} = (1 − α) i_t, also i_t = −(1 − α)^(t−1). α = 1/5: monoton; α = 6/5: Vorzeichen wechselt je Periode, klingt ab; α = 9/5: langsam mit Wechsel."""
    for alpha in (F(1, 5), F(6, 5), F(9, 5)):
        i = exact_pulse(1, alpha, 40)
        assert all(i[t] == -((1 - alpha) ** (t - 1)) for t in range(1, 41))
    assert exact_pulse(1, F(6, 5), 4)[1:] == [F(-1), F(1, 5), F(-1, 25), F(1, 125)]
    assert exact_pulse(1, F(9, 5), 3)[1:] == [F(-1), F(4, 5), F(-16, 25)]
    assert [float(x) for x in K.pulse(1, 1.2, 4)[1:]] == pytest.approx([-1.0, 0.2, -0.04, 0.008])


def test_l1_totbeat_is_a_named_power_of_two_case():
    """α = 1 (Zweierpotenz, benannter Ausnahmefall): der Stoß ist nach einer Periode weg, exakt in Brüchen; bei α (1 ± 1e-3) bleibt höchstens 1e-3."""
    assert exact_pulse(1, F(1), 5)[2] == 0
    for a in (1 - 1e-3, 1 + 1e-3):
        assert abs(K.pulse(1, a, 5)[2]) == pytest.approx(1e-3)
    assert K.alpha_star_exact(1) == 1 and K.alpha_star(1) == 1.0


def test_l2_period_six_at_the_boundary():
    """L = 2, α = 1 (Grenze, Zweierpotenz): i_1 … i_12 = −1, −1, 0, 1, 1, 0, … exakt, Periode 6 = 4L − 2; Wurzeln e^(±iπ/3)."""
    i = exact_pulse(2, F(1), 12)
    assert i[1:] == [-1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0]
    z = np.sort_complex(K.roots_p(2, 1.0))
    assert z == pytest.approx(np.sort_complex(np.array([np.exp(-1j * np.pi / 3), np.exp(1j * np.pi / 3)])), abs=1e-12)
    assert K.period_krit(2) == 6 and K.alpha_krit(2) == pytest.approx(1.0, abs=1e-15)


def test_l2_double_root_decay_time_eleven():
    """L = 2, α* = 1/4: i_t = −t · 2^(1−t) exakt; i_11 = −11/1024 > 0,01, i_12 = −3/512: Abklingzeit 11 (Schwelle 1 %)."""
    i = exact_pulse(2, F(1, 4), 40)
    assert all(i[t] == -F(t) * F(2) ** (1 - t) for t in range(1, 41))
    assert i[11] == F(-11, 1024) and i[12] == F(-3, 512)
    assert K.decay_time([float(x) for x in i], 0.01) == 11
    assert K.alpha_star_exact(2) == F(1, 4)


def test_l3_hand_example(hand):
    """L = 3, α = 3/5: i_1 … i_7 von Hand (Fixture); i_7 = 26/25 > 1, obwohl stabil; Abklingzeit 542, ρ = 0,99145."""
    i = exact_pulse(3, F(3, 5), 7)
    assert i[1:8] == hand
    f = K.pulse(3, 0.6, 7)
    assert list(f[1:8]) == pytest.approx([float(x) for x in hand], abs=1e-12)
    a = K.analyse_pulse(3, 0.6, 0.01)
    assert a.regime == "stable" and a.rho == pytest.approx(0.99145, abs=5e-6)
    assert a.i[7] == pytest.approx(26 / 25, abs=1e-12) and a.peak > 1.0 and a.peak == pytest.approx(1.04, abs=5e-3)
    assert 538 <= a.decay <= 546                      # Messwert 542; die Zählung hängt an der Schwelle, Band statt Gleichheit
    assert a.estimate == pytest.approx(math.log(0.01) / math.log(a.rho))


def test_l3_boundary_is_the_golden_ratio_case():
    """α_krit(3) erfüllt α² + α − 1 = 0 (exakt in ℚ(√5)); 1/α_krit = (1 + √5)/2 (Goldener Schnitt); Wurzeln e^(±iπ/5), Periode 10."""
    a = K.alpha_krit(3)
    assert a * a + a - 1 == pytest.approx(0.0, abs=1e-15)
    assert 1 / a == pytest.approx((1 + math.sqrt(5)) / 2, abs=1e-14)
    z = np.exp(1j * np.pi / 5)
    assert abs(z ** 3 - z ** 2 + a) < 1e-15
    r = K.roots_p(3, a * (1 + 1e-6))
    assert 2 * math.pi / abs(np.angle(r[int(np.argmax(abs(r)))])) == pytest.approx(10.0, rel=1e-3)
    # exakt in Q(√5) als Paar (Rationalteil, Koeffizient von √5): α = (−1 + √5)/2, α² = 3/2 − √5/2, also α² + α − 1 = (3/2 − 1/2 − 1) + (−1/2 + 1/2) √5 = 0
    s2 = (F(1, 4) + F(5, 4), F(-1, 2))                # α² = 3/2 − √5/2
    assert (s2[0] + F(-1, 2) - 1, s2[1] + F(1, 2)) == (0, 0)


def test_l3_double_root_closed_form():
    """L = 3, α* = 4/27: P = (z − 2/3)² (z + 1/3); i_t = (−1/3 − t)(2/3)^t + (1/3)(−1/3)^t exakt für t = 0 bis 40; Abklingzeit 18."""
    assert K.alpha_star_exact(3) == F(4, 27)
    i = exact_pulse(3, F(4, 27), 40)
    assert all(i[t] == (F(-1, 3) - t) * F(2, 3) ** t + F(1, 3) * F(-1, 3) ** t for t in range(41))
    assert K.decay_time([float(x) for x in i], 0.01) == 18


def test_double_root_conditions_hold_exactly_for_l_2_to_12():
    """P(z*) = P'(z*) = 0 in Brüchen mit z* = (L − 1)/L."""
    for L in range(2, 13):
        z = F(L - 1, L)
        alpha = K.alpha_star_exact(L)
        assert z ** L - z ** (L - 1) + alpha == 0
        assert L * z ** (L - 1) - (L - 1) * z ** (L - 2) == 0


def test_summation_rules_and_constant_shift_exact():
    """Σ i = −1/α und Σ u = 1 (Herleitung: Rekursion über t summieren); dauerhaft verschobener Bedarf e: Bestand −e/α (Brüche, L = 1, 2, 3)."""
    for L, alpha, e in ((1, F(3, 10), F(1, 5)), (3, F(3, 10), F(1, 5)), (2, F(3, 5), F(1, 10))):
        i = [F(0)] * 2001
        for t in range(2000):
            prev = i[t + 1 - L] if t + 1 - L >= 0 else F(0)
            i[t + 1] = i[t] - alpha * prev - e
        assert float(i[2000]) == pytest.approx(float(-e / alpha), abs=1e-9)
    for L in (1, 2, 3):
        s_i, s_u, done = K.pulse_sums(L, 0.3)
        assert done and s_i == pytest.approx(-1 / 0.3, abs=1e-9) and s_u == pytest.approx(1.0, abs=1e-9)
    assert K.shift_response(3, 0.3, 0.2, 3000)[-1] == pytest.approx(-0.2 / 0.3, abs=1e-9)


def test_continuous_table_of_hand_values():
    """α_krit · L = 2, 2, 1,854 für L = 1, 2, 3 gegen π/2 = 1,571 (+27,3 %, +27,3 %, +18,0 %); α_krit · (L − ½) = 1,000 / 1,500 / 1,545; α_krit · (L − 1) = 0 / 1,000 / 1,236."""
    t = K.continuous_table([1, 2, 3])
    assert t["L"] == pytest.approx([2.0, 2.0, 1.854102], abs=1e-6)
    assert t["rel_L"] == pytest.approx([0.2732, 0.2732, 0.1804], abs=5e-5)
    assert t["L_half"] == pytest.approx([1.0, 1.5, 1.545085], abs=1e-6)
    assert t["L_minus1"] == pytest.approx([0.0, 1.0, 1.236068], abs=1e-6)


def test_edge_cases_horizon_and_blow_up():
    """Ränder: L = 1 und L = 40, α am Rand der Regler, Instabilität bricht ab, Horizont begrenzt (nie eine erfundene Abklingzeit)."""
    a = K.analyse_pulse(40, 0.001, 0.01)
    assert a.regime == "stable" and a.decay is not None and a.rho < 1
    b = K.analyse_pulse(3, 2.5, 0.01)
    assert b.regime == "unstable" and b.decay is None and b.estimate is None
    capped = K.analyse_pulse(3, 0.001, 0.01, cap=1000)           # Abklingzeit ~4600 liegt über dem Horizont von 1000
    assert capped.decay is None and capped.regime == "stable"
    assert K.classify(1.0) == "boundary" and K.classify(1.0 + 1e-6) == "unstable" and K.classify(1.0 - 1e-6) == "stable"
    assert len(K.pulse(3, 2.5, 100000)) < 100000 + 1          # bricht bei |i| > 1e12 ab
    assert K.decay_estimate(1.0, 0.01) is None and K.decay_estimate(0.0, 0.01) is None


def test_decay_time_is_the_last_period_above_the_threshold():
    assert K.decay_time([0, -1.0, -0.5, -0.011, 0.009, 0.0], 0.01) == 3          # 0,011 > 0,01 · 1; 0,009 nicht mehr
    assert K.decay_time([0, -1.0, -0.01, 0.0], 0.01) == 1                         # 0,01 > 0,01 gilt nicht (strikt größer)
    assert K.decay_time([0, -1.0, 0.0], 0.01) == 1
