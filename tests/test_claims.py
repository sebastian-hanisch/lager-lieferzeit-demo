"""Claims-Tests: jede Zahl der Befunde B1 bis B11 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: dieses Modell (β = 0, s = 0, ohne Klemmung, Zeitlage wie in llz_dynamics.py), Bedarfsstoß e₀ = 1, Anfangszustand 0. Zahlen, die auf anderen
Plattformen anders ausfallen können (Eigenwerte, lange Läufe), stehen als Band; ganze Zählungen und Abklingzeiten, die an keiner Schwelle knapp liegen, exakt.
Die vorgerechneten Zählungen (`data/llz_frozen.json`) werden hier auf Konsistenz mit ihrer eigenen Erzeugung und mit der README geprüft; `tools/make_frozen.py` läuft dazu
in voller Größe (einige Sekunden)."""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import llz_dynamics as K
import llz_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import make_frozen  # noqa: E402

FROZEN = json.loads((ROOT / "data" / "llz_frozen.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
NB = " "


def _readme_has(*numbers):
    flat = README.replace(NB, " ").replace("\\*", "*")
    for n in numbers:
        assert n in flat, n


# --- B1 --------------------------------------------------------------------------------------------------------------------------------

def test_b1_formula_residual_and_period():
    dev = max(abs(K.alpha_krit_from_eigenvalues(L) - K.alpha_krit(L)) for L in range(1, 41))
    assert dev < 1e-9                                              # README: höchstens 1e-9 (gemessen 5e-15)
    assert ev.residual_check() < 1e-13                             # README: höchstens 1e-13 (gemessen 3,6e-16)
    chk = ev.period_check()
    assert chk["n"] == 39 and chk["deviating"] == 0 and chk["max_rel_dev"] < 1e-3
    _readme_has("höchstens 10⁻⁹", "höchstens 10⁻¹³", "0 von 39 Fällen (L = 2 bis 40)")


# --- B2 --------------------------------------------------------------------------------------------------------------------------------

def test_b2_frozen_counts_are_reproduced_by_the_script_and_stand_in_the_readme():
    live_grid = make_frozen.stability_grid()
    assert live_grid == FROZEN["stability_grid"]
    assert (live_grid["below"], live_grid["above"], live_grid["exceptions_below"], live_grid["exceptions_above"]) == (11559, 84441, 0, 0)
    assert live_grid["below"] + live_grid["above"] == 12 * 8000
    live_pairs = make_frozen.random_pairs()
    assert live_pairs == FROZEN["random_pairs"]
    assert (live_pairs["below"], live_pairs["above"], live_pairs["exceptions"], live_pairs["seed"]) == (696, 19304, 0, 15)
    _readme_has("11 559 Punkte unterhalb, 84 441 oberhalb", "696 unterhalb, 19 304 oberhalb", "Zufalls-Seed 15", "0 Ausnahmen")


# --- B3 --------------------------------------------------------------------------------------------------------------------------------

def test_b3_continuous_limit():
    rows = {r["L"]: r for r in ev.continuous_rows((1, 2, 3, 6, 12, 100))}
    assert [round(rows[L]["aL"], 3) for L in (1, 2, 3, 6, 12)] == [2.0, 2.0, 1.854, 1.708, 1.638]
    assert [round(100 * rows[L]["rel_L"], 1) for L in (1, 2, 3)] == [27.3, 27.3, 18.0]
    assert round(100 * rows[12]["rel_L"], 2) == 4.27 and round(100 * rows[100]["rel_L"], 3) == 0.501
    assert rows[100]["inv"] == pytest.approx(0.00503, abs=5e-6) and rows[100]["rel_L"] == pytest.approx(rows[100]["inv"], rel=5e-3)
    s = ev.continuous_summary()
    assert (s["within_1"], s["within_01"], s["half_within_1"], s["half_within_01"], s["L_max"]) == (51, 501, 4, 11, 200000)
    assert s["slope_L"] == pytest.approx(-1.00, abs=0.01) and s["slope_half"] == pytest.approx(-2.00, abs=0.01)
    assert s["above_pi_half"] and s["half_below_pi_half"]
    _readme_has("ab L = 51", "ab L = 4", "ab L = 501 bzw. 11", "+27,3 %", "+18,0 %", "+4,27 %", "+0,501 %", "−1,00", "bis L = 200 000")


# --- B4 --------------------------------------------------------------------------------------------------------------------------------

def test_b4_euler_number_from_the_eigenvalues():
    for m, expected in ((1, 1.000), (2, 1.236), (5, 1.423), (10, 1.495), (100, 1.563)):
        L = m + 1
        euler = 2 * m * math.sin(math.pi / (4 * m + 2))
        from_eigenvalues = K.alpha_krit_from_eigenvalues(L) * m          # nicht aus der Formel: Bisektion auf ρ = 1
        assert from_eigenvalues == pytest.approx(euler, abs=1e-8)
        assert round(euler, 3) == expected
    _readme_has("1,000 (L = 2), 1,236 (3), 1,423 (6), 1,495 (11), 1,563 (101)")


# --- B5 --------------------------------------------------------------------------------------------------------------------------------

def test_b5_double_root_and_the_monotone_grid_is_stored_for_every_L():
    assert [round(K.alpha_star(L) / K.alpha_krit(L), 4) for L in (2, 3, 6, 12)] == [0.2500, 0.2397, 0.2353, 0.2345]
    assert round(K.alpha_star(1000) / K.alpha_krit(1000), 4) == round(2 / (math.e * math.pi), 4) == 0.2342
    mono = FROZEN["rho_monotone"]
    assert mono["n_each_side"] == 2000 and sorted(map(int, mono["L"])) == list(range(2, 13))
    assert all(v["left_largest_step"] < 0 < v["right_smallest_step"] for v in mono["L"].values())
    live = make_frozen.rho_monotone()
    assert live["L"].keys() == mono["L"].keys()
    assert all(live["L"][k]["left_largest_step"] < 0 < live["L"][k]["right_smallest_step"] for k in live["L"])
    assert [str(K.alpha_star_exact(L)) for L in (2, 3, 4)] == ["1/4", "4/27", "27/256"]
    _readme_has("0,2500 (L = 2), 0,2397 (3), 0,2353 (6), 0,2345 (12)", "2/(eπ) = 0,2342", "2 × 2 000 Punkte")


# --- B6 --------------------------------------------------------------------------------------------------------------------------------

def test_b6_estimate_against_measurement_in_35_cells():
    c = ev.decay_cells()
    assert c["n"] == 35 and c["within_10"] == 29 and c["n_outside"] == 6
    assert c["ratio_min"] == pytest.approx(0.954, abs=1e-3) and c["ratio_max"] == pytest.approx(1.463, abs=1e-3) and c["ratio_mean"] == pytest.approx(1.055, abs=1e-3)
    assert c["outside_ratio_min"] == pytest.approx(1.116, abs=1e-3)
    assert set(c["outside"]) == {(1, 0.26), (1, 0.52), (2, 0.26), (3, 0.26), (6, 0.26), (12, 0.26)} and c["outside_near_star"] == 5
    _readme_has("29 von 35", "0,954 bis 1,463, Mittel 1,055", "12 % bis 46 %")


# --- B7 --------------------------------------------------------------------------------------------------------------------------------

def test_b7_fastest_return_is_not_at_alpha_star():
    expected = {2: (7, 11), 3: (11, 18), 4: (16, 26), 6: (25, 40), 12: (53, 84), 24: (109, 172)}
    ratios, lo, hi = [], [], []
    for L, (tmin, tstar) in expected.items():
        f = K.fastest_return(L, 0.01)
        assert (f["t_min"], f["t_star"]) == (tmin, tstar), L
        ratios.append(tmin / tstar)
        lo.append(f["alpha_lo"] / f["alpha_star"])
        hi.append(f["alpha_hi"] / f["alpha_star"])
    assert 0.61 <= min(ratios) and max(ratios) < 0.645
    assert min(lo) == pytest.approx(1.14, abs=5e-3) and max(hi) == pytest.approx(1.20, abs=5e-3)
    share = [K.fastest_return(L, 0.01)["alpha_lo"] / K.alpha_krit(L) for L in expected]
    assert 0.28 <= min(share) and max(share) <= 0.30
    _readme_has("7 gegen 11", "11 gegen 18", "16 gegen 26", "25 gegen 40", "53 gegen 84", "109 gegen 172", "0,61 bis 0,64", "1,14 bis 1,20 · α*", "0,28 bis 0,30 · α_krit")


def test_b7_the_time_jumps_with_alpha_and_depends_on_the_threshold():
    j = ev.jump_example()
    assert j[0.1769]["time"] == 11 and j[0.1772]["time"] == 15                              # max i 0,00991 gegen 0,01020: kreuzt die 1 %-Linie
    assert j[0.1769]["max_pos"] < 0.01 < j[0.1772]["max_pos"]
    by_thr = {L: [K.fastest_return(L, t)["t_min"] for t in (0.005, 0.01, 0.02, 0.05)] for L in (3, 12)}
    assert by_thr == {3: [13, 11, 10, 8], 12: [61, 53, 46, 37]}
    formula = {L: [math.log(t) / math.log((L - 1) / L) for t in (0.005, 0.01, 0.02, 0.05)] for L in (3, 12)}
    assert [round(x, 1) for x in formula[12]] == [60.9, 52.9, 45.0, 34.4] and [round(x, 1) for x in formula[3]] == [13.1, 11.4, 9.6, 7.4]
    _readme_has("L = 3, α = 0,1769: 11 Perioden, α = 0,1772: 15", "13 / 11 / 10 / 8", "61 / 53 / 46 / 37", "60,9 / 52,9 / 45,0 / 34,4", "13,1 / 11,4 / 9,6 / 7,4")


# --- B8 --------------------------------------------------------------------------------------------------------------------------------

def test_b8_near_the_boundary():
    expected = {1: (24, 44, 228), 3: (172, 317, 1582), 6: (379, 697, 3534), 12: (792, 1479, 7413)}
    for L, times in expected.items():
        got = tuple(K.analyse_pulse(L, s * K.alpha_krit(L), 0.01).decay for s in (0.91, 0.95, 0.99))
        assert got == times, L
    a = K.analyse_pulse(3, 0.6, 0.01)
    assert a.rho == pytest.approx(0.99145, abs=5e-6) and a.decay == 542 and a.estimate == pytest.approx(536.3, abs=0.05)
    assert a.peak == pytest.approx(1.04, abs=5e-3) and a.i[7] == pytest.approx(26 / 25, abs=1e-12) and 0.6 / K.alpha_krit(3) == pytest.approx(0.971, abs=5e-4)
    _readme_has("24 / 44 / 228", "172 / 317 / 1 582", "379 / 697 / 3 534", "792 / 1 479 / 7 413", "ρ = 0,99145, Abklingzeit 542, Schätzung 536,3", "26/25", "1,04")
    # Plan: Abklingzeit bei 0,95 α_krit zwischen 317 (L = 3) und 1 479 (L = 12), Beispiele der Einleitung
    _readme_has("für L = 3 317 Perioden, für L = 12 1 479")


# --- B9 --------------------------------------------------------------------------------------------------------------------------------

def _sign_changes(L, factor):
    i = K.pulse(L, factor * K.alpha_star(L), 6000)
    idx = np.nonzero(np.abs(i) > 1e-9)[0]
    s = np.sign(i[idx[idx > 0]])
    return int((np.diff(s) != 0).sum())


def test_b9_overshoot_amplification_and_the_stored_counts():
    assert [_sign_changes(L, 0.9) for L in range(1, 13)] == [0] * 12
    assert [_sign_changes(L, 1.1) for L in range(2, 13)] == [3] * 11 and _sign_changes(1, 1.1) == 9
    below = FROZEN["below_star"]
    assert (below["n_pairs"], below["positive"], below["seed"], below["L_min"], below["L_max"]) == (5000, 0, 16, 2, 40)
    assert make_frozen.below_star() == below
    amp = FROZEN["amplification"]
    assert amp["n_grid"] == 20000
    over = [v["first_overshoot_share"] for v in amp["per_L"].values()]
    ampl = [v["first_amplification_share"] for k, v in amp["per_L"].items() if int(k) >= 3]
    assert min(over) == pytest.approx(0.236, abs=5e-4) and max(over) == pytest.approx(0.252, abs=5e-4)
    assert min(ampl) == pytest.approx(0.948, abs=5e-4) and max(ampl) == pytest.approx(0.960, abs=5e-4) and amp["per_L"]["2"]["first_amplification_share"] is None
    p99 = [K.analyse_pulse(L, 0.99 * K.alpha_krit(L), 0.01).peak for L in range(3, 13)]
    p999 = [K.analyse_pulse(L, 0.999 * K.alpha_krit(L), 0.01).peak for L in range(3, 13)]
    assert (round(min(p99), 3), round(max(p99), 3)) == (1.052, 1.073) and (round(min(p999), 3), round(max(p999), 3)) == (1.069, 1.095)
    assert K.analyse_pulse(1, 1.999, 0.01).peak == 1.0 and K.analyse_pulse(2, 0.9999 * K.alpha_krit(2), 0.01).peak == pytest.approx(1.0, abs=1e-9)
    _readme_has("L = 1 bis 12", "dreimal (L = 2 bis 12", "neunmal", "0,236 bis 0,252 · α_krit", "0,948 bis 0,960 · α_krit", "1,052 bis 1,073", "1,069 bis 1,095", "Zufalls-Seed 16")


# --- B10 -------------------------------------------------------------------------------------------------------------------------------

def test_b10_summation_rules_and_the_area_at_the_boundary():
    chk = ev.sums_check()
    assert chk["n"] == 36 and chk["dev_u"] < 1e-9 and chk["dev_i"] < 1e-9
    assert [round(1 / K.alpha_krit(L), 3) for L in (1, 2, 3, 6, 12)] == [0.5, 1.0, 1.618, 3.513, 7.327]
    assert 1 / K.alpha_krit(3) == pytest.approx((1 + math.sqrt(5)) / 2, abs=1e-14)
    for L in (1, 4, 9):
        assert K.shift_response(L, 0.7 * K.alpha_krit(L), 0.2, K.analyse_pulse(L, 0.7 * K.alpha_krit(L), 0.01).horizon + 100)[-1] == pytest.approx(-0.2 / (0.7 * K.alpha_krit(L)), rel=1e-6)
    _readme_has("36 Fällen", "0,500 (L = 1), 1,000 (2), 1,618 (3, Goldener Schnitt), 3,513 (6), 7,327 (12)")


# --- B11 und Einleitung ----------------------------------------------------------------------------------------------------------------

def test_b11_two_independent_ways_agree_in_60_cases():
    scipy_signal = pytest.importorskip("scipy.signal")
    worst = 0.0
    n = 0
    for L in range(1, 13):
        for s in (0.13, 0.31, 0.47, 0.73, 0.93):
            a = s * K.alpha_krit(L)
            den = np.zeros(L + 1)
            den[0], den[1] = 1.0, -1.0
            den[L] += a
            e = np.zeros(301)
            e[0] = 1.0
            worst = max(worst, float(np.abs(scipy_signal.lfilter(np.r_[0.0, -1.0], den, e) - K.pulse(L, a, 300)).max()))
            n += 1
    assert n == 60 and worst < 1e-12
    _readme_has("60 Fälle (L = 1 bis 12, 5 α, 300 Perioden)")


def test_intro_numbers_and_the_hand_values():
    assert [round(K.alpha_krit(L), 3) for L in (1, 2, 3, 6, 12)] == [2.0, 1.0, 0.618, 0.285, 0.136]
    assert [K.period_krit(L) for L in (1, 2, 3, 6, 12)] == [2, 6, 10, 22, 46]
    assert K.analyse_pulse(3, 0.7, 0.01).rho == pytest.approx(1.03686, abs=5e-6)
    _readme_has("2 (L = 1), 1 (L = 2), 0,618 (L = 3), 0,285 (L = 6) und 0,136 (L = 12)", "ab L = 3 rund 0,23 bis 0,24 · α_krit")
    assert 0.2342 <= K.alpha_star(12) / K.alpha_krit(12) <= 0.2346


def test_results_table_rows_match_the_plan_table():
    rows = {r["L"]: r for r in ev.results_table()}
    assert [(r["t_star"], r["t_min"], r["t_near"]) for r in rows.values()] == [(1, 1, 44), (11, 7, 184), (18, 11, 317), (40, 25, 697), (84, 53, 1479)]
    assert [round(r["rho_star"], 3) for r in rows.values()] == [0.0, 0.5, 0.667, 0.833, 0.917]


def test_readme_names_the_presets_numbers_only_where_the_tests_have_them():
    """Die Zahlen der Preset-Hilfetexte stehen in test_verdicts.py; hier nur: die Befunde der README und die App nennen dieselbe Lage."""
    assert ev.analyse(ev.Settings(3, 0.175, 1.0)).decay == 12
    _readme_has("α = 0,175 (12 Perioden, Überschwinger 0,0082")
    assert ev.analyse(ev.Settings(3, 0.175, 1.0)).max_pos == pytest.approx(0.0082, abs=5e-5)
    assert re.search(r"Stück 1 der Linie", README)


def test_euler_scheme_is_the_lager_with_m_equal_L_minus_1():
    """README: das Lager ist exakt das Euler-Schema x_(n+1) = x_n − a·h·x_(n−m) für x′ = −a·x(t − τ) mit m = L − 1 Schritten je Verzögerung und α = a·h (hier mit
    dem Stoß als Anfangsstörung x_1 = −1 und Vorgeschichte 0); eigener Euler-Lauf, nicht der Kern."""
    for L, alpha in ((1, 0.37), (2, 0.61), (3, 0.43), (7, 0.11)):
        m = L - 1
        x = [0.0, -1.0] + [0.0] * 60
        for n in range(1, 61):
            hist = x[n - m] if n - m >= 0 else 0.0
            x[n + 1] = x[n] - alpha * hist
        assert np.array(x[:62]) == pytest.approx(K.pulse(L, alpha, 61), abs=1e-12), L
