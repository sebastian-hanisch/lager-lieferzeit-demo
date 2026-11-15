"""Verdikte (DEMO-PLAYBOOK §12 und §11a Punkt 7): reine Funktionen (Art, Text). Je Preset die erwartete Zeile, je Zweig eine handgesetzte Analyse, jede Zahl im Text.

Ein Preset ohne Effekt meldet keinen Effekt: „Fast-Totbeat“ meldet keinen Überschwinger, „Doppelwurzel“ kein Vorzeichenwechsel, „Hinter der Grenze“ kein Abklingen."""

import numpy as np
import pytest

import llz_constants as C
import llz_dynamics as K
import llz_evaluation as ev


def _a(L=3, alpha=0.6, rho=0.99, regime="stable", peak=1.0, max_pos=0.0, decay=20, estimate=18.0, horizon=100, thr=0.01):
    ak, ast = K.alpha_krit(L), K.alpha_star(L)
    return K.PulseAnalysis(L, alpha, thr, ak, ast, alpha / ak, rho, regime, np.array([0.0, -1.0]), peak, max_pos, decay, estimate, horizon)


def _preset(name):
    p = C.PRESETS[name]
    return ev.analyse(ev.Settings(p["L"], p["alpha"], p["thr"]))


def test_presets_have_the_expected_verdict_line():
    kind, text = ev.verdict_pulse(_preset("Von Hand"))
    assert kind == "warning" and "max|i| = 1,04" in text and "klingt in 542 Perioden auf 1 % der Spitze ab" in text and "Schätzung aus ρ: 536,3" in text
    kind, text = ev.verdict_pulse(_preset("Überregeln"))
    assert kind == "info" and text.startswith("Überregeln: ") and "ρ = |1 − α| = 0,200" in text and "klingt in 3 Perioden" in text
    kind, text = ev.verdict_pulse(_preset("Fast-Totbeat"))
    assert kind == "success" and "kein Vorzeichenwechsel" in text and "klingt in 2 Perioden" in text and "Überschwinger" in text
    kind, text = ev.verdict_pulse(_preset("Doppelwurzel"))
    assert kind == "success" and "α liegt unter α* = 0,1481" in text and "klingt in 18 Perioden" in text and "Schätzung aus ρ: 11,9" in text
    kind, text = ev.verdict_pulse(_preset("Schnelle Rückkehr"))
    assert kind == "info" and "Stabil, schwingt" in text and "bis 0,0082 über das Soll" in text and "klingt in 12 Perioden" in text
    kind, text = ev.verdict_pulse(_preset("Hinter der Grenze"))
    assert kind == "error" and "ρ = 1,03686 > 1" in text and "α_krit = 0,6180" in text and "Faktor 1,0369" in text and "abklingen" not in text
    kind, text = ev.verdict_pulse(_preset("Lange Lieferzeit"))
    assert kind == "info" and "Stabil, schwingt" in text and "klingt in 243 Perioden" in text


def test_preset_help_texts_agree_with_the_measurements():
    a = _preset("Von Hand")
    assert a.share == pytest.approx(0.971, abs=5e-4) and a.decay == 542
    a = _preset("Überregeln")
    assert a.rho == pytest.approx(0.2, abs=1e-12)
    a = _preset("Fast-Totbeat")
    assert a.rho == pytest.approx(0.03, abs=1e-12)
    a = _preset("Doppelwurzel")
    assert a.decay == 18 and abs(0.148 - a.alpha_star) < 2e-3
    a = _preset("Schnelle Rückkehr")
    assert 0.175 / a.alpha_star == pytest.approx(1.18, abs=5e-3) and a.decay == 12 and a.max_pos < 0.01 * a.peak
    a = _preset("Hinter der Grenze")
    assert a.rho == pytest.approx(1.037, abs=5e-4) and a.alpha_krit == pytest.approx(0.618, abs=5e-4)
    a = _preset("Lange Lieferzeit")
    assert a.share == pytest.approx(0.73, abs=5e-3) and a.decay == 243


def test_branches_and_thresholds_with_hand_set_analyses():
    assert ev.verdict_pulse(_a(regime="unstable", rho=1.2, decay=None, estimate=None))[0] == "error"
    kind, text = ev.verdict_pulse(_a(regime="boundary", rho=1.0, decay=None, estimate=None, L=3))
    assert kind == "warning" and "4L − 2 = 10" in text
    assert ev.verdict_pulse(_a(peak=1.0 + 1e-6))[0] == "warning"                     # verstärkt (über 1 + 1e-9)
    assert ev.verdict_pulse(_a(peak=1.0 + 1e-10))[0] != "warning"                     # Rundung zählt nicht als Verstärkung
    assert ev.verdict_pulse(_a(max_pos=1e-9))[0] == "info"                            # Überschwinger über 1e-12
    assert ev.verdict_pulse(_a(max_pos=1e-13))[0] == "success"                        # darunter: monoton
    kind, text = ev.verdict_pulse(_a(L=1, alpha=1.5, rho=0.5, max_pos=0.5, decay=7))
    assert text.startswith("Überregeln")
    kind, text = ev.verdict_pulse(_a(L=1, alpha=0.5, rho=0.5, decay=7, max_pos=0.0))
    assert kind == "success"                                                          # L = 1, α ≤ 1: monoton, kein Überregeln
    kind, text = ev.verdict_pulse(_a(decay=None, horizon=40000))
    assert "nach 40 000 Perioden noch nicht auf 1 % der Spitze abgeklungen" in text and "Schätzung" not in text


def test_threshold_text_and_german_numbers():
    assert ev.thr_text(0.5) == "0,5 %" and ev.thr_text(1.0) == "1 %" and ev.thr_text(5.0) == "5 %"
    assert ev.de(-0.5, 1) == "−0,5" and ev.de(3.14159, 3) == "3,142" and ev.de(0.2, 1, sign=True) == "+0,2"
    assert ev.fmt_int(1479) == "1 479" and ev.fmt_int(7) == "7"
    assert ev.pct(0.9708, 1) == "97,1 %"


def test_verdict_shift():
    a = _preset("Von Hand")
    kind, text = ev.verdict_shift(a, 0.2, -0.3333, 1500)
    assert kind == "info" and "e = 0,20" in text and "bei −0,3333" in text and "−e/α = −0,3333" in text and "noch nicht eingeschwungen" not in text
    kind, text = ev.verdict_shift(a, 0.2, -0.2, 600)
    assert "noch nicht eingeschwungen" in text
    assert ev.verdict_shift(_preset("Hinter der Grenze"), 0.2, 5.0, 200)[0] == "warning"


def test_verdict_limit_with_the_frozen_counts():
    fr = ev.load_frozen()
    kind, text = ev.verdict_limit(3, 0.6, fr)
    assert kind == "success" and "α_krit = 0,618 bei L = 3" in text and "liegt unter der Grenze (97,1 % von α_krit)" in text
    assert "2, 1, 0,618, 0,445 für L = 1 bis 4" in text and "0 Ausnahmen" in text and "für L > 60 nicht gemessen" in text
    assert ev.verdict_limit(3, 0.7, fr)[0] == "error" and "über der Grenze" in ev.verdict_limit(3, 0.7, fr)[1]
    assert ev.verdict_limit(1, 2.0, fr)[0] == "warning" and "auf der Grenze" in ev.verdict_limit(1, 2.0, fr)[1]
    broken = {"stability_grid": {"below": 10, "above": 10, "exceptions_below": 1, "exceptions_above": 0}, "random_pairs": {"n_pairs": 5, "seed": 1, "exceptions": 0}}
    assert "1 Ausnahmen" in ev.verdict_limit(3, 0.6, broken)[1]


def test_verdict_roots_mentions_the_period_and_the_count():
    kind, text = ev.verdict_roots(3, 0.6, ev.period_check())
    assert kind == "success" and "Winkel π/(2L − 1) = 36,0°" in text and "Periode 4L − 2 = 10" in text and "0 von 39 Fällen (L = 2 bis 40)" in text
    assert "Wurzelbetrag 0,99145" in text
    assert "Bei L = 1 gibt es nur die Wurzel" in ev.verdict_roots(1, 1.2, ev.period_check())[1]
    assert ev.verdict_roots(3, 0.6, {"n": 39, "deviating": 2, "max_rel_dev": 0.01})[0] == "warning"


def test_verdict_decay_names_alpha_star_and_the_fastest_time():
    cells, jump = ev.decay_cells(), ev.jump_example()
    kind, text = ev.verdict_decay(3, 1.0, ev.fastest(3, 0.01), jump, cells)
    assert kind == "info" and "α* = 0,1481" in text and "18 Perioden bei α*, 11 Perioden im Gitter" in text and "1,19 bis 1,19 · α*" in text
    assert "α = 0,1769: 11 Perioden, α = 0,1772: 15 Perioden" in text and "für L = 3 317 Perioden" in text
    assert "trifft in 29 von 35 Zellen" in text and "in den 6 übrigen Zellen" in text
    kind, text = ev.verdict_decay(1, 1.0, ev.fastest(1, 0.01), jump, cells)
    assert "Totbeat" in text and "dauert 1 Periode" in text


def test_verdict_continuous_names_the_thresholds():
    kind, text = ev.verdict_continuous(ev.continuous_summary())
    assert "ab L = 51 dauerhaft innerhalb 1 %" in text and "ab L = 501 innerhalb 0,1 %" in text and "schon ab L = 4" in text and "ab L = 11" in text
    assert "Steigung −1,00" in text and "Steigung −2,00" in text and "bis L = 200 000" in text
