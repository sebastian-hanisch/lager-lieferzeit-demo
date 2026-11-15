"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): L = 1 (kein Pipeline-Zustand), genau L = 2, L = 40, α an den Reglergrenzen,
α knapp an der Grenze, alle Schwellen; jede Kombination läuft durch Auswertung und Verdikt, ohne Fehler und ohne erfundene Zahl."""

import numpy as np
import pytest

import llz_constants as C
import llz_dynamics as K
import llz_evaluation as ev


def test_state_matrix_for_one_and_two_periods():
    assert K.state_matrix(1, 0.3) == pytest.approx(np.array([[0.7]]))
    assert K.state_matrix(2, 0.3) == pytest.approx(np.array([[1.0, -0.3], [1.0, 0.0]]))
    assert K.state_matrix(3, 0.3) == pytest.approx(np.array([[1.0, 0.0, -0.3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert K.char_poly(1, 0.3) == pytest.approx([1.0, -0.7]) and K.char_poly(2, 0.3) == pytest.approx([1.0, -1.0, 0.3])


def test_exactly_two_periods_has_a_boundary_and_a_double_root():
    """L = 2 ist der kleinste Fall mit Pipeline: Grenze 1, α* = 1/4 (Zweierpotenz-Ausnahmefälle, hier nur über die Nachbarn geprüft)."""
    assert K.spectral_radius(2, 0.9990) < 1.0 < K.spectral_radius(2, 1.0010)
    # Nachbarn von α* = 1/4: z = (1 ± √(1 − 4α))/2, bei α = 0,2495 also 0,5 + 0,0224 (Wurzelverhalten am Doppelpunkt: die Abweichung wächst wie √Abstand)
    assert K.spectral_radius(2, 0.2495) == pytest.approx(0.5 + 0.5 * (1 - 4 * 0.2495) ** 0.5, abs=1e-12)
    assert K.spectral_radius(2, 0.2495) < 0.53


@pytest.mark.parametrize("L", [1, 2, 3, 12, 40])
@pytest.mark.parametrize("alpha", [C.ALPHA_MIN, C.ALPHA_MAX])
@pytest.mark.parametrize("thr_pct", C.THR_OPTIONS)
def test_regler_corners_run_through_analysis_and_verdict(L, alpha, thr_pct):
    a = ev.analyse(ev.Settings(L, alpha, thr_pct))
    kind, text = ev.verdict_pulse(a)
    assert kind in ("success", "info", "warning", "error") and text
    if alpha == C.ALPHA_MAX:
        assert a.regime == "unstable" and a.decay is None and kind == "error"     # 2,5 liegt für jedes L ≥ 1 über α_krit ≤ 2
    else:
        assert a.regime == "stable" and a.decay is not None and a.rho < 1.0


def test_just_below_the_boundary_the_decay_is_very_long_and_capped_honestly():
    """0,999 · α_krit: ρ knapp unter 1; reicht der Horizont nicht, steht „noch nicht abgeklungen“ (decay = None) statt einer Zahl."""
    a = ev.analyse(ev.Settings(12, 0.999 * K.alpha_krit(12), 1.0))
    assert a.regime == "stable"
    kind, text = ev.verdict_pulse(a)
    if a.decay is None:
        assert "noch nicht" in text
    else:
        assert a.decay > 10000                                 # Größenordnung: Abklingzeit wächst wie 1/(1 − ρ)


def test_exactly_on_the_boundary_is_not_called_stable():
    a = ev.analyse(ev.Settings(1, 2.0, 1.0))                   # α_krit(1) = 2 (Zweierpotenz, benannter Ausnahmefall): ρ = |1 − α| = 1
    assert a.regime == "boundary" and a.decay is None
    assert ev.verdict_pulse(a)[0] == "warning"


def test_impulse_response_never_returns_a_made_up_decay_for_unstable_cases():
    for L in (1, 3, 12):
        a = K.analyse_pulse(L, 1.1 * K.alpha_krit(L), 0.01)
        assert a.regime == "unstable" and a.decay is None and a.estimate is None


def test_sums_are_undefined_without_decay():
    assert K.pulse_sums(3, 0.7) == (None, None, False)
    assert ev.sums_line(3, 0.7)[0] == "nicht definiert"
