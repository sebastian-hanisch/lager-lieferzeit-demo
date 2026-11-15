"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keine Zweierpotenz als α, Permalink begrenzt und rastet ein."""

import math

import llz_constants as C
import llz_evaluation as ev
import llz_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.L_MIN <= p["L"] <= C.L_MAX and C.ALPHA_MIN <= p["alpha"] <= C.ALPHA_MAX and p["thr"] in C.THR_OPTIONS, name


def test_no_preset_alpha_is_a_power_of_two():
    """Rechner-Artefakt (Dualbrüche werden exakt gerechnet): α weder voreinstellen noch als Zweierpotenz wählen."""
    for name, p in C.PRESETS.items():
        m, e = math.frexp(p["alpha"])
        assert m != 0.5, (name, p["alpha"])
    assert C.DEFAULT_ALPHA != 0.5 and math.frexp(C.DEFAULT_ALPHA)[0] != 0.5


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert ev.Settings(L=C.PRESETS[first]["L"], alpha=C.PRESETS[first]["alpha"], thr_pct=C.PRESETS[first]["thr"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_are_distinct_scenarios():
    values = [(p["L"], p["alpha"], p["thr"]) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_float_equality_for_the_highlight_is_tolerant():
    assert P._gleich(0.6, 0.6000000000000001) and not P._gleich(0.6, 0.601) and P._gleich(3, 3) and not P._gleich(3, 4)
