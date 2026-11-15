"""Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben). Die Demo hat keinen Zufall und keinen Zufalls-Seed."""

import math
from dataclasses import dataclass

import streamlit as st

import llz_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None


SETTING_SPECS = {
    "L_slider": SettingSpec("L", int, C.DEFAULT_L, C.L_MIN, C.L_MAX),
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "thr_select": SettingSpec("schwelle", float, C.DEFAULT_THR, min(C.THR_OPTIONS), max(C.THR_OPTIONS)),
}
PRESET_KEYS = {"L": "L_slider", "alpha": "alpha_slider", "thr": "thr_select"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Schwelle auf die erlaubten Stufen eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if not math.isfinite(value):
                continue
            value = max(spec.lo, min(spec.hi, value))
            if state_key == "thr_select":
                value = min(C.THR_OPTIONS, key=lambda o: abs(o - value))
            if state_key == "alpha_slider":
                value = round(value, 3)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Gleitkommawerte auf 1e-9), sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. tests/test_layout.py prüft für jeden Regler, dass seine Änderung die Markierung aufhebt."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
