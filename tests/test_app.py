"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_permalink_is_clamped_and_snaps_the_threshold():
    """Adresszeile mit Werten außerhalb der Regler: L und α werden begrenzt, die Schwelle rastet auf die nächste Stufe ein; ein Text statt Zahl wird ignoriert."""
    at = AppTest.from_file(APP_PATH)
    at.query_params["L"] = "99"
    at.query_params["alpha"] = "-3"
    at.query_params["schwelle"] = "1.4"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="L_slider").value == 40
    assert at.sidebar.slider(key="alpha_slider").value == pytest.approx(0.001)
    assert at.sidebar.select_slider(key="thr_select").value == 1.0
    at2 = AppTest.from_file(APP_PATH)
    at2.query_params["L"] = "abc"
    at2.run(timeout=TIMEOUT)
    assert not at2.exception and at2.sidebar.slider(key="L_slider").value == 3


def test_every_experiment_section_and_the_expander_are_present():
    at = _fresh_app()
    heads = [h.value for h in at.subheader]
    assert any("Experiment 2" in h for h in heads) and any("Experiment 5" in h for h in heads) and any("Wo die Annahmen enden" in h for h in heads)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any(e.label.startswith("So funktioniert") for e in at.expander)


def test_demand_shift_mode_runs_for_every_preset_and_shows_its_verdict():
    """Umschalter „Dauerhafte Verschiebung“ (Hauptbereich, kein Preset-Regler): jedes Preset läuft ohne Fehler; stabile zeigen −e/α, instabile eine Warnung."""
    import llz_constants as C

    at = _fresh_app()
    for name, p in C.PRESETS.items():
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        at.radio(key="bedarf_mode").set_value(at.radio(key="bedarf_mode").options[1]).run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        texts = " ".join(m.value for kind in (at.info, at.warning, at.error) for m in kind)
        assert "Dauerhaft verschobener Bedarf" in texts, name
        if name == "Hinter der Grenze":
            assert "keinen stationären Bestand" in texts
        else:
            assert "−e/α = −" + f"{0.2 / p['alpha']:.4f}".replace(".", ",") in texts, name
        at.radio(key="bedarf_mode").set_value(at.radio(key="bedarf_mode").options[0]).run(timeout=TIMEOUT)
