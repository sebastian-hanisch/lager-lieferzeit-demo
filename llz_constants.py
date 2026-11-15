"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Presets: keine Zweierpotenz als alpha (Rechner-Artefakt, wenn Dualbrüche exakt gerechnet werden); die benannten Ausnahmefälle alpha_krit(1) = 2,
alpha_krit(2) = 1, alpha*(1) = 1, alpha*(2) = 1/4 stehen nur in den Tests, dort mit Brüchen und zusätzlich bei alpha (1 ± 1e-3)."""

L_MIN, L_MAX, DEFAULT_L = 1, 40, 3
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.001, 2.5, 0.001, 0.6
THR_OPTIONS = (0.5, 1.0, 2.0, 5.0)           # Schwelle der Abklingzeit in Prozent der Spitze
DEFAULT_THR = 1.0
SHIFT_E = 0.2                                # Höhe der dauerhaften Bedarfsverschiebung (Stoß: 1)
PLOT_CAP = 600                               # längste Anzeige der Stoßantwort in Perioden
MAP_ALPHAS = 120                             # Zahl der alpha-Werte der Farbfläche
STABLE_SAMPLE = 400                          # Kurvenpunkte der Abklingzeit über alpha
CURVE_HORIZON = 8000                         # Horizont der Kurve der Abklingzeit über alpha

COLOR_STOCK = "#2F6B65"
COLOR_ORDER = "#c9742a"
COLOR_LINE = "#4c78a8"
COLOR_BAD = "#b23a2a"
COLOR_GREY = "rgba(150,150,150,0.8)"

PRESETS = {
    "Von Hand": {"L": 3, "alpha": 0.6, "thr": 1.0},
    "Überregeln": {"L": 1, "alpha": 1.2, "thr": 1.0},
    "Fast-Totbeat": {"L": 1, "alpha": 0.97, "thr": 1.0},
    "Doppelwurzel": {"L": 3, "alpha": 0.148, "thr": 1.0},
    "Schnelle Rückkehr": {"L": 3, "alpha": 0.175, "thr": 1.0},
    "Hinter der Grenze": {"L": 3, "alpha": 0.7, "thr": 1.0},
    "Lange Lieferzeit": {"L": 12, "alpha": 0.1, "thr": 1.0},
}
PRESET_HELP = {
    "Von Hand": "Lieferzeit 3, α = 0,6 (0,971 · α_krit): das Handbeispiel. Stabil, aber der Stoß wird erst verstärkt und klingt in 542 Perioden ab.",
    "Überregeln": "Lieferzeit 1 (eine Periode Wirkverzug), α = 1,2: der Bestand springt über das Soll, das Vorzeichen wechselt je Periode, die Abweichung klingt trotzdem ab (ρ = 0,2).",
    "Fast-Totbeat": "Lieferzeit 1, α = 0,97: fast die ganze Abweichung wird in einer Periode ausgeglichen (ρ = 0,03).",
    "Doppelwurzel": "Lieferzeit 3, α nahe α* = 0,1481: der Spektralradius ist hier am kleinsten, die Rückkehr dauert trotzdem 18 Perioden.",
    "Schnelle Rückkehr": "Lieferzeit 3, α = 0,175 (1,18 · α*): kürzere Rückkehr als bei α*, mit einem kleinen Überschwinger unter der Schwelle von 1 %.",
    "Hinter der Grenze": "Lieferzeit 3, α = 0,7 > α_krit = 0,618: die Auslenkung wächst je Periode um den Faktor ρ = 1,037.",
    "Lange Lieferzeit": "Lieferzeit 12, α = 0,1 (0,73 · α_krit): stabil, aber die Abweichung klingt erst nach 243 Perioden ab.",
}
