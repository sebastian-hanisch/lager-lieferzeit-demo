"""Ein Lager, eine Regel, eine Lieferzeit – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 1 der Linie „Rückkopplung und Verzögerung“: ein Lager mit der Bestellregel u = −α · Bestandsabweichung und der Lieferzeit L (Perioden).
Grenze, Schwingung und Abklingzeit; alles eigene Herleitung und Messung in diesem Modell (Zeitlage: llz_dynamics.py).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import llz_constants as C
import llz_dynamics as K
import llz_evaluation as ev
from llz_evaluation import Settings, de, fmt_int, pct
from llz_layout import dunkles_design, kopf_css
from llz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from llz_visualization import build_continuous, build_decay, build_map, build_pulse, build_roots

st.set_page_config(page_title="Ein Lager, eine Regel, eine Lieferzeit – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _shift(L, alpha, T):
    return K.shift_response(L, alpha, C.SHIFT_E, T)


@st.cache_data(show_spinner=False)
def _sums(L, alpha):
    return ev.sums_line(L, alpha)


@st.cache_data(show_spinner="Rechne ρ(L, α) für 40 Lieferzeiten …")
def _map():
    return ev.map_data()


@st.cache_data(show_spinner=False)
def _check():
    return ev.period_check()


@st.cache_data(show_spinner=False)
def _trace(L):
    return ev.root_trace(L)


@st.cache_data(show_spinner="Rechne Abklingzeiten über α …")
def _curve(L, thr):
    return ev.decay_curve(L, thr)


@st.cache_data(show_spinner="Suche die kürzeste Rückkehr …")
def _fastest(L, thr):
    return ev.fastest(L, thr)


@st.cache_data(show_spinner=False)
def _cells():
    return ev.decay_cells()


@st.cache_data(show_spinner=False)
def _jump():
    return ev.jump_example()


@st.cache_data(show_spinner=False)
def _table():
    return ev.results_table()


@st.cache_data(show_spinner=False)
def _cont():
    return ev.continuous_summary()


@st.cache_data(show_spinner=False)
def _cont_curve():
    Ls = np.unique(np.round(np.geomspace(1, 1000, 300)).astype(int))
    return Ls, K.continuous_table(Ls)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in llz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Ein Lager, eine Regel, eine Lieferzeit")
st.markdown(
    """
Ein **Lager** bestellt jede Periode nach einer einzigen Regel: **Bestellung = −α × Bestandsabweichung**. Die Ware kommt **L Perioden** später an. Die Demo zeigt in diesem Modell
(ohne Leitungsgewicht, ohne Meldeverzug, ohne Klemmung), **wann ein Bedarfsstoß abklingt**, mit welcher **Periode** der Bestand an der Grenze schwingt, warum **die Grenze keine gute
Einstellung** ist und wie die Grenze für große L in die der stetigen Gleichung (a·τ = π/2) übergeht. Das ist das erste Stück der Linie „Rückkopplung und Verzögerung“; alle Formeln sind
**eigene Herleitung und Messung in diesem Modell**, Literatur dazu wurde nicht gelesen.

Weiter unten: die **Kernfrage** mit Verdikt, fünf Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/lieferzeit-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich ein Lager vor, das jede Periode einen festen Anteil des Fehlbestands nachbestellt, während die Ware **L Perioden** unterwegs ist. Die Regel sieht nur, was im Lager liegt, nicht die Ware auf dem Weg, und bestellt deshalb zu viel nach: Der Bestand schwingt um den Sollwert (Deutung, nicht gemessen). Ab einer **Grenze für die Anpassung** klingt ein Bedarfsstoß nicht mehr ab, und je länger die Lieferzeit, desto kleiner ist diese Grenze. Knapp unter der Grenze kehrt der Bestand nur langsam zurück, die Grenze ist also keine gute Einstellung. Die Demo rechnet das in einem einfachen Modell nach (feste Lieferzeit, kein Meldeverzug, keine Klemmung der Bestellung) und misst Grenze, Abklingzeit und Summenregeln. Über echte Lager sagt sie nichts.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/lieferzeit-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Lager", expanded=True):
    st.markdown(
        """
1. **Bestand:** i_t ist der Bestand minus Soll in Periode t. Der Bedarf weicht um e_t vom Mittel ab; ein **Stoß** ist e_0 = 1 in Periode 0 und danach 0.
2. **Regel:** Die Bestellung ist **u_t = −α · i_t** (Fehlbestand wird zu einem Teil α nachbestellt). Sie steht **L Perioden später** im Bestand: i_(t+1) = i_t + u_(t+1−L) − e_t.
3. **Zeitlage:** **L = 1 heißt „eine Periode Wirkverzug“, nicht „keine Lieferzeit“**; eine Lieferzeit 0 gibt es in dieser Zählung nicht. Eine andere Zählung verschiebt L um eins und damit jede Zahl dieser Demo.
4. **Warum es schwingt (Deutung, nicht gemessen):** Die Regel sieht die schon bestellte, noch unterwegs befindliche Ware nicht (β = 0). Je länger die Lieferzeit, desto später merkt sie, dass sie zu viel bestellt hat.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lager**")
    L = st.slider("Lieferzeit L (Perioden)", *bounds("L_slider"), key="L_slider", help="Eine Bestellung der Periode t steht in der Periode t + L im Bestand. L = 1 ist eine Periode Wirkverzug.")
    alpha = st.slider("Anpassung α", *bounds("alpha_slider"), step=C.ALPHA_STEP, format="%.3f", key="alpha_slider",
                      help="Anteil des Fehlbestands, der je Periode nachbestellt wird. Die Grenze α_krit hängt von L ab (Anzeige unten).")
    st.caption(f"α_krit = {de(K.alpha_krit(int(L)), 4)} bei L = {int(L)}; α = {pct(float(alpha) / K.alpha_krit(int(L)))} der Grenze.")
    st.markdown("**Abklingzeit**")
    thr_pct = st.select_slider("Schwelle", options=list(C.THR_OPTIONS), format_func=lambda x: f"{x:g} %".replace(".", ","), key="thr_select",
                               help="Die Abklingzeit ist die letzte Periode, in der |i_t| über dieser Schwelle der Spitze max|i| liegt. Die Schwelle ist eine Wahl, keine Naturgröße.")

L, alpha, thr_pct = int(L), round(float(alpha), 3), float(thr_pct)
sync_query_params({"L_slider": L, "alpha_slider": alpha, "thr_select": thr_pct})

settings = Settings(L, alpha, thr_pct)
a = _analysis(settings)

# --- Ergebnis: Stoßantwort -----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Experiment 1: ein Bedarfsstoß trifft das Lager")
mode = st.radio("Bedarf", ["Einmaliger Stoß (e₀ = 1)", f"Dauerhafte Verschiebung (e = {de(C.SHIFT_E, 1)} je Periode)"], horizontal=True, key="bedarf_mode",
                help="Stoß: der Bedarf liegt in Periode 0 einmal um 1 über dem Mittel. Verschiebung: er liegt dauerhaft um e darüber.")
shift_mode = mode.startswith("Dauerhafte")

m1, m2, m3 = st.columns(3)
m1.metric("Spektralradius ρ", de(a.rho, 5), delta="stabil" if a.regime == "stable" else ("an der Grenze" if a.regime == "boundary" else "kippt"),
          delta_color="off", help="Größter Wurzelbetrag von P(z) = z^L − z^(L−1) + α. Unter 1 klingt der Stoß ab.")
m2.metric("Grenze α_krit", de(a.alpha_krit, 4), delta=f"α = {pct(a.share)} der Grenze", delta_color="off", help="2 sin(π/(4L − 2)); die Schwingung an der Grenze hat die Periode 4L − 2.")
m3.metric("Spitze max|i|", de(a.peak, 3), delta="Stoß = 1" if a.regime == "stable" else None, delta_color="off", help="Größte Bestandsabweichung nach dem Stoß; über 1 wird der Stoß verstärkt.")
m4, m5, m6 = st.columns(3)
m4.metric(f"Abklingzeit ({de(thr_pct, 1 if thr_pct != int(thr_pct) else 0)} %)", f"{fmt_int(a.decay)} Perioden" if a.decay is not None else ("—" if a.regime != "stable" else "nicht abgeklungen"),
          delta=f"Schätzung aus ρ: {de(a.estimate, 1)}" if a.estimate is not None and a.decay is not None else None, delta_color="off",
          help="Letzte Periode mit |i| über der Schwelle der Spitze; „—“ ohne Abklingen (ρ ≥ 1).")
sum_u, sum_i, sum_done = _sums(L, alpha)
m5.metric("Σ u (soll 1)", sum_u, help="Summe aller Bestellungen nach dem Stoß; im stabilen Fall gleich dem Stoß (eigene Herleitung, unabhängig von L).")
m6.metric("Σ i (soll −1/α)", sum_i, delta=f"−1/α = {de(-1 / alpha, 3)}" if a.regime == "stable" else None, delta_color="off", help="Fehlbestandsfläche; unabhängig von L.")

if shift_mode:
    T_shift = int(max(a.horizon, 60)) if a.regime == "stable" else 200   # gerechnet bis zum Horizont der Stoßantwort, angezeigt höchstens PLOT_CAP
    shift = _shift(L, alpha, T_shift)
    st.plotly_chart(build_pulse(a, shift=shift), width="stretch", key="pulse_chart")
else:
    st.plotly_chart(build_pulse(a), width="stretch", key="pulse_chart")
    if a.decay is not None and a.decay > C.PLOT_CAP:
        st.caption(f"Angezeigt sind die ersten {C.PLOT_CAP} Perioden; die Abklingzeit ({fmt_int(a.decay)}) liegt außerhalb.")
    if a.regime == "unstable":
        st.caption("Die Anzeige endet nach 200 Perioden oder, wenn |i| vorher 10¹² übersteigt.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Klingt der Stoß ab, und wie schnell?")
if shift_mode:
    kind, text = ev.verdict_shift(a, C.SHIFT_E, float(shift[-1]), len(shift) - 1)
else:
    kind, text = ev.verdict_pulse(a)
getattr(st, kind)(text)
st.caption("Alle Zahlen: in diesem Modell, β = 0 (Leitung nicht mitgerechnet), kein Meldeverzug, keine Klemmung, feste Lieferzeit, Zeitlage wie oben.")

st.markdown("---")

# --- Experiment 2: Grenze gegen L ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Grenze α_krit gegen die Lieferzeit")
st.caption("Farbfläche: Spektralradius ρ(L, α) für L = 1 bis 40, grün unter 1 (stabil), rot über 1 (kippt). Linie: α_krit = 2 sin(π/(4L − 2)); Kreise: dieselbe Grenze aus den Eigenwerten gefunden (Bisektion auf ρ = 1).")
mp = _map()
st.plotly_chart(build_map(mp, L, alpha), width="stretch", key="map_chart")
kind, text = ev.verdict_limit(L, alpha, ev.load_frozen(), ev.krit_strictly_falling())
getattr(st, kind)(text)
st.caption(f"Formel gegen Eigenwerte: größte Abweichung {mp['max_dev']:.1e} über L = 1 bis 40 (Bisektion auf 60 Schritte). Die Zählung „0 Ausnahmen“ ist vorgerechnet (`tools/make_frozen.py`, `data/llz_frozen.json`).")

st.markdown("---")

# --- Experiment 3: Periode und Wurzeln -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Periode und Wurzeln an der Grenze")
st.markdown(
    "Der Bestand klingt ab, solange alle Wurzeln von P(z) = z^L − z^(L−1) + α im Einheitskreis liegen. Eine Wurzel e^(iθ) **auf** dem Kreis verlangt z^(L−1)(z − 1) = −α. "
    "Der **Betrag** gibt |z − 1| = 2 sin(θ/2) = α, die **Phase** gibt (L − ½)·θ = π/2, also θ = π/(2L − 1) und **α_krit = 2 sin(π/(4L − 2))**; die Periode ist 2π/θ = **4L − 2**."
)
shares, trace = _trace(L)
_c1, _c2, _c3 = st.columns([1, 1.6, 1])
_c2.plotly_chart(build_roots(L, alpha, shares, trace), width="stretch", key="roots_chart")
kind, text = ev.verdict_roots(L, alpha, _check())
getattr(st, kind)(text)
st.caption("Farbe der Punkte: α als Anteil an α_krit (0 bis 1,5). Die Wurzeln wandern bei wachsendem α nach außen und überschreiten den Kreis bei α_krit.")

st.markdown("---")

# --- Experiment 4: Abklingzeit -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Abklingzeit gegen Spektralradius")
st.caption(f"Stoßantwort, L = {L}, Schwelle {de(thr_pct, 1 if thr_pct != int(thr_pct) else 0)} %. Oben ρ(α), unten die gemessene Abklingzeit (Punkte, α bis 0,95 · α_krit) gegen die Schätzung ln(Schwelle)/ln ρ "
           "(Linie); rot: die Gitterpunkte mit der kürzesten gemessenen Zeit, schwarz: das gewählte α. Zeiten, die bis zum Horizont von "
           f"{fmt_int(C.CURVE_HORIZON)} Perioden nicht abklingen, fehlen.")
fr = _fastest(L, a.thr)
st.plotly_chart(build_decay(L, _curve(L, a.thr), fr, alpha, thr_pct), width="stretch", key="decay_chart")
kind, text = ev.verdict_decay(L, thr_pct, fr, _jump(), _cells())
getattr(st, kind)(text)
st.caption("Die kürzeste Zeit und ihre Lage gelten für dieses Gitter und diese Schwelle; die Zeit ist keine glatte Funktion von α. Ein Mechanismus für „kleinste Zeit ≈ ln(Schwelle)/ln((L − 1)/L)“ "
           "ist nicht hergeleitet, es ist eine Messung (L = 2 bis 24, Schwellen 0,5 bis 5 %).")

with st.expander("Ergebnistabelle (Messung, Bedarfsstoß, Schwelle 1 %)", expanded=False):
    rows = _table()
    st.dataframe(
        [{"L": r["L"], "α_krit": de(r["alpha_krit"], 4), "α_krit · L": de(r["aL"], 3), "Periode 4L − 2": r["period"], "α*": de(r["alpha_star"], 4), "ρ(α*)": de(r["rho_star"], 3),
          "Zeit bei α*": r["t_star"], "kleinste Zeit (Gitter)": r["t_min"], "Zeit bei 0,95 · α_krit": fmt_int(r["t_near"])} for r in rows],
        hide_index=True, width="stretch",
    )
    st.caption("Zeiten in Perioden auf 1 % der Spitze. „Kleinste Zeit“: Gitter von 4000 Punkten in (0, α_krit · 0,999); sie hängt an der Schwelle und am Gitter.")

st.markdown("---")

# --- Experiment 5: stetige Grenze ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Die Brücke zur stetigen Grenze")
st.caption("Für große L gleicht α_krit · L dem Wert π/2 der stetigen Gleichung x′ = −a·x(t − τ) (a·τ = π/2). Das Lager ist das Euler-Schema mit m = L − 1 Schritten je Verzögerung; "
           "die Rechnung dazu steht in der [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo) (Befund B4), hier nur als Brücke.")
Ls_c, tab_c = _cont_curve()
st.plotly_chart(build_continuous(Ls_c, tab_c), width="stretch", key="cont_chart")
summ = _cont()
kind, text = ev.verdict_continuous(summ)
getattr(st, kind)(text)
st.dataframe(
    [{"L": r["L"], "α_krit": de(r["alpha_krit"], 4), "α_krit · L": de(r["aL"], 4), "Abstand zu π/2": pct(r["rel_L"], 2).replace("+", ""), "1/(2L − 1)": de(r["inv"], 5),
      "α_krit · (L − ½)": de(r["a_half"], 4), "Abstand (L − ½)": pct(r["rel_half"], 3), "α_krit · (L − 1)": de(r["a_m1"], 4)} for r in ev.continuous_rows()],
    hide_index=True, width="stretch",
)
st.caption("α_krit · L fällt nur nicht streng: α_krit(1) · 1 = α_krit(2) · 2 = 2 exakt (sin(π/2) = 1, sin(π/6) = ½); eine Rundungsdifferenz im Rechner ist kein Gefälle. "
           "„Dauerhaft innerhalb“ heißt: für alle L bis 200 000 geprüft, nicht bewiesen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Leitung nicht mitgerechnet (β = 0)** | Die Regel sieht die Ware auf dem Weg nicht; deshalb sinkt die Grenze mit L. Mit einem Gewicht β > 0 auf die Leitung ändert sich die Grenze (hier nicht gerechnet). | [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo) |
| **Kein Meldeverzug (s = 0)** | Der Bestand wird ohne Verzug gesehen; eine späte Meldung wirkt bei β = 0 wie eine um s längere Lieferzeit (mit Leitungsgewicht kann sie schlimmer wirken). | [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) |
| **Feste Lieferzeit** | Streut die Lieferzeit, gilt die Grenze dieser Demo nicht; sie ändert sich. | [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo) |
| **Nur Stoß und Dauerverschiebung** | Saison, Rampe und Rauschen sind nicht gemessen (die Summenregeln sind die Verstärkung bei Frequenz 0). | [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) |
| **Keine Klemmung** | Die Bestellung darf negativ werden. Mit der Klemmung (Bestellung ≥ 0) gibt es hinter der Grenze Grenzzyklen und je nach α und L Chaos. | [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) |
| **Diskrete Perioden** | Die stetige Gleichung hat a·τ = π/2 als Grenze; die Brücke steht in Experiment 5. | [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo) |
| **Rechnerische Grenzen** | An der Doppelwurzel α* verlieren Eigenwertlöser etwa 8 Stellen (Jordan-Block); α wird nie auf eine Zweierpotenz voreingestellt. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/lieferzeit-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Lieferzeit $L \ge 1$ (Perioden), die Anpassung $\alpha > 0$ und ein Bedarfsstoß; gesucht sind die Bedingung, unter der der Bestand zum Soll zurückkehrt, und die Zeit dafür.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand minus Soll in Periode $t$ ($i_t = 0$ für $t \le 0$) |
| $u_t$ | Bestellung minus Mittelbedarf in Periode $t$ |
| $e_t$ | Bedarfsabweichung in Periode $t$ (Stoß: $e_0 = 1$, sonst 0) |
| $L$, $\alpha$ | Lieferzeit in Perioden, Anpassung |
| $P(z)$, $\rho$ | charakteristisches Polynom, größter Wurzelbetrag |
| $\alpha_{krit}$, $\alpha^*$ | Stabilitätsgrenze, Anpassung der Doppelwurzel |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/lieferzeit-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind die Lieferzeit $L \ge 1$ (Perioden), die Anpassung $\alpha > 0$ und ein Bedarfsstoß; gesucht sind die Bedingung, unter der der Bestand zum Soll zurückkehrt, und die Zeit dafür.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand minus Soll in Periode $t$ ($i_t = 0$ für $t \le 0$) |
| $u_t$ | Bestellung minus Mittelbedarf in Periode $t$ |
| $e_t$ | Bedarfsabweichung in Periode $t$ (Stoß: $e_0 = 1$, sonst 0) |
| $L$, $\alpha$ | Lieferzeit in Perioden, Anpassung |
| $P(z)$, $\rho$ | charakteristisches Polynom, größter Wurzelbetrag |
| $\alpha_{krit}$, $\alpha^*$ | Stabilitätsgrenze, Anpassung der Doppelwurzel |

**Modell.** $u_t = -\alpha\, i_t$ und $i_{t+1} = i_t + u_{t+1-L} - e_t$, also

$$i_{t+1} = i_t - \alpha\, i_{t+1-L} - e_t .$$

**Charakteristisches Polynom.** Der Ansatz $i_t = z^t$ ohne Bedarf gibt

$$P(z) = z^L - z^{L-1} + \alpha = 0 .$$

Der Stoß klingt ab, wenn $\rho = \max |z| < 1$ ist.

**Grenze (eigene Herleitung).** Eine Wurzel $z = e^{\mathrm{i}\theta}$ auf dem Kreis verlangt $z^{L-1}(z - 1) = -\alpha$. Der Betrag gibt $|z - 1| = 2\sin(\theta/2) = \alpha$, die Phase
$(L-1)\theta + \theta/2 + \pi/2 = \pi$, also $(L - \tfrac12)\,\theta = \tfrac{\pi}{2}$:

$$\theta = \frac{\pi}{2L - 1}, \qquad \alpha_{krit} = 2\sin\frac{\pi}{4L - 2}, \qquad \text{Periode } \frac{2\pi}{\theta} = 4L - 2 .$$

Dass der Stoß **genau** für $0 < \alpha < \alpha_{krit}$ abklingt, ist hier **gemessen** (Gitter und Zufallspaare, siehe Experiment 2), nicht bewiesen.

**Doppelwurzel.** Wegen $P'(z) = z^{L-2}\,(Lz - (L-1))$ ist $z^* = (L-1)/L$ die Stelle, an der $P$ und $P'$ zugleich verschwinden, wenn

$$\alpha^* = \frac{1}{L}\left(\frac{L-1}{L}\right)^{L-1} \quad (L = 1:\ \alpha^* = 1) .$$

Dort ist $\rho = (L-1)/L$; das Minimum von $\rho(\alpha)$ bei $\alpha^*$ ist für $L = 2$ bis $12$ im Gitter gemessen.

**Summenregeln.** Summiert man die Rekursion über alle $t$ und nutzt $i_0 = i_\infty = 0$ und $\sum e_t = 1$, so folgt für stabile $(L, \alpha)$

$$\sum_t i_t = -\frac{1}{\alpha}, \qquad \sum_t u_t = 1 ,$$

unabhängig von $L$ (es ist $P(1) = \alpha$). Bei dauerhaft verschobenem Bedarf $e$ sitzt der Bestand bei $i_\infty = -e/\alpha$ (stationär: $u = e$).

**Abklingzeit.** $t_{ab}$ ist die letzte Periode mit $|i_t| > s \cdot \max |i|$ (Schwelle $s$, eine Wahl). Die Schätzung $\ln s / \ln \rho$ ist nur eine Schätzung (Experiment 4).

**Stetige Grenze.** $i_{t+1} = i_t - \alpha\, i_{t-(L-1)}$ ist das Euler-Verfahren für $x'(t) = -a\,x(t-\tau)$ mit $m = L - 1$ Schritten je Verzögerung und $\alpha = a h$, $h = \tau / m$; $\alpha_{krit}\,(L-1) = 2m\sin\frac{\pi}{4m+2}$.

**Literatur.** Keine Quelle gelesen; alle Formeln sind eigene Herleitung in diesem Modell. Dass sie in der Fachliteratur zu Bestandsregeln mit Verzögerung bekannt sind, ist wahrscheinlich, aber ungeprüft; eine Neuheitsaussage gibt es nicht.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
