"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import llz_constants as C
import llz_dynamics as K


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_pulse(a: K.PulseAnalysis, shift: np.ndarray | None = None, e: float = C.SHIFT_E):
    """Bestand i_t (oben) und Bestellung u_t = −α i_t (unten) als Stäbe; Schwelle und Abklingzeit eingezeichnet.
    Mit `shift`: Bestand bei dauerhaft verschobenem Bedarf e als Linie mit der Marke −e/α."""
    T = min(len(a.i) - 1, C.PLOT_CAP)
    if a.decay is not None:
        T = min(T, max(30, int(a.decay * 1.25) + 5))
    t = np.arange(T + 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Bestandsabweichung i_t", "Bestellung u_t = −α · i_t"))
    if shift is None:
        fig.add_trace(go.Bar(x=t[1:], y=a.i[1: T + 1], marker_color=C.COLOR_STOCK, name="Bestand i_t", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=t[1:], y=-a.alpha * a.i[1: T + 1], marker_color=C.COLOR_ORDER, name="Bestellung u_t", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"), row=2, col=1)
        if a.regime == "stable":
            band = a.thr * a.peak
            for sgn in (1, -1):
                fig.add_hline(y=sgn * band, line=dict(color=C.COLOR_GREY, dash="dot", width=1), row=1, col=1)
        if a.decay is not None and a.decay <= T:
            fig.add_vline(x=a.decay, line=dict(color=C.COLOR_BAD, dash="dash", width=1.5), annotation_text=f"Abklingzeit {a.decay}", annotation_position="top right")
    else:
        s = np.asarray(shift)[: T + 1]
        fig.add_trace(go.Scatter(x=t[: len(s)], y=s, mode="lines", line=dict(color=C.COLOR_STOCK, width=2), name="Bestand i_t", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t[: len(s)], y=-a.alpha * s, mode="lines", line=dict(color=C.COLOR_ORDER, width=2), name="Bestellung u_t", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"), row=2, col=1)
        if a.regime == "stable":
            fig.add_hline(y=-e / a.alpha, line=dict(color=C.COLOR_BAD, dash="dash", width=1.5), row=1, col=1, annotation_text="−e/α", annotation_position="bottom right")
            fig.add_hline(y=e, line=dict(color=C.COLOR_BAD, dash="dash", width=1.5), row=2, col=1, annotation_text="Bedarf e", annotation_position="bottom right")
    fig.update_xaxes(title_text="Periode t", row=2, col=1)
    fig.update_layout(showlegend=False)
    return _base(fig, height=470)


def build_map(data: dict, L: int, alpha: float):
    """Farbfläche ρ(L, α) (Mitte der Skala: ρ = 1) mit α_krit aus der Formel (Linie), aus den Eigenwerten (Punkte) und dem gewählten (L, α)."""
    z = np.clip(np.array(data["rho"]).T, 0.0, 2.0)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=data["L"], y=data["alpha"], z=z, zmin=0.0, zmax=2.0, zsmooth=False, name="ρ",
                             colorscale=[[0.0, "#1b6a5f"], [0.45, "#e3efec"], [0.5, "#ffffff"], [0.55, "#f6d5c8"], [1.0, "#b23a2a"]],
                             colorbar=dict(title=dict(text="ρ", side="top"), tickvals=[0, 0.5, 1, 1.5, 2], len=0.9),
                             hovertemplate="L = %{x}, α = %{y:.3f}: ρ = %{z:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=data["L"], y=data["formula"], mode="lines", line=dict(color="#222", width=2), name="α_krit = 2 sin(π/(4L − 2))",
                             hovertemplate="L = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=data["L"], y=data["eigen"], mode="markers", marker=dict(color="#222", size=5, symbol="circle-open"), name="aus den Eigenwerten (Bisektion auf ρ = 1)",
                             hovertemplate="L = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[L], y=[alpha], mode="markers", marker=dict(color="#fff", size=12, symbol="star", line=dict(color="#000", width=1.5)), name="gewählt (L, α)",
                             hovertemplate="L = %{x}, α = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Lieferzeit L (Perioden)")
    fig.update_yaxes(title_text="Anpassung α (logarithmisch)", type="log", range=[math.log10(0.01), math.log10(C.ALPHA_MAX)])
    return _base(fig, height=430)


def build_roots(L: int, alpha: float, shares: np.ndarray, trace: np.ndarray):
    """Wurzeln von P in der komplexen Ebene: Einheitskreis, Spur für α = 0 bis 1,5 · α_krit (Farbe = Anteil), die Wurzeln beim gewählten α."""
    fig = go.Figure()
    th = np.linspace(0.0, 2.0 * math.pi, 361)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color=C.COLOR_GREY, width=1), name="Einheitskreis", hoverinfo="skip"))
    xs = np.real(trace).ravel()
    ys = np.imag(trace).ravel()
    col = np.repeat(shares, trace.shape[1])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=4, color=col, colorscale="RdYlGn_r", cmin=0.0, cmax=1.5, showscale=True,
                                                                     colorbar=dict(title=dict(text="α / α_krit", side="top"), len=0.8, tickvals=[0, 0.5, 1, 1.5])),
                             name="Spur der Wurzeln", hovertemplate="Re = %{x:.3f}, Im = %{y:.3f}<extra></extra>"))
    ak = K.alpha_krit(L)
    zk = K.roots_p(L, ak * (1.0 + 1e-6))
    fig.add_trace(go.Scatter(x=np.real(zk), y=np.imag(zk), mode="markers", marker=dict(size=9, color="#222", symbol="diamond-open"), name="bei α_krit",
                             hovertemplate="Re = %{x:.3f}, Im = %{y:.3f}<extra></extra>"))
    z = K.roots_p(L, alpha)
    fig.add_trace(go.Scatter(x=np.real(z), y=np.imag(z), mode="markers", marker=dict(size=11, color="#fff", symbol="star", line=dict(color="#000", width=1.5)), name="beim gewählten α",
                             hovertemplate="Re = %{x:.3f}, Im = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Realteil", range=[-1.7, 1.7])
    fig.update_yaxes(title_text="Imaginärteil", range=[-1.7, 1.7])
    return _base(fig, height=460)


def build_decay(L: int, curve: dict, f: dict, alpha: float, thr_pct: float):
    """Oben ρ(α) mit α*; unten gemessene Abklingzeit (Punkte) gegen die Schätzung ln(Schwelle)/ln ρ (Linie), logarithmische Zeitachse."""
    a = curve["alpha"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Spektralradius ρ(α)", f"Abklingzeit auf {thr_pct:g} % der Spitze (Perioden)".replace(".", ",")))
    fig.add_trace(go.Scatter(x=a, y=curve["rho"], mode="lines", line=dict(color=C.COLOR_LINE, width=2.5), name="ρ", hovertemplate="α = %{x:.4f}: ρ = %{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=a, y=curve["measured"], mode="markers", marker=dict(size=4, color=C.COLOR_STOCK), name="gemessen",
                             hovertemplate="α = %{x:.4f}: %{y:.0f} Perioden<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=a, y=curve["estimate"], mode="lines", line=dict(color=C.COLOR_ORDER, width=1.5, dash="dot"), name="Schätzung ln(Schwelle)/ln ρ",
                             hovertemplate="α = %{x:.4f}: %{y:.1f}<extra></extra>"), row=2, col=1)
    ast = f["alpha_star"]
    if ast <= float(a.max()):
        for r in (1, 2):
            fig.add_vline(x=ast, line=dict(color=C.COLOR_GREY, dash="dash", width=1), row=r, col=1, annotation_text="α*" if r == 1 else None, annotation_position="top")
    if f["alpha_lo"] <= float(a.max()):
        fig.add_vrect(x0=f["alpha_lo"], x1=max(f["alpha_hi"], f["alpha_lo"] * 1.003), fillcolor="rgba(178,58,42,0.35)", line_width=0, row=2, col=1)
    if alpha <= float(a.max()):
        for r in (1, 2):
            fig.add_vline(x=alpha, line=dict(color="#000", width=1), row=r, col=1)
    fig.update_yaxes(title_text="ρ", row=1, col=1)
    fig.update_yaxes(title_text="Perioden", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Anpassung α", row=2, col=1)
    return _base(fig, height=520)


def build_continuous(Ls, table: dict):
    """α_krit · L, α_krit · (L − ½) und α_krit · (L − 1) gegen π/2 über L (logarithmische Achse)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ls, y=table["L"], mode="lines", line=dict(color=C.COLOR_LINE, width=2.5), name="α_krit · L", hovertemplate="L = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=Ls, y=table["L_half"], mode="lines", line=dict(color=C.COLOR_STOCK, width=2.5), name="α_krit · (L − ½)", hovertemplate="L = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=Ls, y=table["L_minus1"], mode="lines", line=dict(color=C.COLOR_ORDER, width=2.5), name="α_krit · (L − 1)", hovertemplate="L = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=math.pi / 2, line=dict(color="#222", dash="dash", width=1.5), annotation_text="π/2 (stetig)", annotation_position="bottom right")
    fig.update_xaxes(title_text="Lieferzeit L (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Wert", range=[0.9, 2.1])
    return _base(fig, height=380)
