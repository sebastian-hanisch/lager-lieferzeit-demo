# Ein Lager, eine Regel, eine Lieferzeit – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lager-lieferzeit-demo.streamlit.app/)**

Stück 1 der Linie **Rückkopplung und Verzögerung** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es ist die Wurzel der Linie und zugleich die lineare Hälfte zur
Schwesterlinie Dynamische Systeme: [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) benutzt dasselbe Lager (ohne Leitungsgewicht) mit Klemmung und misst Chaos; die Grenze α_krit = 2 sin(π/(4L − 2)) und die Periode 4L − 2 sind dort nur
Ausgangspunkt und Testwerte. Hier kommt die **lineare** Hälfte: Stoßantwort, Abklingzeit, welches α unterhalb der Grenze gut ist, Summenregeln. Die stetige Delay-Differentialgleichung (DDE) rechnet die
[`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo); hier steht nur die Brücke dorthin. Nachbarn in der Linie: [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo) (Leitungsgewicht β),
[`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (Meldeverzug s), [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo) (streuende Lieferzeit) und
[`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) (Saison und Rauschen). Das Lager ist exakt das Euler-Schema mit m = L − 1 Schritten je Verzögerung; die Rechnung dazu ist Befund B4 der `verzoegerung-hopf-demo`, hier nur ein Verweis.

## Warum dieses Problem

Ein Lager bestellt nach **einer** Regel: Bestellung = −α × Bestandsabweichung, die Ware kommt L Perioden später. **Klingt ein Bedarfsstoß ab, mit welcher Periode schwingt der Bestand an der Grenze, und ist die Grenze eine gute Einstellung?**
In diesem Modell (β = 0, s = 0, ohne Klemmung, feste Lieferzeit) klingt ein Stoß genau für 0 < α < α_krit ab (gemessen, siehe B2): α_krit ist 2 (L = 1), 1 (L = 2), 0,618 (L = 3), 0,285 (L = 6) und 0,136 (L = 12); an der Grenze schwingt der Bestand mit der Periode 4L − 2.
Die Grenze ist keine gute Einstellung: der Spektralradius ist bei α\* = (1/L)((L − 1)/L)^(L−1) am kleinsten (ab L = 3 rund 0,23 bis 0,24 · α_krit; bei L = 1 ist es 0,5 · α_krit, bei L = 2 0,25 · α_krit), die kürzeste gemessene Rückkehr liegt knapp darüber, und bei 0,95 · α_krit dauert sie für L = 3 317 Perioden, für L = 12 1 479.
Alles ist **eigene Herleitung und Messung in diesem Modell**. Die Fachliteratur zu Bestandsregeln mit Verzögerung wurde nicht gelesen; dass Stabilitätsgrenze, Wurzeln auf dem Einheitskreis und die Euler-Deutung dort vermutlich bekannt sind, ist ungeprüft. Eine Neuheitsaussage gibt es nicht.

## Modell

**Zeitlage (gilt für jede Zahl):** i_(t+1) = i_t + u_(t+1−L) − e_t mit u_t = −α · i_t, also i_(t+1) = i_t − α · i_(t+1−L) − e_t. i ist der Bestand minus Soll, u die Bestellung minus Mittelbedarf, e die Bedarfsabweichung. Eine Bestellung der Periode t steht im Bestand der Periode t + L.
**L = 1 heißt „eine Periode Wirkverzug“, nicht „keine Lieferzeit“**; L = 0 gibt es in dieser Zählung nicht. Eine andere Zählung verschiebt L um eins und damit jede Zahl dieser README.

- Charakteristisches Polynom **P(z) = z^L − z^(L−1) + α**; der Stoß klingt ab, wenn der Spektralradius ρ (größter Wurzelbetrag) unter 1 liegt.
- Alle Zahlen: β = 0 (die Leitung wird nicht mitgerechnet), kein Meldeverzug (s = 0), keine Klemmung, feste Lieferzeit, Bedarfsstoß e₀ = 1 in Periode 0 (also i₁ = −1), Anfangszustand 0, Zähleinheit Periode. **Abklingzeit** heißt: letzte Periode t mit |i_t| über einer Schwelle (0,5 / 1 / 2 / 5 %, Voreinstellung 1 %) der Spitze max|i|. Die Schwelle ist eine Wahl, keine Naturgröße.
- Zweite Bedarfsform: dauerhaft verschobener Bedarf e (Umschalter bei Experiment 1).
- Die Formeln stehen im Expander 📐 der App. Die Demo hat keine Zufallsgröße, also keinen Zufalls-Seed; die Gegenproben mit Zufallspaaren (nur in `tools/make_frozen.py` und den Tests) benutzen einen ganzzahligen SplitMix64 (`llz_rng.py`).

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Stoßantwort** (`llz_dynamics.pulse`) | Rekursion i_(t+1) = i_t − α i_(t+1−L) − e_t in Gleitkomma, Horizont wächst, bis die Antwort sicher unter der Schwelle liegt (Obergrenze 40 000 Perioden; danach „noch nicht abgeklungen“, nie eine erfundene Zahl) | bricht ab, wenn \|i\| 10¹² übersteigt |
| **Spektralradius** (`spectral_radius`, `rho_grid`) | Eigenwerte der Zustandsmatrix (L × L), gestapelt über α | an der Doppelwurzel α\* verlieren Eigenwerte etwa 8 Stellen (Jordan-Block; gemessen bis 1,6·10⁻⁸): dort nur Bänder |
| **Grenze** (`alpha_krit`) | Formel 2 sin(π/(4L − 2)) aus Betrag und Phase einer Wurzel auf dem Kreis; Gegenprobe: Bisektion auf ρ = 1 aus den Eigenwerten | gilt als Stabilitätsgebiet (0, α_krit) nur gemessen, nicht bewiesen |
| **Kürzeste Rückkehr** (`fastest_return`) | Abklingzeit auf einem Gitter von 4000 α in (0, 0,999 · α_krit), vektorisiert über α | Lage und Wert hängen an Gitter und Schwelle; die Zeit ist keine glatte Funktion von α |
| **Summenregeln** (`pulse_sums`) | Summe der Antwort bis der Betrag unter 10⁻¹⁵ liegt | nur im stabilen Fall |

Die Live-Rechnung der App (jeder Abschnitt) braucht an Rechenzeit beim ersten Aufruf etwa 0,4 Sekunden (die Farbfläche ρ(L, α) in Experiment 2), danach je Eingabe höchstens etwa 0,1 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben; das Zeichnen kommt hinzu). **Vorgerechnet** ist die Zählung „0 Ausnahmen“ (`data/llz_frozen.json`, erzeugt von `tools/make_frozen.py`, einige Sekunden Laufzeit):
das Gitter mit 96 000 Punkten, die 20 000 Zufallspaare, die 5 000 Paare unterhalb α\*, das Monotonie-Gitter und die Gitter für den ersten Überschwinger und die erste Verstärkung. Die App kennzeichnet sie als „vorgerechnet“.
Keine Voreinstellung und kein Preset ist eine Zweierpotenz als α (Rechner-Artefakt bei Dualbrüchen; der Regler lässt sie zu); die benannten exakten Fälle α_krit(1) = 2, α_krit(2) = 1, α\*(1) = 1 und α\*(2) = 1/4 stehen nur in Tests, dort mit Brüchen und zusätzlich bei α (1 ± 10⁻³).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: in diesem Modell, β = 0, s = 0, ohne Klemmung, Zeitlage wie oben, Stoß e₀ = 1. Die Spalte „Test“ nennt die Datei und die Funktion in `tests/`, die genau diese Form prüft (Bänder dort, wo Plattformen abweichen können; „Messung“ heißt: Zahl dieser README aus `tools/make_frozen.py` oder aus der App, im Test nur auf Konsistenz oder als Band).

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | Formel gegen Eigenwerte | Die Formel α_krit = 2 sin(π/(4L − 2)) weicht von der Grenze aus den Eigenwerten (Bisektion auf ρ = 1) um höchstens 10⁻⁹ ab (L = 1 bis 40; gemessen 5·10⁻¹⁵). Das Residuum \|P(e^(iθ))\| bei θ = π/(2L − 1) ist höchstens 10⁻¹³ (gemessen 3,6·10⁻¹⁶). Die Periode aus dem Wurzelwinkel weicht in 0 von 39 Fällen (L = 2 bis 40) um mehr als 0,1 % von 4L − 2 ab. | `test_claims.py::test_b1_*`, `test_oracle_*::test_alpha_krit_*` |
| B2 | Stabilitätsgebiet = (0, α_krit)? | Gitter 8 000 α in (0, 4), L = 1 bis 12: 11 559 Punkte unterhalb, 84 441 oberhalb, **0 Ausnahmen** (ρ < 1 unterhalb, ρ > 1 oberhalb). Gegenprobe mit 20 000 Zufallspaaren (L gleichverteilt 1 bis 60, α gleichverteilt in (0, 4), Zufalls-Seed 15): 696 unterhalb, 19 304 oberhalb, 0 Ausnahmen. Für L > 60 nicht gemessen. | `test_b2_*` (Messung, vorgerechnet; kleine Fassung live) |
| B3 | Stetige Grenze | α_krit · L = 2,000 (L = 1, 2), 1,854 (3), 1,708 (6), 1,638 (12) gegen π/2 = 1,571: relativer Abstand +27,3 % (L = 1, 2), +18,0 % (3), +4,27 % (12), +0,501 % (100). Er gleicht 1/(2L − 1) (0,00503 bei L = 100); Steigung im Log-Log-Bild −1,00 (L = 100 bis 3 200). Dauerhaft innerhalb 1 % von π/2: α_krit · L ab L = 51, α_krit · (L − ½) ab L = 4; innerhalb 0,1 %: ab L = 501 bzw. 11 (geprüft bis L = 200 000). | `test_b3_*` |
| B4 | Euler-Bezug | α_krit(L) · (L − 1) ist die Zahl 2m sin(π/(4m + 2)) mit m = L − 1: 1,000 (L = 2), 1,236 (3), 1,423 (6), 1,495 (11), 1,563 (101). Aus den Eigenwerten gefunden, nicht aus der Formel. Dieselbe Rechnung, anders gelesen: das Lager erzeugt Befund B4 der `verzoegerung-hopf-demo` nicht neu. | `test_b4_*` |
| B5 | Doppelwurzel, kleinster Spektralradius | Bei α\* = (1/L)((L − 1)/L)^(L−1) hat P eine Doppelwurzel z\* = (L − 1)/L (P(z\*) = P′(z\*) = 0 exakt in Brüchen, L = 2 bis 12; α\* = 1/4, 4/27, 27/256, …). Das Gitterminimum von ρ(α) liegt bei α\* (auf zwei Gitterschritte) und hat den Wert (L − 1)/L (auf 10⁻³) in 11 von 11 Fällen (L = 2 bis 12, 3 000 Punkte). ρ fällt streng links und steigt streng rechts von α\* (2 × 2 000 Punkte je L, L = 2 bis 12; vorgerechnet). α\*/α_krit = 0,2500 (L = 2), 0,2397 (3), 0,2353 (6), 0,2345 (12), gegen 2/(eπ) = 0,2342 für große L. L = 1: α\* = 1, ρ = 0 (Totbeat). | `test_b5_*`, `test_oracle_*::test_double_root_*` |
| B6 | Abklingzeit gegen Schätzung | Die Schätzung ln 0,01/ln ρ trifft die gemessene Abklingzeit in **29 von 35** Zellen (L = 1, 2, 3, 6, 12 × α = 0,11 / 0,26 / 0,52 / 0,77 / 0,91 / 0,95 / 0,99 · α_krit, Schwelle 1 %) auf ±10 %; Verhältnis gemessen/Schätzung 0,954 bis 1,463, Mittel 1,055. In den sechs übrigen Zellen liegt die Messung 12 % bis 46 % darüber; fünf davon bei 0,26 · α_krit (nahe α\*, wo die Antwort t · ρ^t statt ρ^t folgt), die sechste bei L = 1, 0,52 · α_krit. | `test_b6_*` |
| B7 | Das ρ-Minimum ist nicht die kürzeste Rückkehr | Kleinste gemessene Zeit (Schwelle 1 %, 4 000 Punkte) gegen die Zeit bei α\*: L = 2: 7 gegen 11; L = 3: 11 gegen 18; L = 4: 16 gegen 26; L = 6: 25 gegen 40; L = 12: 53 gegen 84; L = 24: 109 gegen 172 (Verhältnis 0,61 bis 0,64). Der günstigste Punkt liegt bei 1,14 bis 1,20 · α\* (L = 2 bis 24), also bei 0,28 bis 0,30 · α_krit. Die Zeit ist keine glatte Funktion von α: L = 3, α = 0,1769: 11 Perioden, α = 0,1772: 15 (die Nachschwingung kreuzt die 1 %-Linie). Der Wert hängt an der Schwelle: 0,5 / 1 / 2 / 5 %: kleinste Zeit L = 3: 13 / 11 / 10 / 8, L = 12: 61 / 53 / 46 / 37. Gemessen ist auch, dass die kleinste Zeit nahe bei ln(Schwelle)/ln((L − 1)/L) liegt (L = 12: 60,9 / 52,9 / 45,0 / 34,4; L = 3: 13,1 / 11,4 / 9,6 / 7,4; in diesen acht Fällen weicht die gemessene Zeit um höchstens rund 8 % vom Formelwert ab). Ein Mechanismus ist nicht hergeleitet. | `test_b7_*` |
| B8 | Nahe der Grenze | Abklingzeit (Schwelle 1 %) bei 0,91 / 0,95 / 0,99 · α_krit: L = 1: 24 / 44 / 228; L = 3: 172 / 317 / 1 582; L = 6: 379 / 697 / 3 534; L = 12: 792 / 1 479 / 7 413. Beispiel L = 3, α = 0,6 (0,971 · α_krit): ρ = 0,99145, Abklingzeit 542, Schätzung 536,3; der Stoß wird erst verstärkt (i₇ = 26/25 von Hand), max\|i\| = 1,04. | `test_b8_*` |
| B9 | Schwingen und Verstärken | Der Bestand wird bei 0,9 · α\* nie positiv (kein Vorzeichenwechsel der Antwort, L = 1 bis 12, bis \|i\| < 10⁻⁹), bei 1,1 · α\* wechselt das Vorzeichen dreimal (L = 2 bis 12, bis \|i\| < 10⁻⁹), bei L = 1 neunmal; die Antwort schwingt weiter (komplexes Wurzelpaar), die Zahl hängt an der Schwelle, bei kleinerer Schwelle sind es mehr. 5 000 Zufallspaare (L = 2 bis 40, α zufällig in (0, 0,99 α\*), Zufalls-Seed 16): 0 mit Bestand über Soll (vorgerechnet). Erster Überschwinger bei 0,236 bis 0,252 · α_krit, **Verstärkung** (max\|i\| über 1) erst ab 0,948 bis 0,960 · α_krit (L = 3 bis 12, Gitter 20 000; L = 2 und L = 1 nie); max\|i\| = 1,052 bis 1,073 bei 0,99 · α_krit und 1,069 bis 1,095 bei 0,999 · α_krit (L = 3 bis 12). | `test_b9_*` |
| B10 | Summenregeln | Nach dem Stoß gilt Σ u = 1 und Σ i = −1/α, unabhängig von L (Herleitung: Rekursion über alle t summieren, i₀ = i_∞ = 0). Gemessen in 36 Fällen (L = 1 bis 12, α = 0,21 / 0,47 / 0,83 · α_krit): größte Abweichung unter 10⁻⁹ (gemessen 7,8·10⁻¹⁶). Fehlbestandsfläche an der Grenze 1/α_krit: 0,500 (L = 1), 1,000 (2), 1,618 (3, Goldener Schnitt), 3,513 (6), 7,327 (12). Bei dauerhaft verschobenem Bedarf e sitzt der Bestand bei −e/α, unabhängig von L. | `test_b10_*` |
| B11 | Zwei Rechenwege | Rekursion gegen Zustandsmatrixpotenz und gegen `scipy.signal.lfilter` mit der Übertragungsfunktion: 60 Fälle (L = 1 bis 12, 5 α, 300 Perioden), Abweichung höchstens 10⁻¹² (Messung 8,3·10⁻¹⁷). | `test_oracle_*::test_pulse_matches_*` |

### Befunde und Korrekturen gegenüber dem Plan

- **Gleiche Aussagen, andere Zählungen.** Der Plan nannte für B2 11 556 / 84 442 Gitterpunkte und 676 / 19 324 Zufallspaare. Im Bau (anderes Gitter, SplitMix64 statt `numpy.random`, Zufalls-Seed 15) sind es 11 559 / 84 441 und 696 / 19 304; die Aussage „0 Ausnahmen“ bleibt. Vormessungszahlen sind Erwartungen; hier steht die Messung des Baus.
- **Preset „Schnelle Rückkehr“.** Die Vormessung (`out_rk_v15_gegenprobe_presets.txt`) listete α = 0,177 mit Abklingzeit 15; der Plan nannte α = 0,175 mit Zeit 12. Die App benutzt α = 0,175 (12 Perioden, Überschwinger 0,0082 unter der 1 %-Linie); α = 0,1769 gibt 11 und 0,1772 gibt 15 (B7).
- **Doppelwurzelpunkt.** Eigenwerte am Punkt α\* sind auf etwa 10⁻⁸ genau (Beispiel L = 3: ρ = 0,66666667 statt 2/3); Tests und Anzeige nehmen dort Bänder. Das Preset „Doppelwurzel“ benutzt α = 0,148 (nahe bei α\* = 0,1481, nicht auf dem Punkt).
- **Zeitlage der Gliederung.** Die Gliederung der Linie sagt bei V1 „ohne Lieferzeit (L = 1)“; richtig ist „eine Periode Wirkverzug“ (siehe Modell). Die Demo nennt L = 1 nie „keine Lieferzeit“.
- **α_krit · L fällt nicht streng.** α_krit(1) · 1 = α_krit(2) · 2 = 2 exakt; eine Rundungsdifferenz im Rechner ist kein Gefälle. Der Test prüft die strenge Abnahme nur für α_krit und für α_krit · L ab L = 2.

## Quellen

Keine Quelle ist im Wortlaut gelesen. Die Stabilitätsgrenze, die Wurzelrechnung auf dem Einheitskreis, die Doppelwurzel und die Summenregeln sind eigene Herleitung in diesem Modell; dass sie in der Fachliteratur zu Bestandsregeln mit Verzögerung bekannt sind, ist wahrscheinlich, aber ungeprüft.
Es gibt deshalb kein „nach Autor X“ und keine Neuheitsaussage. Der eigene Anteil sind die Zusammenführung in einem Modell, das Zahlenbild, die exakten Gegenproben (Brüche, zwei unabhängige Rechenwege) und die Befunde B6 bis B9 zur Abklingzeit. Quellenbelege zu den Grundlagen nachzutragen ist offen.

## Ehrliche Grenzen

- **Nur dieses Modell.** β = 0, s = 0, ohne Klemmung, feste Lieferzeit; die Stücke der Linie ändern je genau eine dieser Größen. Jede Zahl gilt nur hier; eine andere Zählung der Lieferzeit verschiebt L um eins.
- **Das Stabilitätsgebiet (0, α_krit) ist gemessen, nicht bewiesen** (B2: 96 000 Gitterpunkte für L ≤ 12 und 20 000 Zufallspaare für L ≤ 60; für L > 60 nichts). Ebenso ist die strenge Monotonie von ρ(α) links und rechts von α\* nur im Gitter gemessen (L = 2 bis 12).
- **Die Abklingzeit ist eine Wahl der Schwelle.** Die kleinste Zeit und ihre Lage hängen an Schwelle und Gitter (B7); die Zeit ist keine glatte Funktion von α. Die Schätzung ln(Schwelle)/ln ρ trifft nicht überall (B6). Ein Mechanismus für „kleinste Zeit ≈ ln(Schwelle)/ln((L − 1)/L)“ ist nicht hergeleitet: es ist eine Messung für L = 2 bis 24 und vier Schwellen. Dass die Rückkehr bei ρ-Minimum nicht die kürzeste ist, deutet auf das schnellste Abklingen ohne Nachschwinger über der Schwelle; das ist eine Deutung, nicht gemessen.
- **Nur Stoß und dauerhafte Verschiebung.** Saison, Rampe und Rauschen sind nicht gemessen (Stück 5 der Linie). Die Summenregeln sind der Frequenzgang bei der Frequenz 0.
- **Rechner.** Eigenwerte am Doppelwurzelpunkt: Bänder (siehe Methodik). Wer α in der App auf eine Zweierpotenz stellt (1/4, 1, 2), rechnet exakt in Dualbrüchen; an der Grenze (L = 1, α = 2; L = 2, α = 1) liegt ρ = 1 bis auf Rundung, die App meldet das als „an der Grenze“ (Toleranz 10⁻⁹).
- **Keine Echtdaten.** Das Lager ist ein Rechenmodell, kein gemessener Betrieb.
- **Browser.** Die fünf Diagramme und die Formeln wurden im Browser-Fenster der Entwicklungsumgebung (ein Browser, eine Fenstergröße) angesehen, nicht auf Telefon und nicht in beiden Designs; AppTest sieht Diagramme nicht.

## Verifikation

- **Kern:** Handbeispiele in Brüchen (`tests/test_dynamics.py`: L = 1 mit α = 1/5, 6/5, 9/5; L = 2 mit α = 1 und 1/4; L = 3 mit α = 3/5 und α\* = 4/27; α_krit(3) mit α² + α − 1 = 0), Randfälle (`test_edge_cases.py`: L = 1, genau L = 2, L = 40, α an den Reglergrenzen, alle Schwellen).
- **Orakel** (`test_oracle_lager_lieferzeit.py`): α_krit gegen Eigenwerte und `numpy.roots`; Stoßantwort gegen `numpy.linalg.matrix_power` und `scipy.signal.lfilter`; Abklingzeit gegen geschlossene Formeln; Doppelwurzel in Brüchen; ρ-Minimum gegen Gitter; Stabilitätsgebiet gegen Gitter und Zufallspaare; Summenregeln.
- **Eigenschaften** (`test_eigenschaften_lager_lieferzeit.py`): Linearität, Überlagerung, Zeitinvarianz, Spiegelung, Summenregeln und Dauerverschiebung für Zufallspaare, α → 0, α_krit streng fallend, ρ links fallend und rechts steigend.
- **Claims** (`test_claims.py`): jede Zahl der Befunde B1 bis B11 mit ihren Voraussetzungen, die vorgerechneten Zählungen auf Konsistenz und mit kleinen Live-Fassungen.
- **Verdikte** (`test_verdicts.py`): je Preset die erwartete Zeile, je Zweig eine handgesetzte Analyse; ein Preset ohne Effekt meldet keinen Effekt.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Regleränderung hebt die Markierung auf), Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfrage, fünf Experimente, Grenzen, Formeln |
| `data/llz_frozen.json` | vorgerechnete Zählungen (B2, B5, B9) |
| `llz_constants.py` | Regler-Grenzen, Farben, Presets |
| `llz_dynamics.py` | Kern: Formeln, Zustandsmatrix, Stoßantwort, Abklingzeit, Summen, stetige Tabelle |
| `llz_evaluation.py` | Auswertungen der Abschnitte, Verdikte (reine Funktionen), Tabellen |
| `llz_layout.py` | Fester Kopf (Stylesheet) |
| `llz_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart, aktives Preset hervorgehoben) |
| `llz_rng.py` | SplitMix64 für die Gegenproben (Kopie aus `lager-chaos-demo`, mit Präfix umbenannt) |
| `llz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenprobe) |
| `tools/make_frozen.py` | erzeugt `data/llz_frozen.json` |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Claims, Verdikte, Presets, Layout, AppTest-Smoke, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Leitungsgewicht, Meldeverzug, streuende Lieferzeit, Saison und Rauschen:** eigene Stücke der Linie (siehe oben); hier β = 0, s = 0, feste Lieferzeit, Stoß und Dauerverschiebung.
- **Klemmung, Grenzzyklus, Chaos:** nur als Verweis auf [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo), kein Nachbau.
- **Zeitschritt-Regler und Hopf-Gleichung** der stetigen Seite: gehören zur [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo); Experiment 5 zeigt nur die Brücke als Kurve und Tabelle.
- **Reglerentwurf und Parameterschätzung aus Daten:** nicht Gegenstand der Linie.
- **PDF-Export des Ergebnisses:** nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Rückkopplung und Verzögerung: das Lager](https://sebastianhanisch.net/konzepte-rueckkopplung.html).
