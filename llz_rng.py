"""Ganzzahliger Zufallsgenerator (SplitMix64) für die Gegenproben mit festem Zufalls-Seed (tools/make_frozen.py und tests/).

Herkunft: Kopie der Teile `SplitMix64`, `splitmix_array` und `uniform_array` aus `lgc_rng.py` der `lager-chaos-demo` (dort Eigenständigkeit nach
demselben Muster), mit Präfix umbenannt; keine Normalverteilung, weil diese Demo kein Rauschen hat. Die App selbst braucht keinen Zufall:
sie ist deterministisch und hat keinen Zufalls-Seed. Python-`hash()` und `random` sind für feste Zufalls-Seeds nicht geeignet (pro Prozess gesalzen bzw.
nicht versionsstabil)."""

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / float(1 << 53)


def splitmix_array(seed: int, n: int) -> np.ndarray:
    """Die ersten n Ausgaben von SplitMix64(seed) als uint64-Feld (vektorisiert, gleiche Folge wie die Klasse)."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK) + k * np.uint64(GAMMA)          # uint64-Felder rechnen modulo 2^64
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform_array(seed: int, n: int) -> np.ndarray:
    """n Gleichverteilte in [0, 1) mit 53 Bit."""
    return (splitmix_array(seed, n) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
