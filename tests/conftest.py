"""Gemeinsame Fixtures und ein exakter Hilfsrechner für die Handbeispiele (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def exact_pulse(L: int, alpha: Fraction, T: int) -> list:
    """Stoßantwort i_0 ... i_T in Brüchen, unabhängig vom Kern geschrieben: i_{t+1} = i_t − α i_{t+1−L} − e_t mit e_0 = 1, i_t = 0 für t <= 0."""
    i = [Fraction(0)] * (T + 1)
    for t in range(T):
        prev = i[t + 1 - L] if t + 1 - L >= 0 else Fraction(0)
        i[t + 1] = i[t] - alpha * prev - (1 if t == 0 else 0)
    return i


@pytest.fixture
def hand():
    """Das Handbeispiel L = 3, α = 3/5 (knapp unter der Grenze 0,618): i_1 … i_7 von Hand gerechnet.
    i_1 = i_2 = i_3 = −1; i_4 = −1 + 3/5 = −2/5; i_5 = −2/5 + 3/5 = 1/5; i_6 = 1/5 + 3/5 = 4/5; i_7 = 4/5 + (3/5)(2/5) = 26/25."""
    return [Fraction(-1), Fraction(-1), Fraction(-1), Fraction(-2, 5), Fraction(1, 5), Fraction(4, 5), Fraction(26, 25)]
