import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algathon.indices import fai, ndvi  # noqa: E402


def test_ndvi_range():
    v = ndvi(np.array([0.1, 0.5]), np.array([0.5, 0.1]))
    assert np.all((v >= -1) & (v <= 1)) and v[0] > 0 > v[1]


def test_fai_positive_for_floating_algae():
    # pic NIR au-dessus de la ligne de base rouge-SWIR
    assert fai(0.02, 0.10, 0.01) > 0
    assert fai(0.02, 0.015, 0.01) < 0.01
