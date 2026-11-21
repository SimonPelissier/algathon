import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

calibration = pytest.importorskip("algathon.detection.calibration")


def test_decalage_baseline_04():
    s2 = pytest.importorskip("algathon.detection.sentinel2")  # demande rasterio
    dn = np.array([1500], dtype="uint16")
    assert np.isclose(s2.reflectance(dn, "02.12")[0], 0.15)
    assert np.isclose(s2.reflectance(dn, "04.00")[0], 0.05)   # décalage +1000 retiré
    assert np.isclose(s2.reflectance(dn, "05.10")[0], 0.05)


def _scene(date, valid, eau, ulva, zone=100.0, baie="b"):
    return {"baie": baie, "date": date, "zone_ha": zone, "valid_ha": valid,
            "eau_ha": eau, "ulva_ha_s020": ulva}


def test_selection_garde_la_plus_basse_mer_parmi_les_scenes_claires():
    s = pd.DataFrame([
        _scene("2020-07-02", 99, 90, 5),    # claire, marée haute
        _scene("2020-07-10", 98, 40, 30),   # claire, marée basse : gardée
        _scene("2020-07-20", 50, 0, 80),    # trop nuageuse : écartée
    ])
    sel = calibration.selection_basse_mer(s, part_valide_min=0.9)
    assert len(sel) == 1 and sel.loc[0, "date"] == "2020-07-10"
    assert sel.loc[0, "n_scenes_claires"] == 2


def test_cumul_inconnu_si_un_mois_manque():
    sel = pd.DataFrame({"baie": "b", "annee": [2020, 2020, 2020, 2021, 2021],
                        "mois": [5, 7, 9, 5, 7], "ulva_ha_s020": [1.0, 2.0, 3.0, 1.0, 1.0]})
    c = calibration.cumul_mois(sel, [5, 7, 9]).set_index("annee")
    assert c.loc[2020, "ulva_ha_s020"] == 6.0
    assert np.isnan(c.loc[2021, "ulva_ha_s020"])
