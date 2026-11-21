import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.prevision_5j_app import facteurs_ceva, scores_par_baie  # noqa: E402


def test_scores_par_baie_et_total():
    ev = pd.DataFrame({"baie": ["a", "a", "b"], "observe_ha": [10.0, 20.0, 5.0],
                       "prevu_ha": [10.0, 40.0, 5.0], "persistance_ha": [20.0, 20.0, 50.0]})
    sc = scores_par_baie(ev).set_index("baie")
    assert sc.loc["toutes", "previsions"] == 3
    assert sc.loc["a", "part_a_50pct"] == 50.0          # 40 pour 20 : hors de ±50 %
    assert sc.loc["b", "gain_vs_persistance_pct"] == 100.0
    assert np.isclose(sc.loc["a", "erreur_moyenne_ha"], 10.0)


def test_facteur_ceva_sur_annees_communes():
    cu = pd.DataFrame({"baie": ["a", "a", "a"], "cumul_prevu": [10.0, 30.0, 50.0],
                       "surface_ha": [20.0, 60.0, np.nan]})
    assert facteurs_ceva(cu)["a"] == 2.0                # l'année sans CEVA est ignorée
