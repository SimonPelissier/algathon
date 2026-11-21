import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.ingestion.houle import indicateurs_hiver  # noqa: E402


def test_hiver_de_novembre_a_mars():
    jours = pd.date_range("2018-10-01", "2019-05-31")
    h = np.where(jours.month.isin([11, 12, 1, 2, 3]), 3.0, 9.0)   # 9 m hors hiver : ne doit pas compter
    h[(jours >= "2019-01-10") & (jours < "2019-01-20")] = 5.0        # 10 jours de très forte houle
    ind = indicateurs_hiver(pd.DataFrame({"date": jours, "houle_max_m": h}), [2019]).iloc[0]
    assert ind["annee"] == 2019
    assert ind["jours_houle_25"] == 151                              # tout l'hiver (nov.-mars 2018-2019)
    assert ind["jours_houle_45"] == 10
    assert 3.0 < ind["houle_hiver_moy"] < 3.3


def test_hiver_incomplet_ignore():
    jours = pd.date_range("2019-01-01", "2019-03-31")                # novembre-décembre manquants
    ind = indicateurs_hiver(pd.DataFrame({"date": jours, "houle_max_m": 2.0}), [2019])
    assert ind.empty
