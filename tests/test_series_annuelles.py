import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.baseline_annuel import plis_glissants, plis_loyo  # noqa: E402
from algathon.series_annuelles import (  # noqa: E402
    agreger_saison, concentration_journaliere, serie_riviere,
)


def test_flux_unites():
    # 1 m³/s à 62 mg NO3/L pendant un jour = 62 g/m³ × 86 400 m³ × 14/62 = 1 209,6 kg N
    jours = pd.date_range("2020-05-01", "2020-05-03")
    q = pd.DataFrame({"date": jours, "debit_m3s": 1.0})
    c = pd.DataFrame({"date": jours, "no3_mgL": 62.0})
    s = serie_riviere(q, [c])
    assert np.allclose(s["flux_kgN"], 1209.6)


def test_interpolation_bornee():
    prel = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-11", "2020-06-01"]),
                         "no3_mgL": [10.0, 20.0, 30.0]})
    jours = pd.date_range("2020-01-01", "2020-06-01")
    c = concentration_journaliere(prel, jours, ecart_max=62)
    assert c["2020-01-06"] == 15.0              # interpolé entre deux prélèvements proches
    assert np.isnan(c["2020-03-01"])            # trou de 142 jours : pas d'interpolation
    assert c["2020-06-01"] == 30.0              # jour de prélèvement


def test_priorite_des_stations():
    jours = pd.date_range("2020-01-01", "2020-01-05")
    q = pd.DataFrame({"date": jours, "debit_m3s": 1.0})
    a = pd.DataFrame({"date": jours[:2], "no3_mgL": [10.0, 10.0]})
    b = pd.DataFrame({"date": jours, "no3_mgL": 50.0})
    s = serie_riviere(q, [a, b])
    assert list(s["no3_mgL"]) == [10.0, 10.0, 50.0, 50.0, 50.0]


def test_saison_couverture_insuffisante():
    jours = pd.date_range("2020-05-01", "2020-08-31")
    j = pd.DataFrame({"debit_m3s": 1.0, "no3_mgL": np.nan, "flux_kgN": np.nan}, index=jours)
    a = agreger_saison(j, [5, 6, 7, 8])
    assert a.loc[0, "debit"] == 1.0
    assert np.isnan(a.loc[0, "flux_t"])         # aucun jour de nitrates connu


def test_plis_sans_fuite():
    annees = list(range(2002, 2023))
    for train, test in plis_loyo(annees):
        assert test not in train and test + 1 not in train   # N+1 porte la cible de N
    for train, test in plis_glissants(annees):
        assert max(train) < test


def test_modeles_par_defaut_et_validation_glissante():
    from algathon.baseline_annuel import evaluer, fabriquer_modeles
    assert {"ridge", "random_forest", "extra_trees"} <= set(fabriquer_modeles())
    rng = np.random.default_rng(0)
    lignes = []
    for b in ("a", "b"):
        for an in range(2002, 2016):
            lignes.append({"baie": b, "annee": an, "surface_ha": rng.uniform(10, 100),
                           "surface_n_1": rng.uniform(10, 100), "debit_jan_avr": 1.0})
    pred = evaluer(pd.DataFrame(lignes), ["surface_n_1", "debit_jan_avr"])
    assert {"pred_random_forest", "pred_extra_trees"} <= set(pred.columns)
    assert pred["annee"].min() == 2010          # 8 années d'apprentissage minimum
    assert (pred["pred_extra_trees"] >= 0).all()
