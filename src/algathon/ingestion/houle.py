"""Houle au large de chaque baie : réanalyse ERA5 des vagues via Open-Meteo Marine (sans clé).

Pourquoi : un hiver agité disperse le stock d'ulves qui a passé l'hiver dans la baie, et la
saison suivante démarre plus tard. C'est le premier facteur cité par le CEVA dans ses bulletins
(jours de houle de plus de 2,5 m, 3,5 m et 4,5 m de novembre à mars).

La grille ERA5 des vagues fait 0,5° : le point demandé est ramené à la maille de mer la plus
proche (Saint-Brieuc et la Fresnaye partagent ainsi la même maille).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .cache import DOSSIER_RAW, get_json, json_en_cache

URL_MARINE = "https://marine-api.open-meteo.com/v1/marine"
SEUILS_M = (2.5, 3.5, 4.5)  # seuils de houle utilisés par le CEVA


def houle_journaliere(lat: float, lon: float, debut: str = "2001-01-01",
                      fin: str = None, force: bool = False) -> pd.DataFrame:
    """Colonnes : date, houle_max_m (hauteur significative maximale du jour), periode_max_s."""
    if fin is None:
        fin = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    params = {"latitude": lat, "longitude": lon, "start_date": debut, "end_date": fin,
              "daily": "wave_height_max,wave_period_max", "models": "era5_ocean"}
    chemin = DOSSIER_RAW / "houle" / ("era5_%.2f_%.2f.json" % (lat, lon))
    brut = json_en_cache(chemin, lambda: get_json(URL_MARINE, params), force=force)
    j = brut["daily"]
    return pd.DataFrame({"date": pd.to_datetime(j["time"]),
                         "houle_max_m": pd.to_numeric(pd.Series(j["wave_height_max"]), errors="coerce"),
                         "periode_max_s": pd.to_numeric(pd.Series(j["wave_period_max"]), errors="coerce")})


def indicateurs_hiver(houle: pd.DataFrame, annees) -> pd.DataFrame:
    """Pour chaque année Y, l'hiver de novembre Y-1 à mars Y (connu avant la saison).

    houle_hiver_moy : moyenne de la houle maximale journalière (m)
    jours_houle_XX  : nombre de jours où la houle maximale dépasse XX m
    """
    lignes = []
    h = houle.set_index("date")["houle_max_m"]
    for y in annees:
        hiver = h.loc["%d-11-01" % (y - 1):"%d-03-31" % y].dropna()
        if len(hiver) < 120:  # hiver incomplet
            continue
        ligne = {"annee": y, "houle_hiver_moy": hiver.mean()}
        for s in SEUILS_M:
            ligne["jours_houle_%02d" % round(s * 10)] = int((hiver > s).sum())
        lignes.append(ligne)
    return pd.DataFrame(lignes)
