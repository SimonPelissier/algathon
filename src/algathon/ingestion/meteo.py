"""Météo historique journalière via l'API Archive d'Open-Meteo (réanalyse ERA5), sans clé.

Choisi plutôt que le Copernicus CDS, qui demande un compte et une clé API.
Les données récentes arrivent avec environ 5 jours de retard.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .cache import DOSSIER_RAW, get_json, json_en_cache

URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m_mean", "shortwave_radiation_sum", "precipitation_sum"]


def meteo_journaliere(lat: float, lon: float, debut: str = "2001-01-01",
                      fin: str = None, force: bool = False) -> pd.DataFrame:
    """Colonnes : date, temp_c (moyenne), ray_mj_m2 (rayonnement global), pluie_mm.

    Par défaut, `fin` est la date du jour moins 7 jours. Le cache est indexé par point :
    relancer avec force=True pour prolonger la série.
    """
    if fin is None:
        fin = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": debut,
        "end_date": fin,
        "daily": ",".join(VARIABLES),
        "timezone": "Europe/Paris",
    }
    chemin = DOSSIER_RAW / "meteo" / ("openmeteo_%.3f_%.3f.json" % (lat, lon))
    brut = json_en_cache(chemin, lambda: get_json(URL_ARCHIVE, params), force=force)
    j = brut["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(j["time"]),
        "temp_c": j["temperature_2m_mean"],
        "ray_mj_m2": j["shortwave_radiation_sum"],
        "pluie_mm": j["precipitation_sum"],
    })
