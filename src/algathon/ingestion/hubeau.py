"""API Hub'Eau (https://hubeau.eaufrance.fr), sans compte ni clé.

- Hydrométrie, données élaborées : débit moyen journalier (QmnJ), en l/s dans l'API,
  converti ici en m³/s.
- Qualité des cours d'eau : analyses physico-chimiques, paramètre Sandre 1340
  (nitrates), en mg(NO3)/L. Environ 15 à 25 prélèvements par an selon la station.
"""
from __future__ import annotations

import pandas as pd

from .cache import DOSSIER_RAW, get_json, json_en_cache

URL_HYDRO = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
URL_QUALITE = "https://hubeau.eaufrance.fr/api/v2/qualite_rivieres/analyse_pc"
CODE_NITRATES = "1340"
TAILLE_PAGE = 20000  # maximum accepté par Hub'Eau
UNITES_NO3 = {"mg(NO3)/L", "mg/L"}


def _toutes_les_pages(url: str, params: dict) -> list:
    """Suit les liens `next` de Hub'Eau et concatène les résultats."""
    page = get_json(url, params)
    resultats = list(page.get("data", []))
    while page.get("next"):
        page = get_json(page["next"])
        resultats.extend(page.get("data", []))
    return resultats


def debits_journaliers(code_station: str, debut: str = "2001-01-01",
                       force: bool = False) -> pd.DataFrame:
    """Débit moyen journalier d'une station. Colonnes : date, debit_m3s.

    Le cache contient tout l'historique depuis `debut` jusqu'au jour du téléchargement.
    """
    params = {
        "code_entite": code_station,
        "grandeur_hydro_elab": "QmnJ",
        "date_debut_obs_elab": debut,
        "fields": "date_obs_elab,resultat_obs_elab",
        "size": TAILLE_PAGE,
    }
    chemin = DOSSIER_RAW / "hubeau" / "debits" / ("%s_QmnJ.json" % code_station)
    brut = json_en_cache(chemin, lambda: _toutes_les_pages(URL_HYDRO, params), force=force)
    df = pd.DataFrame(brut, columns=["date_obs_elab", "resultat_obs_elab"])
    df = df.rename(columns={"date_obs_elab": "date", "resultat_obs_elab": "debit_ls"})
    df["date"] = pd.to_datetime(df["date"])
    df["debit_m3s"] = pd.to_numeric(df["debit_ls"], errors="coerce") / 1000.0
    return (df.dropna(subset=["debit_m3s"])
              .drop_duplicates("date")
              .sort_values("date")[["date", "debit_m3s"]]
              .reset_index(drop=True))


def nitrates(code_station: str, debut: str = "2001-01-01",
             force: bool = False) -> pd.DataFrame:
    """Analyses de nitrates d'une station. Colonnes : date, no3_mgL.

    Plusieurs analyses le même jour sont moyennées.
    """
    params = {
        "code_station": code_station,
        "code_parametre": CODE_NITRATES,
        "date_debut_prelevement": debut,
        "fields": "date_prelevement,resultat,symbole_unite",
        "size": TAILLE_PAGE,
        "sort": "asc",
    }
    chemin = DOSSIER_RAW / "hubeau" / "nitrates" / ("%s_NO3.json" % code_station)
    brut = json_en_cache(chemin, lambda: _toutes_les_pages(URL_QUALITE, params), force=force)
    df = pd.DataFrame(brut, columns=["date_prelevement", "resultat", "symbole_unite"])
    if df.empty:
        return pd.DataFrame(columns=["date", "no3_mgL"])
    # Le paramètre 1340 est exprimé en NO3. Certains laboratoires notent simplement « mg/L »
    # (même ordre de grandeur vérifié sur 04168140) ; toute autre unité, p. ex. mg(N)/L, est refusée.
    unites = set(df["symbole_unite"].dropna())
    if unites - UNITES_NO3:
        raise ValueError("Unité inattendue pour %s : %s" % (code_station, unites))
    df["date"] = pd.to_datetime(df["date_prelevement"].str[:10])
    df["no3_mgL"] = pd.to_numeric(df["resultat"], errors="coerce")
    df = df.dropna(subset=["no3_mgL"])
    return (df.groupby("date", as_index=False)["no3_mgL"].mean()
              .sort_values("date").reset_index(drop=True))
