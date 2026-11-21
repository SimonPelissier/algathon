"""Assemblage de la table annuelle « une ligne par baie et par année ».

Cible : surface cumulée d'échouage d'ulves (CEVA) de la baie, en ha.
Variables explicatives, calculées pour chaque baie :
  - surface_n_1           surface de l'année précédente (stock hérité)
  - débit, nitrates et flux d'azote des rivières, sur deux fenêtres :
      *_jan_avr   connu au 1er mai : utilisable pour une vraie prévision de saison
      *_mai_aout  pendant la saison : explique la saison, mais n'est pas connu à l'avance
  - météo (rayonnement, température, pluie) en mars-avril et en mars-mai

Flux d'azote (t N) = somme sur les jours de Q (m³/s) × [NO3] (mg/L = g/m³) × 86 400 s
                     × 14/62 (passage de NO3 à N) / 1e6 (g → t).
La concentration journalière est interpolée linéairement entre deux prélèvements
s'ils sont distants de moins de `ECART_MAX_JOURS` ; sinon le jour reste inconnu.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .ingestion import ceva, hubeau, meteo

RACINE = Path(__file__).resolve().parents[2]
ECART_MAX_JOURS = 62          # au-delà, on n'interpole pas entre deux prélèvements
COUVERTURE_MIN = 0.8          # part minimale de jours connus pour calculer un agrégat de saison
PART_MANQUANTE_MAX = 0.10     # part maximale (en moyenne historique) des sites ou communes absents
NO3_VERS_N = 14.0 / 62.0


def charger_config(chemin=None) -> dict:
    chemin = Path(chemin) if chemin else RACINE / "configs" / "baies.yaml"
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- nitrates et flux

def concentration_journaliere(prelevements: pd.DataFrame, jours: pd.DatetimeIndex,
                              ecart_max: int = ECART_MAX_JOURS) -> pd.Series:
    """Interpole [NO3] sur `jours` à partir des prélèvements d'une station.

    Un jour n'est renseigné que s'il est encadré par deux prélèvements séparés
    de `ecart_max` jours au plus (ou s'il est lui-même un jour de prélèvement).
    """
    if prelevements.empty:
        return pd.Series(np.nan, index=jours)
    s = prelevements.set_index("date")["no3_mgL"].sort_index()
    s = s[~s.index.duplicated()]
    dates = s.index.values
    pos = np.searchsorted(dates, jours.values)          # indice du prélèvement suivant
    avant = np.clip(pos - 1, 0, len(dates) - 1)
    apres = np.clip(pos, 0, len(dates) - 1)
    exact = (pos < len(dates)) & (dates[apres] == jours.values)
    ecart = (dates[apres] - dates[avant]) / np.timedelta64(1, "D")
    encadre = (pos > 0) & (pos < len(dates)) & (ecart <= ecart_max)
    union = s.reindex(s.index.union(jours)).interpolate(method="time", limit_area="inside")
    valeurs = union.reindex(jours).values
    return pd.Series(np.where(exact | encadre, valeurs, np.nan), index=jours)


def serie_riviere(debits: pd.DataFrame, prelevements: List[pd.DataFrame]) -> pd.DataFrame:
    """Série journalière d'une rivière : debit_m3s, no3_mgL, flux_kgN.

    Les stations de nitrates sont combinées par priorité : la première station
    est utilisée quand elle est disponible, les suivantes comblent ses trous.
    """
    if debits.empty:
        return pd.DataFrame(columns=["debit_m3s", "no3_mgL", "flux_kgN"])
    jours = pd.date_range(debits["date"].min(), debits["date"].max(), freq="D")
    q = debits.set_index("date")["debit_m3s"].reindex(jours)
    c = pd.Series(np.nan, index=jours)
    for p in prelevements:
        c = c.combine_first(concentration_journaliere(p, jours))
    flux_kg = q * c * 86400.0 * NO3_VERS_N / 1000.0  # g/j → kg N/j
    return pd.DataFrame({"debit_m3s": q, "no3_mgL": c, "flux_kgN": flux_kg})


def agreger_saison(journalier: pd.DataFrame, mois: List[int],
                   couverture_min: float = COUVERTURE_MIN) -> pd.DataFrame:
    """Agrège une série journalière de rivière par année sur les mois donnés.

    Renvoie, par année : debit (m³/s moyen), no3 (mg/L pondéré par le débit),
    flux_t (t N). Une valeur est NaN si moins de `couverture_min` des jours sont connus.
    """
    d = journalier[journalier.index.month.isin(mois)].copy()
    d["annee"] = d.index.year
    lignes = []
    for annee, g in d.groupby("annee"):
        n = len(pd.date_range("%d-01-01" % annee, "%d-12-31" % annee).to_series()
                 .loc[lambda x: x.dt.month.isin(mois)])
        cov_q = g["debit_m3s"].notna().sum() / n
        cov_f = g["flux_kgN"].notna().sum() / n
        connus = g.dropna(subset=["flux_kgN"])
        lignes.append({
            "annee": annee,
            "n_jours": n,
            "debit": g["debit_m3s"].mean() if cov_q >= couverture_min else np.nan,
            # flux moyen journalier × nombre de jours : corrige les quelques jours manquants
            "flux_t": connus["flux_kgN"].mean() * n / 1000.0 if cov_f >= couverture_min else np.nan,
            "no3": ((connus["debit_m3s"] * connus["no3_mgL"]).sum() / connus["debit_m3s"].sum()
                    if cov_f >= couverture_min and connus["debit_m3s"].sum() > 0 else np.nan),
        })
    return pd.DataFrame(lignes, columns=["annee", "n_jours", "debit", "flux_t", "no3"])


# ---------------------------------------------------------------- cibles CEVA

def somme_avec_tolerance(df: pd.DataFrame, cle: str, valeur: str, poids: pd.Series,
                         part_max: float = PART_MANQUANTE_MAX) -> pd.DataFrame:
    """Somme `valeur` par année sur les entités `cle`.

    L'année vaut NaN si les entités manquantes représentent plus de `part_max`
    du total historique (`poids` = moyenne historique de chaque entité).
    """
    total = poids.sum()
    lignes = []
    for annee, g in df.groupby("annee"):
        manquants = g.loc[g[valeur].isna(), cle]
        part = poids.reindex(manquants).fillna(0).sum() / total if total > 0 else 0.0
        somme = g[valeur].sum(min_count=1)
        lignes.append({"annee": annee,
                       valeur: somme if part <= part_max else np.nan,
                       "n_" + cle + "_manquants": len(manquants)})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- table complète

def table_annuelle(config: dict = None, force: bool = False) -> pd.DataFrame:
    """Construit la table baie × année (utilise le cache data/raw/)."""
    config = config or charger_config()
    debut = config["periode"]["debut_donnees"]
    saisons = config["saisons"]
    surfaces = ceva.surfaces_sites(force=force)
    volumes = ceva.volumes_communes(force=force)

    tables = []
    for code, baie in config["baies"].items():
        # cible : surface CEVA
        s = surfaces[surfaces["id_site"].isin(baie["sites_ceva"])]
        poids_s = s.groupby("id_site")["surface_ha"].mean()
        t = somme_avec_tolerance(s, "id_site", "surface_ha", poids_s)

        # volumes ramassés par les communes
        v = volumes[volumes["insee"].isin(baie["communes_insee"])]
        poids_v = v.groupby("insee")["volume_m3"].mean()
        tv = somme_avec_tolerance(v, "insee", "volume_m3", poids_v)
        t = t.merge(tv[["annee", "volume_m3"]], on="annee", how="outer")

        # rivières : on additionne débits et flux ; une rivière inconnue rend la baie inconnue
        agregats: Dict[str, List[pd.DataFrame]] = {"jan_avr": [], "mai_aout": []}
        for riv in baie["rivieres"]:
            q = hubeau.debits_journaliers(riv["debit"], debut=debut, force=force)
            p = [hubeau.nitrates(c, debut=debut, force=force) for c in riv["nitrates"]]
            jour = serie_riviere(q, p)
            agregats["jan_avr"].append(agreger_saison(jour, saisons["prevision_jan_avr"]))
            agregats["mai_aout"].append(agreger_saison(jour, saisons["diagnostic_mai_aout"]))
        for suffixe, liste in agregats.items():
            t = t.merge(_combiner_rivieres(liste, suffixe), on="annee", how="left")

        # météo au point de la baie
        m = meteo.meteo_journaliere(baie["meteo"]["lat"], baie["meteo"]["lon"],
                                    debut=debut, force=force)
        for nom, mois in (("mars_avr", saisons["meteo_mars_avr"]),
                          ("mars_mai", saisons["meteo_mars_mai"])):
            mm = m[m["date"].dt.month.isin(mois)].copy()
            mm["annee"] = mm["date"].dt.year
            agg = mm.groupby("annee").agg(**{
                "ray_" + nom: ("ray_mj_m2", "sum"),
                "temp_" + nom: ("temp_c", "mean"),
                "pluie_" + nom: ("pluie_mm", "sum"),
            }).reset_index()
            t = t.merge(agg, on="annee", how="left")

        t.insert(0, "baie", code)
        t["debit_proxy"] = any(r.get("debit_proxy", False) for r in baie["rivieres"])
        tables.append(t)

    table = pd.concat(tables, ignore_index=True).sort_values(["baie", "annee"])
    # règle de qualité : une surface CEVA quasi nulle est traitée comme manquante
    part_min = config["periode"].get("part_min_ceva")
    if part_min:
        moyenne = table.groupby("baie")["surface_ha"].transform("mean")
        table["ceva_quasi_nul"] = table["surface_ha"] < part_min * moyenne
        table.loc[table["ceva_quasi_nul"], "surface_ha"] = np.nan
    # variables « année précédente » : connues avant la saison
    g = table.groupby("baie")
    table["surface_n_1"] = g["surface_ha"].shift(1)
    table["volume_n_1"] = g["volume_m3"].shift(1)
    # on garde uniquement les années consécutives (shift suppose une ligne par année)
    trou = g["annee"].diff() != 1
    table.loc[trou, ["surface_n_1", "volume_n_1"]] = np.nan
    p = config["periode"]
    table = table[(table["annee"] >= p["annee_min"])].reset_index(drop=True)
    return table


def _combiner_rivieres(liste: List[pd.DataFrame], suffixe: str) -> pd.DataFrame:
    """Somme débits et flux des rivières d'une baie ; NO3 moyen pondéré par le débit."""
    if not liste:
        return pd.DataFrame(columns=["annee"])
    annees = sorted(set().union(*[set(x["annee"]) for x in liste]))
    base = pd.DataFrame({"annee": annees})
    debit = 0.0
    flux = 0.0
    n_jours = None
    for x in liste:
        x = base.merge(x, on="annee", how="left")
        debit = debit + x["debit"].values      # NaN si une rivière manque
        flux = flux + x["flux_t"].values
        n_jours = x["n_jours"].values if n_jours is None else np.fmax(n_jours, x["n_jours"].values)
    # NO3 moyen pondéré par le débit = masse de NO3 / volume d'eau écoulé (g/m³ = mg/L)
    volume_m3 = debit * n_jours * 86400.0
    with np.errstate(invalid="ignore", divide="ignore"):
        no3 = np.where(volume_m3 > 0, flux * 1e6 / NO3_VERS_N / volume_m3, np.nan)
    return pd.DataFrame({
        "annee": annees,
        "debit_" + suffixe: debit,
        "no3_" + suffixe: no3,
        "flux_n_" + suffixe + "_t": flux,
    })
