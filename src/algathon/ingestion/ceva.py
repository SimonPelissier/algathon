"""Données CEVA diffusées par le serveur WFS de GéoSAS (Licence Ouverte Etalab).

Deux couches :
- greenseas:prolif_AV_site : 129 sites, surface cumulée d'échouage d'ulves par an
  (champs cumul_2002 … cumul_2022). Pour les baies sableuses, c'est la somme des
  surfaces mesurées lors des survols de l'année (mai, juillet, septembre).
  Unité : hectares d'après les bilans CEVA (à confirmer, non précisée dans la fiche).
- greenseas:enquete_communale_AV_bretagne : 279 communes, volumes d'algues vertes
  ramassés déclarés par les communes, en m³ par an (VOL_1997 … VOL_2022).

Remarque : le jeu PNDB « enquêtes communales CEVA » (opendatasoft) est vide ;
la même donnée est servie ici par GéoSAS.
"""
from __future__ import annotations

import re

import pandas as pd

from .cache import DOSSIER_RAW, get_json, json_en_cache

URL_WFS = "https://geosas.fr/geoserver/wfs"
COUCHE_SITES = "greenseas:prolif_AV_site"
COUCHE_ENQUETES = "greenseas:enquete_communale_AV_bretagne"


def telecharger_couche(nom_couche: str, force: bool = False) -> dict:
    """Télécharge une couche WFS en GeoJSON (WGS84, ordre lon/lat) et la met en cache."""
    params = {
        "service": "WFS",
        "version": "1.0.0",           # 1.0.0 garantit l'ordre lon/lat en EPSG:4326
        "request": "GetFeature",
        "typeName": nom_couche,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
    }
    chemin = DOSSIER_RAW / "ceva" / (nom_couche.split(":")[1] + ".geojson")
    return json_en_cache(chemin, lambda: get_json(URL_WFS, params), force=force)


def surfaces_sites(force: bool = False) -> pd.DataFrame:
    """Table longue : une ligne par site et par année.

    Colonnes : id_site, site, typologie, secteur, annee, surface_ha, moyenne_ha.
    `surface_ha` vaut NaN quand le CEVA n'a pas de valeur (site non inventorié).
    Pour les vasières, la couche donne aussi un maximum annuel (MAX_2008 …), non repris ici.
    """
    couche = telecharger_couche(COUCHE_SITES, force=force)
    lignes = []
    for f in couche["features"]:
        p = f["properties"]
        for cle, valeur in p.items():
            m = re.fullmatch(r"cumul_(\d{4})", cle)
            if m:
                lignes.append({
                    "id_site": int(p["ID"]),
                    "site": p["site"],
                    "typologie": p["typologie"],
                    "secteur": p["GRAND_SECT"],
                    "annee": int(m.group(1)),
                    "surface_ha": valeur,
                    "moyenne_ha": p.get("moy_cumul"),
                })
    df = pd.DataFrame(lignes)
    df["surface_ha"] = pd.to_numeric(df["surface_ha"], errors="coerce")
    return df.sort_values(["id_site", "annee"]).reset_index(drop=True)


def volumes_communes(force: bool = False) -> pd.DataFrame:
    """Table longue : une ligne par commune et par année (volume ramassé en m³)."""
    couche = telecharger_couche(COUCHE_ENQUETES, force=force)
    lignes = []
    for f in couche["features"]:
        p = f["properties"]
        for cle, valeur in p.items():
            m = re.fullmatch(r"VOL_(\d{4})", cle)
            if m:
                lignes.append({
                    "insee": str(p["INSEE"]),
                    "commune": p["NomCom"],
                    "annee": int(m.group(1)),
                    "volume_m3": valeur,
                })
    df = pd.DataFrame(lignes)
    df["volume_m3"] = pd.to_numeric(df["volume_m3"], errors="coerce")
    return df.sort_values(["insee", "annee"]).reset_index(drop=True)
