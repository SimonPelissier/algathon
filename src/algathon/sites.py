"""Sites de prolifération CEVA et secteurs pilotes."""
import json
from pathlib import Path

import pandas as pd
import yaml

CEVA_WFS = (
    "https://geosas.fr/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=greenseas:prolif_AV_site&outputFormat=application/json&srsName=EPSG:4326"
)


def load_config(path="configs/sectors.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ceva(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sector_sites(ceva: dict, names: list[str], sector: str) -> dict:
    """FeatureCollection des sites CEVA d'un secteur (propriétés réduites à site/sector)."""
    by_name = {f["properties"]["site"]: f for f in ceva["features"]}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"Sites CEVA introuvables : {missing}")
    feats = [
        {"type": "Feature", "geometry": by_name[n]["geometry"],
         "properties": {"site": n, "sector": sector}}
        for n in names
    ]
    return {"type": "FeatureCollection", "features": feats}


def ceva_annual_table(ceva: dict) -> pd.DataFrame:
    """Cumuls annuels de surface d'ulves (ha) par site, format long : site, year, cumul_ha."""
    rows = []
    for f in ceva["features"]:
        p = f["properties"]
        for k, v in p.items():
            if k.startswith("cumul_") and v is not None:
                rows.append({"site": p["site"], "typologie": p["typologie"],
                             "year": int(k[6:]), "cumul_ha": float(v)})
    return pd.DataFrame(rows)
