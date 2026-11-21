"""Surface d'ulves par baie et par scène Sentinel-2 L2A (Microsoft Planetary Computer).

Aucun compte n'est nécessaire : la recherche STAC est publique et les fichiers sont
lus avec un jeton SAS anonyme. On ne lit que la fenêtre de l'emprise de la baie
(lecture partielle des COG), environ 3 s par scène.

Pour chaque scène, dans la zone de la baie (sites CEVA ∩ estran/eau WorldCover) :
  valid_ha    pixels sans nuage ni trou (SCL)
  eau_ha      pixels valides couverts d'eau (NDWI > 0) : indicateur de marée
  ulva_ha_sXX pixels valides avec NDVI > seuil XX (ulves émergées ou flottantes)

La marée n'est pas corrigée ici : la sélection de la scène de plus basse mer
se fait ensuite (voir calibration.py).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds, transform_geom
from rasterio.windows import from_bounds

from ..ingestion.cache import DOSSIER_RAW, json_en_cache
from ..ingestion import cache

URL_JETON = "https://planetarycomputer.microsoft.com/api/sas/v1/token/%s"
ENV_GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
HA_PAR_PIXEL = 0.01  # 10 m × 10 m


# ---------------------------------------------------------------- accès Planetary Computer

class Jetons:
    """Jetons SAS anonymes, renouvelés avant expiration (un par collection)."""

    def __init__(self):
        self._jetons = {}
        self._verrou = threading.Lock()

    def signer(self, href: str, collection: str) -> str:
        with self._verrou:
            jeton, expire = self._jetons.get(collection, (None, 0.0))
            if time.time() > expire - 300:
                r = cache.get_json(URL_JETON % collection)
                expire = pd.Timestamp(r["msft:expiry"]).timestamp()
                jeton = r["token"]
                self._jetons[collection] = (jeton, expire)
        return href + ("&" if "?" in href else "?") + jeton


JETONS = Jetons()


def rechercher(url_stac: str, collection: str, emprise: Sequence[float], debut: str,
               fin: str, nuages_max: Optional[float] = None) -> List[dict]:
    """Recherche STAC paginée (POST). Renvoie la liste des items (dict GeoJSON)."""
    import requests
    corps = {"collections": [collection], "bbox": list(emprise),
             "datetime": "%sT00:00:00Z/%sT23:59:59Z" % (debut, fin), "limit": 200}
    if nuages_max is not None:
        corps["query"] = {"eo:cloud_cover": {"lt": nuages_max}}
    items, url = [], url_stac.rstrip("/") + "/search"
    while True:
        for essai in range(5):
            try:
                r = requests.post(url, json=corps, headers=cache.ENTETES, timeout=120)
                r.raise_for_status()
                page = r.json()
                break
            except Exception:
                if essai == 4:
                    raise
                time.sleep(3 * 2 ** essai)
        items.extend(page["features"])
        suivant = [l for l in page.get("links", []) if l.get("rel") == "next"]
        if not suivant:
            return items
        corps = suivant[0].get("body", corps)
        url = suivant[0]["href"]


def scenes_baie(conf: dict, code_baie: str, emprise, annee: int, force: bool = False) -> List[dict]:
    """Items Sentinel-2 d'une baie pour une année (mois de la config), mis en cache."""
    mois = conf["periode"]["mois"]
    debut = "%d-%02d-01" % (annee, min(mois))
    fin = (pd.Timestamp("%d-%02d-01" % (annee, max(mois))) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    chemin = DOSSIER_RAW / "sentinel2" / "stac" / ("%s_%d.json" % (code_baie, annee))
    s = conf["stac"]
    # on ne garde que les champs utiles pour garder un cache léger
    def produire():
        items = rechercher(s["url"], s["collection"], emprise, debut, fin, s["nuages_tuile_max"])
        return [{"id": it["id"], "properties": {k: it["properties"].get(k) for k in (
                    "datetime", "eo:cloud_cover", "s2:processing_baseline", "s2:mgrs_tile",
                    "proj:epsg")},
                 "assets": {b: {"href": it["assets"][b]["href"]} for b in ("B03", "B04", "B08", "SCL")}}
                for it in items if int(it["properties"]["datetime"][5:7]) in mois]
    return json_en_cache(chemin, produire, force=force)


# ---------------------------------------------------------------- zone d'analyse

def grille_fenetre(src, emprise):
    """Fenêtre entière (pixels de 10 m) couvrant l'emprise, dans le CRS de la scène."""
    bornes = transform_bounds("EPSG:4326", src.crs, *emprise)
    fen = from_bounds(*bornes, transform=src.transform).round_offsets().round_lengths()
    return fen, src.window_transform(fen), (int(fen.height), int(fen.width))


_VERROU_MASQUE = threading.Lock()


def masque_zone(conf: dict, code_baie: str, geoms: List[dict], emprise, crs, transform,
                forme) -> np.ndarray:
    """Sites CEVA ∩ classes WorldCover gardées, sur la grille de la scène (mis en cache)."""
    cle = "%s_%s_%.0f_%.0f_%dx%d" % (code_baie, str(crs).replace(":", ""), transform.c,
                                     transform.f, forme[0], forme[1])
    chemin = DOSSIER_RAW / "sentinel2" / "masques" / (cle + ".npy")
    with _VERROU_MASQUE:
        if chemin.exists():
            return np.load(str(chemin))
        sites = rasterize([(transform_geom("EPSG:4326", crs, g), 1) for g in geoms],
                          out_shape=forme, transform=transform, fill=0, dtype="uint8")
        wc = np.zeros(forme, dtype="uint8")
        w = conf["worldcover"]
        items = rechercher(conf["stac"]["url"], w["collection"], emprise,
                           "%d-01-01" % w["annee"], "%d-12-31" % w["annee"])
        if not items:
            raise RuntimeError("Aucune tuile WorldCover %d pour %s" % (w["annee"], code_baie))
        with rasterio.Env(**ENV_GDAL):
            for it in items:
                with rasterio.open(JETONS.signer(it["assets"]["map"]["href"], w["collection"])) as src:
                    tmp = np.zeros(forme, dtype="uint8")
                    reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=crs,
                              resampling=Resampling.nearest, dst_nodata=0)
                    wc = np.maximum(wc, tmp)
        zone = (sites == 1) & np.isin(wc, w["classes_gardees"])
        chemin.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(chemin), zone)
        return zone


# ---------------------------------------------------------------- analyse d'une scène

def reflectance(dn: np.ndarray, baseline: Optional[str]) -> np.ndarray:
    """DN → réflectance. Depuis la version de traitement 04.00 (janvier 2022),
    l'ESA ajoute un décalage de +1000 aux bandes L2A : il faut le retirer."""
    decalage = 1000.0 if baseline and float(baseline) >= 4.0 else 0.0
    return (dn.astype("float32") - decalage) / 10000.0


def analyser_scene(conf: dict, code_baie: str, item: dict, geoms: List[dict], emprise) -> dict:
    p = item["properties"]
    coll = conf["stac"]["collection"]
    with rasterio.Env(**ENV_GDAL):
        bandes = {}
        with rasterio.open(JETONS.signer(item["assets"]["B04"]["href"], coll)) as src:
            fen, transform, forme = grille_fenetre(src, emprise)
            crs = src.crs
            bandes["B04"] = src.read(1, window=fen, boundless=True, fill_value=0)
        for b in ("B03", "B08"):
            with rasterio.open(JETONS.signer(item["assets"][b]["href"], coll)) as src:
                bandes[b] = src.read(1, window=fen, boundless=True, fill_value=0)
        # SCL est à 20 m : on lit la même emprise rééchantillonnée à 10 m (plus proche voisin)
        with rasterio.open(JETONS.signer(item["assets"]["SCL"]["href"], coll)) as src:
            bornes = array_bounds(forme[0], forme[1], transform)
            fen_scl = from_bounds(*bornes, transform=src.transform)
            scl = src.read(1, window=fen_scl, out_shape=forme, boundless=True, fill_value=0,
                           resampling=Resampling.nearest)

    zone = masque_zone(conf, code_baie, geoms, emprise, crs, transform, forme)
    base = p.get("s2:processing_baseline")
    g, r, nir = (reflectance(bandes[b], base) for b in ("B03", "B04", "B08"))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - r) / (nir + r)
        ndwi = (g - nir) / (g + nir)
    valide = zone & (bandes["B04"] > 0) & ~np.isin(scl, conf["scl_invalides"]) & np.isfinite(ndvi)
    eau = valide & (ndwi > 0)
    date_utc = pd.Timestamp(p["datetime"])
    res = {
        "baie": code_baie,
        "id": item["id"],
        "date": date_utc.tz_convert("Europe/Paris").strftime("%Y-%m-%d"),
        "heure_utc": date_utc.strftime("%H:%M"),
        "tuile": p.get("s2:mgrs_tile"),
        "baseline": base,
        "nuages_tuile": p.get("eo:cloud_cover"),
        "zone_ha": float(zone.sum() * HA_PAR_PIXEL),
        "valid_ha": float(valide.sum() * HA_PAR_PIXEL),
        "eau_ha": float(eau.sum() * HA_PAR_PIXEL),
    }
    for s in conf["seuils_ndvi"]:
        res["ulva_ha_s%03d" % round(s * 100)] = float((valide & (ndvi > s)).sum() * HA_PAR_PIXEL)
    return res


def analyser_baie(conf: dict, code_baie: str, baie: dict, geoms: List[dict],
                  annees: Sequence[int], workers: int = 6, force: bool = False,
                  journal=print) -> pd.DataFrame:
    """Analyse toutes les scènes d'une baie (résultat de chaque scène mis en cache)."""
    emprise = baie["emprise"]
    items = []
    for a in annees:
        items.extend(scenes_baie(conf, code_baie, emprise, a, force=force))
    dossier = DOSSIER_RAW / "sentinel2" / "scenes" / code_baie
    journal("[%s] %d scènes à analyser" % (code_baie, len(items)))

    def une(item):
        chemin = dossier / (item["id"] + ".json")
        try:
            return json_en_cache(chemin, lambda: analyser_scene(conf, code_baie, item, geoms, emprise),
                                 force=force)
        except Exception as e:  # une scène illisible ne doit pas arrêter la baie
            return {"baie": code_baie, "id": item["id"], "erreur": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        resultats = list(pool.map(une, items))
    df = pd.DataFrame(resultats)
    if "erreur" in df:
        n = df["erreur"].notna().sum()
        if n:
            journal("[%s] %d scènes en erreur (relancer pour réessayer)" % (code_baie, n))
        df = df[df["erreur"].isna()].drop(columns="erreur")
    return df


def geometries_sites(ids: Sequence[int], force: bool = False) -> List[dict]:
    from ..ingestion import ceva
    couche = ceva.telecharger_couche(ceva.COUCHE_SITES, force=force)
    return [f["geometry"] for f in couche["features"] if int(f["properties"]["ID"]) in set(ids)]
