"""Surfaces d'ulves par site et par date à partir de Sentinel-2 L2A.

Deux backends, même sortie (site, date, ulva_ha, valid_ha, site_ha) :
- "cdse" : Copernicus Data Space Ecosystem via openEO (compte gratuit sur dataspace.copernicus.eu)
- "gee"  : Google Earth Engine (projet Cloud enregistré pour Earth Engine)

Méthode (baseline, à calibrer) : masque nuages/ombres via la bande SCL, puis pixel = ulve
si NDVI > seuil, à l'intérieur des polygones de sites CEVA.
"""
import json
from pathlib import Path

import pandas as pd

# Classes SCL écartées : 0 no data, 1 saturé, 3 ombre nuage, 8-9 nuages, 10 cirrus
SCL_BAD = [0, 1, 3, 8, 9, 10]


# --------------------------------------------------------------------------- CDSE / openEO
def fetch_cdse(sites_fc: dict, start: str, end: str, ndvi_threshold: float,
               max_cloud_cover: float, out_dir: Path) -> pd.DataFrame:
    import openeo

    conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=[start, end],
        bands=["B04", "B08", "SCL"],
        max_cloud_cover=max_cloud_cover,
    )
    scl = cube.band("SCL")
    bad = scl == SCL_BAD[0]
    for c in SCL_BAD[1:]:
        bad = bad | (scl == c)
    b4, b8 = cube.band("B04"), cube.band("B08")
    ndvi = (b8 - b4) / (b8 + b4)

    pix_ha = 0.01  # pixel 10 m x 10 m = 0.01 ha
    ulva = ((ndvi > ndvi_threshold) * pix_ha).mask(bad).add_dimension("bands", "ulva_ha", type="bands")
    valid = (ndvi * 0 + pix_ha).mask(bad).add_dimension("bands", "valid_ha", type="bands")
    allpix = (ndvi * 0 + pix_ha).add_dimension("bands", "site_ha", type="bands")
    merged = ulva.merge_cubes(valid).merge_cubes(allpix)

    agg = merged.aggregate_spatial(geometries=sites_fc, reducer="sum")
    job = agg.create_job(title="algathon-ulva", out_format="JSON")
    job.start_and_wait()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = job.get_results().download_files(out_dir)
    js_path = next(p for p in paths if str(p).endswith(".json"))
    return _parse_openeo_json(json.loads(Path(js_path).read_text()), sites_fc)


def _parse_openeo_json(js: dict, sites_fc: dict) -> pd.DataFrame:
    """Format openEO aggregate_spatial JSON : {date: [[ulva, valid, site] par géométrie]}."""
    names = [f["properties"]["site"] for f in sites_fc["features"]]
    rows = []
    for date, per_geom in js.items():
        for name, vals in zip(names, per_geom):
            if vals and len(vals) == 3:
                rows.append({"site": name, "date": pd.to_datetime(date).tz_localize(None).normalize(),
                             "ulva_ha": vals[0], "valid_ha": vals[1], "site_ha": vals[2]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Earth Engine
def fetch_gee(sites_fc: dict, start: str, end: str, ndvi_threshold: float,
              max_cloud_cover: float, project: str) -> pd.DataFrame:
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

    fc = ee.FeatureCollection(sites_fc)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(fc.geometry())
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover)))

    def per_image(img):
        scl = img.select("SCL")
        bad = scl.remap(SCL_BAD, [1] * len(SCL_BAD), 0)
        ndvi = img.normalizedDifference(["B8", "B4"])
        area_ha = ee.Image.pixelArea().divide(1e4)
        stack = ee.Image.cat(
            area_ha.updateMask(ndvi.gt(ndvi_threshold)).updateMask(bad.Not()).rename("ulva_ha"),
            area_ha.updateMask(bad.Not()).rename("valid_ha"),
            area_ha.updateMask(img.select("B4").mask()).rename("site_ha"),
        )
        stats = stack.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=10)
        date = img.date().format("YYYY-MM-dd")
        return stats.map(lambda f: f.set("date", date))

    rows = []
    years = range(int(start[:4]), int(end[:4]) + 1)
    for y in years:  # découpage annuel pour rester sous les limites de getInfo
        sub = col.filterDate(max(start, f"{y}-01-01"), min(end, f"{y + 1}-01-01"))
        feats = sub.map(per_image).flatten().getInfo()["features"]
        rows += [{k: f["properties"].get(k) for k in ("site", "date", "ulva_ha", "valid_ha", "site_ha")}
                 for f in feats]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # plusieurs tuiles le même jour -> on garde la meilleure couverture
    return df.sort_values("valid_ha").groupby(["site", "date"]).tail(1)


def finalize(df: pd.DataFrame, min_valid_frac: float = 0.5) -> pd.DataFrame:
    df = df.dropna(subset=["valid_ha"]).copy()
    df["valid_frac"] = df["valid_ha"] / df["site_ha"]
    df = df[df["valid_frac"] >= min_valid_frac]
    df["ulva_frac"] = df["ulva_ha"] / df["valid_ha"]
    # surface extrapolée à tout le site (pour les images partiellement nuageuses)
    df["ulva_ha_est"] = df["ulva_frac"] * df["site_ha"]
    return df.sort_values(["site", "date"]).reset_index(drop=True)
