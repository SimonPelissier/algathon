"""Rivières côtières pour la carte : tracé OSM, stations nitrates/débit rattachées, plage d'arrivée.

  python scripts/build_rivers.py
Sorties (app/data/rivers/) :
  rivers.json          tracés simplifiés par rivière (nom, chemin [[lon, lat], ...], embouchure, plage)
  river_no3_st.csv     stations de nitrates rattachées à chaque rivière (à < 1,5 km du tracé)
  river_flow.csv       débit moyen mensuel par rivière (station de débit la plus en aval rattachée)

Source du tracé : OpenStreetMap (waterway=river, + ruisseaux nommés des baies pilotes), via Overpass, ODbL.
"""
import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from _common import ROOT
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

from algathon.sites import load_config

OVERPASS = "https://overpass-api.de/api/interpreter"
BBOX = (47.0, -5.3, 49.0, -0.9)  # sud, ouest, nord, est (Bretagne et marges)
out = ROOT / "app/data/rivers"
out.mkdir(parents=True, exist_ok=True)
cfg = load_config(ROOT / "configs/sectors.yaml")
pilot_rivers = sorted({r for s in cfg["sectors"].values() for r in (s.get("rivers") or [])})


def overpass(q):
    for i in range(4):
        r = requests.post(OVERPASS, data={"data": q}, timeout=600,
                          headers={"User-Agent": "algathon-hackathon/0.1"})
        if r.status_code == 200:
            return r.json()
        print("overpass", r.status_code, "nouvel essai", flush=True)
        time.sleep(30 * (i + 1))
    raise RuntimeError("Overpass indisponible")


s, w, n, e = BBOX
names = "|".join(pilot_rivers)
q = f"""[out:json][timeout:500];
(way["waterway"="river"]({s},{w},{n},{e});
 way["waterway"="stream"]["name"~"({names})$"]({s},{w},{n},{e}););
out geom tags;"""
print("Overpass : rivières…", flush=True)
js = overpass(q)
rows = []
for el in js["elements"]:
    name = el.get("tags", {}).get("name")
    if not name or "geometry" not in el:
        continue
    rows.append({"name": name, "geometry": LineString([(p["lon"], p["lat"]) for p in el["geometry"]])})
g = gpd.GeoDataFrame(rows, crs=4326)
print(f"{len(g)} tronçons, {g['name'].nunique()} noms", flush=True)

# une géométrie par nom de rivière (fusion des tronçons), en Lambert 93
g = g.to_crs(2154)
merged = g.groupby("name")["geometry"].apply(lambda s_: linemerge(list(s_)))
riv = gpd.GeoDataFrame({"name": merged.index}, geometry=list(merged.values), crs=2154)
riv["length_km"] = riv.length / 1000

# côte : on garde les rivières dont une extrémité est proche d'une plage suivie (embouchure)
sites = pd.read_csv(ROOT / "app/data/sites.csv")
sp = gpd.GeoSeries(gpd.points_from_xy(sites["lon"], sites["lat"]), crs=4326).to_crs(2154)


def ends(geom):
    lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
    return [Point(c) for ln in lines for c in (ln.coords[0], ln.coords[-1])]


keep = []
for r in riv.itertuples():
    best = None
    for p in ends(r.geometry):
        d = sp.distance(p)
        i = int(d.values.argmin())
        if best is None or d.iloc[i] < best[0]:
            best = (float(d.iloc[i]), i, p)
    if best[0] <= 10000 and r.length_km >= 5:  # embouchure à moins de 10 km d'une plage, rivière >= 5 km
        keep.append((r.Index, sites["site"].iloc[best[1]], best[2], best[0] / 1000))
riv = riv.loc[[k[0] for k in keep]].copy()
riv["site"] = [k[1] for k in keep]
riv["mouth_km"] = [round(k[3], 1) for k in keep]
mouth = gpd.GeoSeries([k[2] for k in keep], crs=2154, index=riv.index).to_crs(4326)
riv["mouth_lon"], riv["mouth_lat"] = mouth.x.values, mouth.y.values
print(f"{len(riv)} rivières côtières retenues (embouchure à < 10 km d'une plage suivie)", flush=True)

# stations de nitrates à moins de 1,5 km du tracé
st = pd.read_parquet(ROOT / "app/data/no3_station_year.parquet").groupby("code_station")[["lon", "lat"]].first()
stp = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]), crs=4326).to_crs(2154)
j = gpd.sjoin_nearest(stp, riv[["name", "geometry"]], max_distance=1500, distance_col="d").dropna(subset=["name"])
j = j.reset_index()[["code_station", "name", "d"]]
j.to_csv(out / "river_no3_st.csv", index=False)
print(f"{len(j)} stations de nitrates rattachées à {j['name'].nunique()} rivières", flush=True)

# débit : station la plus en aval (plus fort débit moyen) à moins de 1,5 km
f = pd.read_parquet(ROOT / "data/processed/regional/flows_monthly.parquet")
fst = f.groupby("code_station")[["lon", "lat"]].first()
fsp = gpd.GeoDataFrame(fst, geometry=gpd.points_from_xy(fst["lon"], fst["lat"]), crs=4326).to_crs(2154)
jf = gpd.sjoin_nearest(fsp, riv[["name", "geometry"]], max_distance=1500).dropna(subset=["name"]).reset_index()
f = f.merge(jf[["code_station", "name"]], on="code_station")
f["month"] = pd.to_datetime(f["date"]).dt.month
f["year"] = pd.to_datetime(f["date"]).dt.year
mean_q = f.groupby("code_station")["q_m3s"].mean()
best = f.assign(mq=f["code_station"].map(mean_q)).sort_values("mq").groupby("name")["code_station"].last()
flow = (f[f["code_station"].isin(best.values)].groupby(["name", "year", "month"])["q_m3s"].mean().reset_index())
flow.to_csv(out / "river_flow.csv", index=False)
print(f"débit disponible pour {flow['name'].nunique()} rivières", flush=True)

# tracés simplifiés (≈ 150 m) pour l'affichage
riv4 = riv.copy()
riv4["geometry"] = riv4.simplify(150)
riv4 = riv4.to_crs(4326)
feats = []
for r in riv4.itertuples():
    lines = list(r.geometry.geoms) if isinstance(r.geometry, MultiLineString) else [r.geometry]
    for ln in lines:
        if len(ln.coords) >= 2:
            feats.append({"name": r.name, "site": r.site, "path": [[round(x, 5), round(y, 5)] for x, y in ln.coords],
                          "mouth": [round(r.mouth_lon, 5), round(r.mouth_lat, 5)], "length_km": round(r.length_km, 1)})
(out / "rivers.json").write_text(json.dumps(feats), encoding="utf-8")
print(f"rivers.json : {len(feats)} tracés, {(out / 'rivers.json').stat().st_size / 1e6:.2f} Mo")
