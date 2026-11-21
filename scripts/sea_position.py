"""Direction du large pour chaque plage, pour dessiner les cercles de marée verte en mer.

Trait de côte : polygones terre simplifiés d'OpenStreetMap (osmdata.openstreetmap.de,
simplified-land-polygons-complete-3857.zip, ODbL), à télécharger à part.

  python scripts/sea_position.py <dossier/simplified_land_polygons.shp>

Pour chaque plage (centre de la zone CEVA), on teste 36 directions ; la direction retenue est celle
dont les points de 1 à 20 km tombent le plus souvent en mer (lissée sur les directions
voisines). On mesure aussi la place disponible en mer dans cette direction (jusqu'à la terre suivante).
Ajoute sea_deg (direction du large, degrés depuis le nord) et sea_room_km à sites.csv.
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from _common import ROOT
from shapely.geometry import Point, box
from shapely.prepared import prep

shp = sys.argv[1]
sites_path = ROOT / "data/processed/regional/sites.csv"
sites = pd.read_csv(sites_path)
bb = gpd.GeoSeries([box(sites.lon.min() - 0.3, sites.lat.min() - 0.3, sites.lon.max() + 0.3, sites.lat.max() + 0.3)],
                   crs=4326).to_crs(3857).total_bounds
land = gpd.read_file(shp, bbox=tuple(bb)).to_crs(2154)
L = prep(land.union_all())
pts = gpd.GeoSeries(gpd.points_from_xy(sites.lon, sites.lat), crs=4326).to_crs(2154)

angles = np.arange(0, 360, 10)
dists = np.array([1, 2, 3, 4, 6, 8, 11, 15, 20]) * 1000.0  # plus loin = départage (le large ouvert)
deg, room = [], []
for p in pts:
    score = np.array([sum(not L.contains(Point(p.x + d * np.sin(np.radians(a)), p.y + d * np.cos(np.radians(a))))
                          for d in dists) for a in angles], dtype=float)
    smooth = score + 0.5 * np.roll(score, 1) + 0.5 * np.roll(score, -1)
    a = angles[int(np.argmax(smooth))]
    # place en mer : distance parcourue avant de retoucher la terre (après avoir quitté la plage)
    steps = np.arange(250, 15001, 250)
    wet = [not L.contains(Point(p.x + d * np.sin(np.radians(a)), p.y + d * np.cos(np.radians(a)))) for d in steps]
    first = next((i for i, w in enumerate(wet) if w), None)
    if first is None:
        r = 0.0
    else:
        nxt = next((i for i in range(first, len(steps)) if not wet[i]), len(steps))
        r = steps[nxt - 1] / 1000.0
    deg.append(int(a))
    room.append(round(r, 2))

sites["sea_deg"], sites["sea_room_km"] = deg, room
sites.to_csv(sites_path, index=False)
print(sites[["site", "sea_deg", "sea_room_km"]].to_string())
