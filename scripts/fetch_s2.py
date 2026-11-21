"""Surfaces d'ulves Sentinel-2 par site et par date.

  python scripts/fetch_s2.py --backend cdse                  # Copernicus Data Space (openEO)
  python scripts/fetch_s2.py --backend gee --project MON_PROJET  # Google Earth Engine
"""
import argparse
import time

import pandas as pd
from _common import ROOT
from algathon.ingest import sentinel2 as s2
from algathon.sites import load_ceva, load_config, sector_sites

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["cdse", "gee"], default="cdse")
ap.add_argument("--project", help="ID du projet Google Cloud (backend gee)")
ap.add_argument("--sector", help="un seul secteur (défaut : tous)")
ap.add_argument("--start")
ap.add_argument("--end")
ap.add_argument("--force", action="store_true", help="recalcule les années déjà téléchargées")
args = ap.parse_args()

cfg = load_config(ROOT / "configs/sectors.yaml")
ceva = load_ceva(ROOT / cfg["ceva_sites_file"])
p = cfg["sentinel2"]
start = args.start or cfg["period"]["start"]
end = args.end or cfg["period"]["end"]

out_dir = ROOT / "data/processed/s2"
out_dir.mkdir(parents=True, exist_ok=True)
for name, sec in cfg["sectors"].items():
    if args.sector and name != args.sector:
        continue
    fc = sector_sites(ceva, sec["ceva_sites"], name)
    dst = out_dir / f"{name}_{args.backend}.csv"
    # découpage annuel : un job par année, résultats partiels conservés
    parts = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        y0, y1 = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        part = out_dir / "parts" / f"{name}_{args.backend}_{y}.csv"
        if part.exists() and not args.force:
            parts.append(pd.read_csv(part, parse_dates=["date"]))
            continue
        print(f"[{name}] {args.backend} {y0} -> {y1}", flush=True)
        df = None
        for attempt in range(6):  # 429 = trop de calculs simultanés chez Copernicus -> on patiente
            try:
                if args.backend == "cdse":
                    df = s2.fetch_cdse(fc, y0, y1, p["ndvi_threshold"], p["max_cloud_cover"],
                                       ROOT / "data/raw/s2" / name / str(y))
                else:
                    if not args.project:
                        ap.error("--project requis pour gee")
                    df = s2.fetch_gee(fc, y0, y1, p["ndvi_threshold"], p["max_cloud_cover"], args.project)
                break
            except Exception as e:
                if "429" in repr(e) and attempt < 5:
                    print(f"  {name} {y} : 429, nouvelle tentative dans {60 * (attempt + 1)} s", flush=True)
                    time.sleep(60 * (attempt + 1))
                    continue
                print(f"  ECHEC {name} {y}: {e!r}", flush=True)
                break
        if df is None:  # on continue avec les autres années
            continue
        df = s2.finalize(df)
        df.insert(0, "sector", name)
        part.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(part, index=False)
        print(f"  {name} {y}: {len(df)} observations valides", flush=True)
        parts.append(df)
    if parts:
        allp = pd.concat(parts, ignore_index=True).sort_values(["site", "date"])
        allp.to_csv(dst, index=False)
        print(f"[{name}] total {len(allp)} observations -> {dst}", flush=True)
