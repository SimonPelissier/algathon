"""Surfaces d'ulves Sentinel-2 par site et par date.

  python scripts/fetch_s2.py --backend cdse                  # Copernicus Data Space (openEO)
  python scripts/fetch_s2.py --backend gee --project MON_PROJET  # Google Earth Engine
"""
import argparse
from _common import ROOT
from algathon.ingest import sentinel2 as s2
from algathon.sites import load_ceva, load_config, sector_sites

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["cdse", "gee"], default="cdse")
ap.add_argument("--project", help="ID du projet Google Cloud (backend gee)")
ap.add_argument("--sector", help="un seul secteur (défaut : tous)")
ap.add_argument("--start")
ap.add_argument("--end")
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
    print(f"[{name}] {args.backend} {start} -> {end}")
    fc = sector_sites(ceva, sec["ceva_sites"], name)
    if args.backend == "cdse":
        df = s2.fetch_cdse(fc, start, end, p["ndvi_threshold"], p["max_cloud_cover"],
                           ROOT / "data/raw/s2" / name)
    else:
        if not args.project:
            ap.error("--project requis pour gee")
        df = s2.fetch_gee(fc, start, end, p["ndvi_threshold"], p["max_cloud_cover"], args.project)
    df = s2.finalize(df)
    df.insert(0, "sector", name)
    dst = out_dir / f"{name}_{args.backend}.csv"
    df.to_csv(dst, index=False)
    print(f"  {len(df)} observations valides -> {dst}")
