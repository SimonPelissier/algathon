"""Débits et nitrates journaliers par secteur pilote (Hub'Eau)."""
import argparse
from _common import ROOT
from algathon.ingest.hubeau import sector_hydro
from algathon.sites import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs/sectors.yaml"))
ap.add_argument("--sector", help="un seul secteur (défaut : tous)")
args = ap.parse_args()
cfg = load_config(args.config)
start, end = cfg["period"]["start"], cfg["period"]["end"]

out_dir = ROOT / "data/processed/hydro"
out_dir.mkdir(parents=True, exist_ok=True)
for name, sec in cfg["sectors"].items():
    if args.sector and name != args.sector:
        continue
    print(f"[{name}]")
    daily, stations = sector_hydro(sec["hydro_bbox"], start, end, sec.get("rivers"))
    daily.insert(0, "sector", name)
    daily.to_csv(out_dir / f"{name}.csv", index=False)
    stations.to_csv(out_dir / f"{name}_stations.csv", index=False)
    print(f"  -> {out_dir / (name + '.csv')} ({daily['flow_m3s'].notna().mean():.0%} jours avec débit)")
