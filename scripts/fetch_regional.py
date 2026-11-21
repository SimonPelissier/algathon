"""Météo, nitrates et élevages pour toutes les plages CEVA (modèle saisonnier).

  python scripts/fetch_regional.py --what weather nitrates livestock
"""
import argparse

from _common import ROOT
from algathon.ingest import regional as R
from algathon.sites import load_ceva

# zone : de la baie du Mont-Saint-Michel / Cotentin à la Vendée
BBOX = [-5.3, 46.2, -0.8, 49.8]
DEPS = ["22", "29", "35", "56", "50", "44", "85"]

ap = argparse.ArgumentParser()
ap.add_argument("--what", nargs="+", default=["weather", "nitrates", "livestock", "flows"])
ap.add_argument("--start", default="2001-01-01")
ap.add_argument("--end", default="2026-11-15")
ap.add_argument("--suffix", default="", help="suffixe du fichier nitrates (téléchargement par morceaux)")
args = ap.parse_args()

out = ROOT / "data/processed/regional"
out.mkdir(parents=True, exist_ok=True)
pts = R.site_points(load_ceva(ROOT / "data/raw/ceva/prolif_AV_site.geojson"))
if not (out / "sites.csv").exists():
    pts.to_csv(out / "sites.csv", index=False)
print(f"{len(pts)} sites CEVA (baies sableuses)")

if "weather" in args.what:
    w = R.weather(pts, args.start, args.end)
    w.to_parquet(out / "weather.parquet", index=False)
    print(f"météo : {len(w)} lignes")
if "livestock" in args.what:
    lv = R.livestock(DEPS)
    lv.to_csv(out / "livestock_icpe.csv", index=False)
    print(f"élevages ICPE : {len(lv)}")
if "flows" in args.what:
    q = R.monthly_flows(pts, BBOX, args.start, args.end)
    q.to_parquet(out / "flows_monthly.parquet", index=False)
    print(f"débits : {q['code_station'].nunique()} stations, {len(q)} mois")
if "nitrates" in args.what:
    n = R.nitrates_region(BBOX, args.start, args.end)
    n.to_parquet(out / f"nitrates{args.suffix}.parquet", index=False)
    print(f"nitrates : {len(n)} analyses, {n['code_station'].nunique()} stations")
