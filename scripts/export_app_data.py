"""Exporte des données agrégées et légères dans app/data/ (versionnées, pour déployer l'application)."""
import shutil

import pandas as pd
from _common import ROOT

from algathon.sites import ceva_annual_table, load_ceva, load_config

src, dst = ROOT / "data/processed", ROOT / "app/data"
dst.mkdir(parents=True, exist_ok=True)
cfg = load_config(ROOT / "configs/sectors.yaml")

# sites + secteur pilote
sites = pd.read_csv(src / "regional/sites.csv")
pilot = {s: name for name, sec in cfg["sectors"].items() for s in sec["ceva_sites"]}
sites["pilot_sector"] = sites["site"].map(pilot)
sites.to_csv(dst / "sites.csv", index=False)
ceva_annual_table(load_ceva(ROOT / cfg["ceva_sites_file"])).to_csv(dst / "ceva_annual.csv", index=False)

# modèle saisonnier
for f in ["table.parquet", "cv_predictions.csv", "hindcast.csv", "metrics.json", "model.joblib"]:
    shutil.copy(src / "seasonal" / f, dst / f)

# nitrates : médiane annuelle par station
no3 = pd.concat([pd.read_parquet(p) for p in sorted((src / "regional").glob("nitrates_*.parquet"))])
no3["year"] = no3["date"].dt.year
agg = (no3.groupby(["code_station", "year"])
          .agg(no3=("resultat", "median"), n=("resultat", "size"),
               lon=("longitude", "first"), lat=("latitude", "first")).reset_index())
agg.to_parquet(dst / "no3_station_year.parquet", index=False)

# élevages : position et type seulement (pas de raison sociale)
lv = pd.read_csv(src / "regional/livestock_icpe.csv")
lv[["longitude", "latitude", "porcs", "bovins", "volailles"]].to_csv(dst / "livestock.csv", index=False)

# baies pilotes : débits/nitrates journaliers + surfaces Sentinel-2
for name in cfg["sectors"]:
    for f in [f"hydro/{name}.csv", f"hydro/{name}_stations.csv", f"s2/{name}_cdse.csv"]:
        p = src / f
        if p.exists():
            (dst / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst / f)

total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
print(f"app/data : {total / 1e6:.1f} Mo")
