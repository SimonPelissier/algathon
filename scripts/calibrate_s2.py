"""Calibre Sentinel-2 sur le CEVA et produit les cumuls « équivalent CEVA » des baies pilotes."""
import json

import pandas as pd
from _common import ROOT

from algathon import calibration as C

s2dir = ROOT / "data/processed/s2"
s2 = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in sorted((s2dir / "parts").glob("*_cdse_*.csv"))])
s2 = s2.drop_duplicates(["site", "date"])
ceva = pd.read_csv(ROOT / "data/processed/ceva_annual.csv")

bg = C.background(s2)
s2c = C.corrected(s2, bg)
annual = C.annual_index(s2c)
cal = C.fit(annual, ceva)
annual["cumul_equiv_ha"] = annual["s2_index"] * cal["k"]
annual = annual.merge(ceva[["site", "year", "cumul_ha"]], on=["site", "year"], how="left")

s2c.to_csv(s2dir / "s2_daily_corrected.csv", index=False)
annual.to_csv(s2dir / "s2_annual.csv", index=False)
cal["background_ha"] = bg.round(2).to_dict()
(s2dir / "calibration.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cal.items()}, indent=1,
                 ensure_ascii=False))
print(annual[annual["year"] >= 2023].round(1).to_string(index=False))
