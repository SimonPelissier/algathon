"""Télécharge la couche CEVA des sites de prolifération d'ulves (2002-2022, Licence Ouverte)."""
import requests
from _common import ROOT
from algathon.sites import CEVA_WFS, ceva_annual_table, load_ceva

out = ROOT / "data/raw/ceva/prolif_AV_site.geojson"
out.parent.mkdir(parents=True, exist_ok=True)
r = requests.get(CEVA_WFS, timeout=180)
r.raise_for_status()
out.write_bytes(r.content)

tab = ceva_annual_table(load_ceva(out))
dst = ROOT / "data/processed/ceva_annual.csv"
tab.to_csv(dst, index=False)
print(f"{tab['site'].nunique()} sites, {tab['year'].min()}-{tab['year'].max()} -> {dst}")
