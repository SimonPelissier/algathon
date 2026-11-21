"""Compare les surfaces Sentinel-2 aux cumuls CEVA et produit les séries satellite.

    python scripts/detecter_s2.py     # d'abord
    python scripts/ingest.py          # pour data/processed/baies_annuel.csv
    python scripts/calibrer_s2.py

Sorties dans data/processed/ :
  s2_mensuel.csv  une ligne par baie, année et mois (scène de plus basse mer)
  s2_annuel.csv   cumul mai + juillet + septembre par baie et année, avec la surface CEVA
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.detection import calibration as cal  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
PROC = RACINE / "data" / "processed"


def main():
    conf = yaml.safe_load(open(RACINE / "configs" / "detection.yaml", encoding="utf-8"))
    sel_conf = conf["selection"]
    scenes = pd.read_csv(PROC / "s2_scenes.csv")
    ceva = pd.read_csv(PROC / "baies_annuel.csv")

    mensuel = cal.selection_basse_mer(scenes, sel_conf["part_valide_min"])
    mensuel.to_csv(PROC / "s2_mensuel.csv", index=False)
    cumuls = cal.cumul_mois(mensuel, sel_conf["mois_ceva"])
    annuel = cumuls.merge(ceva[["baie", "annee", "surface_ha"]], on=["baie", "annee"], how="left")
    annuel.to_csv(PROC / "s2_annuel.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 2)
    print("Scènes : %d au total, %d mois-baies avec une scène claire" % (len(scenes), len(mensuel)))
    print("\nMois disponibles par baie (mai, juillet et septembre tous présents) :")
    print(cumuls.dropna(subset=[cal.colonnes_seuils(cumuls)[0]]).groupby("baie")["annee"]
          .agg(["count", "min", "max"]).to_string())
    comp = cal.comparer_ceva(cumuls, ceva)
    print("\nAccord avec le CEVA (années communes, 2016-2022) :")
    print(comp.to_string(index=False))
    meilleur = comp.sort_values("spearman_intra", ascending=False).iloc[0]
    col = "ulva_ha_s%03d" % round(meilleur["seuil_ndvi"] * 100)
    print("\nPar baie, seuil %.2f :" % meilleur["seuil_ndvi"])
    print(cal.correlation_par_baie(cumuls, ceva, col).to_string())


if __name__ == "__main__":
    main()
