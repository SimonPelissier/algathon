"""Surfaces d'ulves par baie et par scène Sentinel-2 (Planetary Computer, sans compte).

    python scripts/detecter_s2.py                          # toutes les baies, période de la config
    python scripts/detecter_s2.py --baies saint_brieuc --annees 2019 2019
    python scripts/detecter_s2.py --force                  # recalcule tout

Chaque scène analysée est mise en cache (data/raw/sentinel2/scenes/<baie>/<id>.json) :
une relance après interruption reprend où elle s'était arrêtée.
Sortie : data/processed/s2_scenes.csv (une ligne par baie et par scène).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.detection import sentinel2 as s2  # noqa: E402
from algathon.series_annuelles import RACINE, charger_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(RACINE / "configs" / "detection.yaml"))
    ap.add_argument("--baies", nargs="*", help="codes de baies (défaut : toutes)")
    ap.add_argument("--annees", nargs=2, type=int, help="première et dernière année")
    ap.add_argument("--workers", type=int, default=6, help="scènes lues en parallèle")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config, encoding="utf-8"))
    baies = charger_config()["baies"]
    a0, a1 = args.annees or conf["periode"]["annees"]
    tables = []
    for code, baie in baies.items():
        if args.baies and code not in args.baies:
            continue
        t = time.time()
        geoms = s2.geometries_sites(baie["sites_ceva"])
        df = s2.analyser_baie(conf, code, baie, geoms, range(a0, a1 + 1),
                              workers=args.workers, force=args.force)
        print("[%s] %d scènes analysées en %.0f s" % (code, len(df), time.time() - t), flush=True)
        tables.append(df)

    sortie = RACINE / "data" / "processed" / "s2_scenes.csv"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    nouveau = pd.concat(tables, ignore_index=True)
    if args.baies and sortie.exists():  # on remplace seulement les baies recalculées
        ancien = pd.read_csv(sortie)
        nouveau = pd.concat([ancien[~ancien["baie"].isin(nouveau["baie"].unique())], nouveau])
    nouveau.sort_values(["baie", "date"]).to_csv(sortie, index=False)
    print("Écrit : %s (%d scènes)" % (sortie.relative_to(RACINE), len(nouveau)))


if __name__ == "__main__":
    main()
