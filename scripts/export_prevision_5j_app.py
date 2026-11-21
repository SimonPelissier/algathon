"""Copie les résultats du système de prévision à 5 jours dans app/data/prevision_5j/.

    python scripts/affiner_s2.py --toutes-claires
    python scripts/prevision_5j_experimentale.py corrige --maree app/data/daily/daily_risk.parquet
    python scripts/export_prevision_5j_app.py

L'onglet « Prévision à 5 jours » de l'application (app/streamlit_app.py) lit ces fichiers.
"""
import argparse
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from algathon.prevision_5j_app import exporter  # noqa: E402
from algathon.series_annuelles import charger_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(RACINE / "data" / "processed" / "prevision_5j"),
                    help="dossier des sorties de prevision_5j_experimentale.py")
    ap.add_argument("--dest", default=str(RACINE / "app" / "data" / "prevision_5j"))
    args = ap.parse_args()

    ceva = json.loads((RACINE / "data" / "raw" / "ceva" / "prolif_AV_site.geojson").read_text(encoding="utf-8"))
    noms = {int(f["properties"]["ID"]): f["properties"]["site"] for f in ceva["features"]}
    resume = exporter(Path(args.source), Path(args.dest), charger_config(), noms)
    print("Écrit dans %s : %s" % (args.dest, resume))


if __name__ == "__main__":
    main()
