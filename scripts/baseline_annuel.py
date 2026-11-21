"""Modèles annuels : Ridge, Random Forest, Extra Trees et gradient boosting, contre deux références naïves.

    python scripts/ingest.py            # d'abord, pour produire data/processed/baies_annuel.csv
    python scripts/baseline_annuel.py
    python scripts/baseline_annuel.py --importance   # + importance des variables (~6 min)

Écrit les prédictions hors échantillon dans data/processed/predictions_<jeu>_<methode>.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.baseline_annuel import (  # noqa: E402
    JEUX_DE_VARIABLES, evaluer, importance_hors_echantillon, scores, scores_par_baie,
)

RACINE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=str(RACINE / "data" / "processed" / "baies_annuel.csv"))
    ap.add_argument("--exclure-baies", nargs="*", default=[],
                    help="baies à exclure (ex. guisseny, dont le débit est un substitut)")
    ap.add_argument("--importance", action="store_true",
                    help="calcule l'importance des variables pour les arbres (~6 min)")
    args = ap.parse_args()

    table = pd.read_csv(args.table)
    table = table[~table["baie"].isin(args.exclure_baies)]
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)

    for jeu, variables in JEUX_DE_VARIABLES.items():
        for methode in ("glissante", "loyo"):
            pred = evaluer(table, variables, methode=methode)
            pred.to_csv(RACINE / "data" / "processed" /
                        ("predictions_%s_%s.csv" % (jeu, methode)), index=False)
            print("\n=== %s | validation %s | années testées %d-%d" % (
                jeu, methode, pred["annee"].min(), pred["annee"].max()))
            print(scores(pred).to_string())
            if methode == "glissante":
                print("\nMAE relative par baie :")
                print(scores_par_baie(pred).to_string())
        if not args.importance:
            continue
        # ce que les arbres utilisent vraiment, mesuré sur des années non vues
        imp = pd.DataFrame({m: importance_hors_echantillon(table, variables, m)
                            for m in ("random_forest", "extra_trees")})
        print("\nImportance hors échantillon des variables (%%), %s :" % jeu)
        print(imp.round(1).to_string())


if __name__ == "__main__":
    main()
