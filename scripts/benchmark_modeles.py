"""Benchmark de 11 modèles (+ 2 références naïves) sur la surface d'échouage annuelle.

    python scripts/ingest.py               # d'abord, pour data/processed/baies_annuel.csv
    python scripts/benchmark_modeles.py

Même validation temporelle pour tous : fenêtre glissante (on n'apprend que sur le passé)
et leave-one-year-out (sans l'année N+1). Intervalles à 90 % par bootstrap sur les années.
Sortie : data/processed/benchmark_<jeu>_<validation>.csv
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.baseline_annuel import JEUX_DE_VARIABLES, evaluer  # noqa: E402
from algathon.benchmark import bootstrap_annees, tous_les_modeles  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
PROC = RACINE / "data" / "processed"


def main():
    warnings.filterwarnings("ignore")  # convergence des petits modèles, sans incidence ici
    table = pd.read_csv(PROC / "baies_annuel.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    for jeu, variables in JEUX_DE_VARIABLES.items():
        for methode in ("glissante", "loyo"):
            t = time.time()
            pred = evaluer(table, variables, methode=methode, modeles=tous_les_modeles)
            res = bootstrap_annees(pred)
            res.to_csv(PROC / ("benchmark_%s_%s.csv" % (jeu, methode)))
            print("\n=== %s | validation %s | %d prédictions | %.0f s" % (
                jeu, methode, len(pred.dropna()), time.time() - t))
            print(res.to_string())


if __name__ == "__main__":
    main()
