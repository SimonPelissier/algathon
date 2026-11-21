"""Télécharge toutes les sources ouvertes dans data/raw/ puis construit la table annuelle.

    python scripts/ingest.py            # utilise le cache s'il existe (idempotent)
    python scripts/ingest.py --force    # retélécharge tout

Sortie : data/processed/baies_annuel.csv (une ligne par baie et par année).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.series_annuelles import RACINE, charger_config, table_annuelle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(RACINE / "configs" / "baies.yaml"))
    ap.add_argument("--force", action="store_true", help="ignore le cache et retélécharge")
    args = ap.parse_args()

    table = table_annuelle(charger_config(args.config), force=args.force)
    sortie = RACINE / "data" / "processed" / "baies_annuel.csv"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(sortie, index=False)

    print("Table écrite : %s (%d lignes, %d baies)" % (
        sortie.relative_to(RACINE), len(table), table["baie"].nunique()))
    resume = table.groupby("baie").agg(
        annees=("annee", "count"),
        surface=("surface_ha", lambda s: int(s.notna().sum())),
        flux_jan_avr=("flux_n_jan_avr_t", lambda s: int(s.notna().sum())),
        flux_mai_aout=("flux_n_mai_aout_t", lambda s: int(s.notna().sum())),
        meteo=("ray_mars_mai", lambda s: int(s.notna().sum())),
    )
    print("\nNombre d'années renseignées par baie (sur %d-%d) :" % (
        table["annee"].min(), table["annee"].max()))
    print(resume.to_string())


if __name__ == "__main__":
    main()
