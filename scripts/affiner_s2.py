"""Affinage Sentinel-2 : exclut la végétation permanente et le liseré côtier, produit les cartes.

    python scripts/detecter_s2.py     # d'abord (totaux par scène)
    python scripts/affiner_s2.py

Relit le NDVI pixel par pixel des seules scènes de basse mer retenues (cache dans
data/raw/sentinel2/pixels/), puis écrit dans data/processed/ :
  s2_mensuel_corrige.csv   une scène par baie et par mois (avril-octobre), surfaces corrigées
  s2_annuel_corrige.csv    cumul des fenêtres de survol CEVA (± 15 j), avec la surface CEVA
  s2_cartes/<baie>.npz     fréquence de détection par pixel, masques d'exclusion
  s2_scenes_claires_corrige.csv   (option --toutes-claires) toutes les images claires, surfaces
                           corrigées avec les masques ci-dessus : série pour la prévision à 5 jours
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.detection import affinage, calibration as cal, sentinel2 as s2  # noqa: E402
from algathon.ingestion.cache import DOSSIER_RAW  # noqa: E402
from algathon.series_annuelles import charger_config  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
PROC = RACINE / "data" / "processed"


def items_par_id(code_baie):
    """Items STAC déjà en cache (écrits par detecter_s2.py), indexés par identifiant."""
    items = {}
    for f in (DOSSIER_RAW / "sentinel2" / "stac").glob(code_baie + "_*.json"):
        for it in json.loads(f.read_text(encoding="utf-8")):
            items[it["id"]] = it
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--baies", nargs="*")
    ap.add_argument("--toutes-claires", action="store_true",
                    help="corrige aussi toutes les images claires (~900, environ 5 min la 1re fois)")
    args = ap.parse_args()

    conf = yaml.safe_load(open(RACINE / "configs" / "detection.yaml", encoding="utf-8"))
    sel, aff = conf["selection"], conf["affinage"]
    baies = charger_config()["baies"]
    scenes = pd.read_csv(PROC / "s2_scenes.csv")
    ceva = pd.read_csv(PROC / "baies_annuel.csv")

    mensuel = cal.selection_basse_mer(scenes, sel["part_valide_min"])
    fenetres = cal.selection_fenetres(scenes, sel["mois_ceva"], sel["tolerance_jours"],
                                      sel["part_valide_min"])
    sorties_m, sorties_f, sorties_c = [], [], []
    for code, baie in baies.items():
        if args.baies and code not in args.baies:
            continue
        m = mensuel[mensuel["baie"] == code]
        f = fenetres[fenetres["baie"] == code]
        ids = sorted(set(m["id"]) | set(f["id"]))
        items = items_par_id(code)
        geoms = s2.geometries_sites(baie["sites_ceva"])

        def lire(i):
            # coupures réseau passagères (DNS, délai) : quelques nouvelles tentatives
            for essai in range(4):
                try:
                    return i, affinage.pixels_scene(conf, code, items[i], geoms, baie["emprise"])
                except Exception:
                    if essai == 3:
                        raise
                    time.sleep(5 * 2 ** essai)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pixels = dict(pool.map(lire, ids))
        print("[%s] %d scènes lues pixel par pixel" % (code, len(pixels)), flush=True)

        # cartes et masque calculés sur les scènes mensuelles (avril-octobre, toutes années)
        annees = dict(zip(m["id"], m["annee"]))
        cartes = affinage.cartes_baie({i: pixels[i] for i in m["id"]}, annees,
                                      aff["seuil_ndvi"], aff)
        affinage.enregistrer_cartes(code, cartes, PROC / "s2_cartes")
        zone_ha = cartes["zone"].sum() * s2.HA_PAR_PIXEL
        print("[%s] exclus : %.0f ha permanents, %.0f ha de liseré (zone %.0f ha)" % (
            code, cartes["permanent"].sum() * s2.HA_PAR_PIXEL,
            (cartes["lisere"] & ~cartes["permanent"]).sum() * s2.HA_PAR_PIXEL, zone_ha), flush=True)

        for table, sorties in ((m, sorties_m), (f, sorties_f)):
            corr = [affinage.surfaces_corrigees(pixels[i], cartes["exclusion"], cartes["ref"],
                                                conf["seuils_ndvi"]) for i in table["id"]]
            sorties.append(pd.concat([table.reset_index(drop=True), pd.DataFrame(corr)], axis=1))

        if args.toutes_claires:
            claires = scenes[(scenes["baie"] == code) & (scenes["zone_ha"] > 0)]
            claires = claires[claires["valid_ha"] / claires["zone_ha"] >= sel["part_valide_min"]]
            claires = claires.sort_values("valid_ha").drop_duplicates("date", keep="last")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                px_c = dict(pool.map(lire, [i for i in claires["id"] if i not in pixels]))
            px_c.update(pixels)
            corr = [affinage.surfaces_corrigees(px_c[i], cartes["exclusion"], cartes["ref"],
                                                conf["seuils_ndvi"]) for i in claires["id"]]
            sorties_c.append(pd.concat([claires.reset_index(drop=True), pd.DataFrame(corr)], axis=1))
            print("[%s] %d images claires corrigées" % (code, len(claires)), flush=True)

    if args.toutes_claires:
        pd.concat(sorties_c, ignore_index=True).to_csv(PROC / "s2_scenes_claires_corrige.csv", index=False)
    mensuel_c = pd.concat(sorties_m, ignore_index=True)
    fenetres_c = pd.concat(sorties_f, ignore_index=True)
    mensuel_c.to_csv(PROC / "s2_mensuel_corrige.csv", index=False)

    # cumul façon CEVA, avant et après correction, sur les fenêtres de survol
    cols_corr = [c for c in fenetres_c.columns if c.startswith("ulva_corr_s")]
    avant = cal.cumul_mois(fenetres_c, sel["mois_ceva"])
    apres = cal.cumul_mois(fenetres_c.drop(columns=cal.colonnes_seuils(fenetres_c))
                           .rename(columns={c: c.replace("ulva_corr_", "ulva_ha_") for c in cols_corr}),
                           sel["mois_ceva"])
    annuel = apres.merge(ceva[["baie", "annee", "surface_ha"]], on=["baie", "annee"], how="left")
    annuel.to_csv(PROC / "s2_annuel_corrige.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 2)
    print("\nAccord avec le CEVA, fenêtres ± %d j, AVANT correction :" % sel["tolerance_jours"])
    print(cal.comparer_ceva(avant, ceva).to_string(index=False))
    print("\nAPRÈS correction (végétation permanente et liseré exclus) :")
    print(cal.comparer_ceva(apres, ceva).to_string(index=False))
    col = "ulva_ha_s%03d" % round(aff["seuil_ndvi"] * 100)
    print("\nPar baie après correction, seuil %.2f :" % aff["seuil_ndvi"])
    print(cal.correlation_par_baie(apres, ceva, col).to_string())


if __name__ == "__main__":
    main()
