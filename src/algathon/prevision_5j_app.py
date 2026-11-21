"""Prépare les résultats du système de prévision à 5 jours pour l'application (onglet dédié).

Source : sorties de scripts/prevision_5j_experimentale.py (version « corrige » : surfaces
satellite corrigées des algues fixées), dans data/processed/prevision_5j/.
Destination : app/data/prevision_5j/ (petits CSV lus par app/streamlit_app.py).

Fichiers écrits :
  courbes.csv     baie, annee, date_cible, visible_ha, stock_ha, stock_equiv_ceva_ha
  evaluation.csv  baie, annee, date, observe_ha, prevu_ha, persistance_ha
  scores.csv      une ligne par baie : prévisions évaluées, erreur moyenne, part à ±50 %, gain
  cumuls.csv      baie, annee, cumul_prevu_ha (échelle CEVA), ceva_ha
  baies.csv       baie, nom, sites CEVA (noms), facteur d'échelle satellite → CEVA
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "corrige"


def _a_moins_de_50(prevu: pd.Series, observe: pd.Series) -> pd.Series:
    return (prevu >= observe / 1.5) & (prevu <= observe * 1.5)


def scores_par_baie(ev: pd.DataFrame) -> pd.DataFrame:
    """Qualité de la prévision faite 5 jours avant chaque image claire, par baie et au total."""
    def ligne(g):
        err_log = (np.log1p(g["prevu_ha"]) - np.log1p(g["observe_ha"])).abs().mean()
        err_log_p = (np.log1p(g["persistance_ha"]) - np.log1p(g["observe_ha"])).abs().mean()
        return pd.Series({
            "previsions": len(g),
            "erreur_moyenne_ha": (g["prevu_ha"] - g["observe_ha"]).abs().mean(),
            "part_a_50pct": 100 * _a_moins_de_50(g["prevu_ha"], g["observe_ha"]).mean(),
            "gain_vs_persistance_pct": 100 * (1 - err_log / err_log_p) if err_log_p > 0 else np.nan,
        })
    par_baie = ev.groupby("baie").apply(ligne)
    par_baie.loc["toutes"] = ligne(ev)
    return par_baie.reset_index().rename(columns={"index": "baie"})


def facteurs_ceva(cumuls: pd.DataFrame) -> pd.Series:
    """Facteur satellite → CEVA par baie : total CEVA / total prévu sur les années communes."""
    c = cumuls.dropna(subset=["surface_ha", "cumul_prevu"])
    return c.groupby("baie")["surface_ha"].sum() / c.groupby("baie")["cumul_prevu"].sum()


def exporter(source: Path, dest: Path, config: dict, noms_sites: dict) -> dict:
    """Lit les sorties du système à 5 jours et écrit les CSV de l'application. Renvoie un résumé."""
    courbes = pd.read_csv(source / ("%s_sys5j_courbes.csv" % VERSION), parse_dates=["date_cible"])
    ev = pd.read_csv(source / ("%s_sys5j_eval.csv" % VERSION), parse_dates=["date1"])
    cumuls = pd.read_csv(source / ("%s_sys5j_cumuls.csv" % VERSION))
    fac = facteurs_ceva(cumuls)

    out_c = pd.DataFrame({
        "baie": courbes["baie"], "annee": courbes["annee"], "date_cible": courbes["date_cible"].dt.date,
        "visible_ha": courbes["visible_ha"].round(1), "stock_ha": courbes["basse_mer_ha"].round(1),
    })
    out_c["stock_equiv_ceva_ha"] = (courbes["basse_mer_ha"] * courbes["baie"].map(fac)).round(1)

    out_e = pd.DataFrame({
        "baie": ev["baie"], "annee": ev["annee"], "date": ev["date1"].dt.date,
        "observe_ha": np.expm1(ev["l_ulve1"]).round(1), "prevu_ha": np.expm1(ev["p_modele"]).round(1),
        "persistance_ha": np.expm1(ev["p_persistance"]).round(1),
    })
    sc = scores_par_baie(out_e)

    out_cu = pd.DataFrame({"baie": cumuls["baie"], "annee": cumuls["annee"],
                           "cumul_prevu_ha": (cumuls["cumul_prevu"] * cumuls["baie"].map(fac)).round(0),
                           "ceva_ha": cumuls["surface_ha"].round(0)})

    baies = pd.DataFrame([{
        "baie": code, "nom": b["nom"],
        "sites_ceva": " / ".join(noms_sites.get(i, str(i)) for i in b["sites_ceva"]),
        "facteur_satellite_vers_ceva": round(float(fac.get(code, np.nan)), 3),
    } for code, b in config["baies"].items()])

    dest.mkdir(parents=True, exist_ok=True)
    out_c.to_csv(dest / "courbes.csv", index=False)
    out_e.to_csv(dest / "evaluation.csv", index=False)
    sc.round(2).to_csv(dest / "scores.csv", index=False)
    out_cu.to_csv(dest / "cumuls.csv", index=False)
    baies.to_csv(dest / "baies.csv", index=False)
    return {"courbes": len(out_c), "evaluation": len(out_e), "cumuls": len(out_cu),
            "annees": "%d-%d" % (out_c["annee"].min(), out_c["annee"].max())}
