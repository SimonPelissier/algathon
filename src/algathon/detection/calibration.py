"""Des scènes Sentinel-2 aux séries mensuelles et annuelles, et comparaison au CEVA.

1. Scènes gardées : au moins `part_valide_min` de la zone sans nuage ni trou.
   Si la même date existe sur deux tuiles, on garde la plus complète.
2. Marée : les ulves ne se voient qu'à marée basse. Pour chaque baie et chaque mois,
   on garde la scène où la surface émergée (valide et non couverte d'eau) est la plus grande.
3. Cumul « façon CEVA » : somme des mois des survols (mai, juillet, septembre). Le CEVA
   additionne de même ses trois inventaires. L'année est inconnue si un des mois manque.
4. Comparaison aux cumuls CEVA, par seuil NDVI :
   - spearman_global : toutes baies confondues (dominé par l'écart de taille entre baies)
   - spearman_intra  : moyenne, sur les baies, de la corrélation entre années.
     C'est ce qui compte pour une prévision : le satellite suit-il les bonnes et mauvaises années ?
   - ratio_median    : satellite / CEVA (échelle ; le CEVA mesure une surface « en lame »)
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def colonnes_seuils(df: pd.DataFrame) -> List[str]:
    return sorted(c for c in df.columns if c.startswith("ulva_ha_s"))


def selection_basse_mer(scenes: pd.DataFrame, part_valide_min: float = 0.9) -> pd.DataFrame:
    """Une scène par (baie, année, mois) : la plus basse mer parmi les scènes claires."""
    s = scenes[scenes["zone_ha"] > 0].copy()
    s["part_valide"] = s["valid_ha"] / s["zone_ha"]
    s = s[s["part_valide"] >= part_valide_min]
    s = s.sort_values("valid_ha").drop_duplicates(["baie", "date"], keep="last")
    s["emergee_ha"] = s["valid_ha"] - s["eau_ha"]
    d = pd.to_datetime(s["date"])
    s["annee"], s["mois"] = d.dt.year, d.dt.month
    s["n_scenes_claires"] = s.groupby(["baie", "annee", "mois"])["date"].transform("count")
    idx = s.groupby(["baie", "annee", "mois"])["emergee_ha"].idxmax()
    return s.loc[idx].sort_values(["baie", "annee", "mois"]).reset_index(drop=True)


def cumul_mois(selection: pd.DataFrame, mois: Sequence[int]) -> pd.DataFrame:
    """Somme des surfaces sur les mois donnés, par baie et par année (NaN si un mois manque)."""
    cols = colonnes_seuils(selection)
    s = selection[selection["mois"].isin(mois)]
    g = s.groupby(["baie", "annee"])
    out = g[cols].sum()
    out[cols] = out[cols].where(g["mois"].nunique() == len(mois))
    out["n_mois"] = g["mois"].nunique()
    return out.reset_index()


def comparer_ceva(cumuls: pd.DataFrame, ceva: pd.DataFrame, min_annees: int = 4) -> pd.DataFrame:
    """Accord entre cumuls satellite et cumuls CEVA, pour chaque seuil NDVI."""
    m = cumuls.merge(ceva[["baie", "annee", "surface_ha"]], on=["baie", "annee"])
    lignes = []
    for c in colonnes_seuils(cumuls):
        d = m.dropna(subset=[c, "surface_ha"])
        intra = []
        for _, g in d.groupby("baie"):
            if len(g) >= min_annees and g[c].nunique() > 1 and g["surface_ha"].nunique() > 1:
                intra.append(g[c].corr(g["surface_ha"], method="spearman"))
        pos = d[d["surface_ha"] > 0]
        lignes.append({
            "seuil_ndvi": int(c[-3:]) / 100.0,
            "n": len(d),
            "spearman_global": d[c].corr(d["surface_ha"], method="spearman"),
            "spearman_intra": float(np.mean(intra)) if intra else np.nan,
            "n_baies_intra": len(intra),
            "ratio_median": float((pos[c] / pos["surface_ha"]).median()) if len(pos) else np.nan,
        })
    return pd.DataFrame(lignes)


def correlation_par_baie(cumuls: pd.DataFrame, ceva: pd.DataFrame, colonne: str) -> pd.DataFrame:
    m = cumuls.merge(ceva[["baie", "annee", "surface_ha"]], on=["baie", "annee"]).dropna(
        subset=[colonne, "surface_ha"])
    return m.groupby("baie").apply(lambda g: pd.Series({
        "n": len(g),
        "spearman": g[colonne].corr(g["surface_ha"], method="spearman"),
        "ratio_median": (g[colonne] / g["surface_ha"].where(g["surface_ha"] > 0)).median(),
    }))


def selection_fenetres(scenes: pd.DataFrame, mois: Sequence[int], tolerance_j: int = 15,
                       part_valide_min: float = 0.9) -> pd.DataFrame:
    """Comme `selection_basse_mer`, mais pour chaque mois des survols CEVA on accepte les
    scènes de ± `tolerance_j` jours autour du mois (les dates de survol varient d'une année
    à l'autre). Une ligne par (baie, année, mois de survol)."""
    s = scenes[scenes["zone_ha"] > 0].copy()
    s["part_valide"] = s["valid_ha"] / s["zone_ha"]
    s = s[s["part_valide"] >= part_valide_min]
    s = s.sort_values("valid_ha").drop_duplicates(["baie", "date"], keep="last")
    s["emergee_ha"] = s["valid_ha"] - s["eau_ha"]
    d = pd.to_datetime(s["date"])
    lignes = []
    for m in mois:
        for annee in sorted(d.dt.year.unique()):
            debut = pd.Timestamp(annee, m, 1) - pd.Timedelta(days=tolerance_j)
            fin = pd.Timestamp(annee, m, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=tolerance_j)
            f = s[(d >= debut) & (d <= fin)].copy()
            if f.empty:
                continue
            f["annee"], f["mois"] = annee, m
            f["n_scenes_claires"] = f.groupby("baie")["date"].transform("count")
            lignes.append(f.loc[f.groupby("baie")["emergee_ha"].idxmax()])
    return pd.concat(lignes).sort_values(["baie", "annee", "mois"]).reset_index(drop=True)
