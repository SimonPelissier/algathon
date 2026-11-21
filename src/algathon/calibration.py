"""Calibration des surfaces d'ulves Sentinel-2 sur les cumuls annuels CEVA.

1. Fond par plage : 25e centile de toutes les dates (végétation dunaire, herbiers, microphytobenthos
   détectés par le seuil NDVI même sans ulves) -> soustrait.
2. Indice annuel : moyenne mai-septembre de la surface corrigée.
3. Cumul « équivalent CEVA » = k x indice, k ajusté sur les années communes (2017-2022), par moindres
   carrés passant par l'origine.
Validation (2017-2022, 3 baies pilotes) : corrélation ~0.86 entre plages-années, ~0.58 d'une année
sur l'autre pour une même plage.
"""
import numpy as np
import pandas as pd


def background(s2: pd.DataFrame, q: float = 0.25) -> pd.Series:
    return s2.groupby("site")["ulva_ha_est"].quantile(q)


def corrected(s2: pd.DataFrame, bg: pd.Series) -> pd.DataFrame:
    s2 = s2.copy()
    s2["ulva_ha_corr"] = (s2["ulva_ha_est"] - s2["site"].map(bg)).clip(lower=0)
    return s2


def annual_index(s2c: pd.DataFrame) -> pd.DataFrame:
    season = s2c[s2c["date"].dt.month.between(5, 9)]
    return (season.groupby(["site", season["date"].dt.year.rename("year")])
                  .agg(s2_index=("ulva_ha_corr", "mean"), n_images=("ulva_ha_corr", "size"))
                  .reset_index())


def fit(annual: pd.DataFrame, ceva: pd.DataFrame) -> dict:
    m = annual.merge(ceva[["site", "year", "cumul_ha"]], on=["site", "year"])
    k = float((m["s2_index"] * m["cumul_ha"]).sum() / (m["s2_index"] ** 2).sum())
    anom = m.assign(a=m.groupby("site")["s2_index"].transform(lambda x: x - x.mean()),
                    b=m.groupby("site")["cumul_ha"].transform(lambda x: x - x.mean()))
    return {"k": k, "n": len(m),
            "pearson": float(m[["s2_index", "cumul_ha"]].corr().iloc[0, 1]),
            "spearman": float(m[["s2_index", "cumul_ha"]].corr("spearman").iloc[0, 1]),
            "intra_site_corr": float(anom[["a", "b"]].corr().iloc[0, 1])}
