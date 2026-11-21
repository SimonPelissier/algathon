"""Prédiction astronomique de la marée à Brest jusqu'à fin 2028 (analyse harmonique).

La marée est déterministe : on ajuste par moindres carrés ~30 ondes (M2, S2, N2, K1, O1...) sur
3 ans de hauteurs horaires observées (SHOM), puis on prolonge. Vérification : ajustement sur les
2 dernières années, prédiction de l'année d'avant, comparaison des coefficients journaliers.

  python scripts/predict_tides.py
Sortie : data/processed/tides/brest_pred.csv (date, coef prédit), tides/validation.json
"""
import json

import numpy as np
import pandas as pd
from _common import ROOT

from algathon import daily as Dl

# vitesses angulaires (degrés / heure)
WAVES = {"M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K2": 30.0821373, "K1": 15.0410686,
         "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609, "2N2": 27.8953548, "MU2": 27.9682084,
         "NU2": 28.5125831, "L2": 29.5284789, "T2": 29.9589333, "LAM2": 29.4556253, "2SM2": 31.0158958,
         "M4": 57.9682084, "MS4": 58.9841042, "MN4": 57.4238337, "S4": 60.0, "M3": 43.4761563,
         "MK3": 44.0251729, "M6": 86.9523127, "2MS6": 87.9682084, "2MN6": 86.4079380, "M8": 115.9364166,
         "MSF": 1.0158958, "MF": 1.0980331, "MM": 0.5443747, "SA": 0.0410686, "SSA": 0.0821373}
T0 = pd.Timestamp("2000-01-01")


def design(idx: pd.DatetimeIndex) -> np.ndarray:
    t = ((idx - T0) / pd.Timedelta(hours=1)).values.astype(float)
    cols = [np.ones_like(t)]
    for w in WAVES.values():
        a = np.radians(w) * t
        cols += [np.cos(a), np.sin(a)]
    return np.column_stack(cols)


def fit(h: pd.Series) -> np.ndarray:
    h = h.dropna()
    beta, *_ = np.linalg.lstsq(design(h.index), h.values, rcond=None)
    return beta


def predict(beta, start, end) -> pd.Series:
    idx = pd.date_range(start, end, freq="1h")
    return pd.Series(design(idx) @ beta, index=idx)


def coef(h: pd.Series) -> pd.Series:
    return Dl.tide_daily(h, gauge_is_brest=True)["coef"]


if __name__ == "__main__":
    out = ROOT / "data/processed/tides"
    out.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    print("marégraphe de Brest, hauteurs horaires 2022-11 -> aujourd'hui", flush=True)
    h = Dl.tide_hourly(Dl.BREST, "2022-11-01", today.strftime("%Y-%m-%d"))
    split = today - pd.DateOffset(years=2)

    # vérification : ajustement sur les 2 dernières années, prédiction de l'année précédente
    b = fit(h[h.index >= split])
    past = h[(h.index < split) & (h.index >= split - pd.DateOffset(years=1))]
    obs_c, pred_c = coef(past), coef(predict(b, past.index.min(), past.index.max()))
    both = pd.concat([obs_c.rename("obs"), pred_c.rename("pred")], axis=1).dropna()
    err = (both["pred"] - both["obs"]).abs()
    val = {"periode_test": f"{past.index.min():%Y-%m-%d} -> {past.index.max():%Y-%m-%d}",
           "n_jours": int(len(both)), "erreur_moyenne_coef": float(err.mean()),
           "erreur_p90_coef": float(err.quantile(0.9)), "correlation": float(both.corr().iloc[0, 1]),
           "n_ondes": len(WAVES)}
    print(json.dumps(val, indent=1, ensure_ascii=False))

    # prédiction finale : ajustement sur toute la série
    b = fit(h)
    p = predict(b, today - pd.Timedelta(days=3), "2028-12-31 23:00")
    c = coef(p).rename("coef_pred").to_frame()
    c.index.name = "date"
    c.to_csv(out / "brest_pred.csv")
    (out / "validation.json").write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
    print(c.loc["2027-03-01":"2027-03-31"].T.to_string())
