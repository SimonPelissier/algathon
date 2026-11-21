"""Indice quotidien de risque d'échouage : historique 2017 -> aujourd'hui, prévision J+1..J+10, vérification.

  python scripts/build_daily.py
Sorties : data/processed/daily/{secteur}.csv, forecast.csv, validation.json
"""
import json

import numpy as np
import pandas as pd
from _common import ROOT

from algathon import daily as Dl
from algathon.sites import load_config

cfg = load_config(ROOT / "configs/sectors.yaml")
out = ROOT / "data/processed/daily"
out.mkdir(parents=True, exist_ok=True)
sites = pd.read_csv(ROOT / "data/processed/regional/sites.csv")
s2 = pd.read_csv(ROOT / "data/processed/s2/s2_daily_corrected.csv", parse_dates=["date"])
today = pd.Timestamp.today().normalize()
# profil saisonnier (surface du mois / moyenne mai-sept.), mesuré par Sentinel-2 sur les baies pilotes
_m = s2.assign(y=s2["date"].dt.year, m=s2["date"].dt.month).groupby(["site", "y", "m"])["ulva_ha_corr"].mean().reset_index()
_ref = _m[_m["m"].between(5, 9)].groupby(["site", "y"])["ulva_ha_corr"].mean().rename("ref")
_m = _m.join(_ref, on=["site", "y"]).query("ref > 1")
PROFILE = (_m["ulva_ha_corr"] / _m["ref"]).groupby(_m["m"]).median().reindex(range(1, 13)).interpolate().bfill().ffill()
start, hist_end = "2017-01-01", (today - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

# coefficient de marée : marégraphe de Brest (référence française), observé
print("marées : Brest", flush=True)
brest = Dl.tide_daily(Dl.tide_hourly(Dl.BREST, start, today.strftime("%Y-%m-%d")), gauge_is_brest=True)
gauges = {}

hist_all, fc_all = [], []
for name, sec in cfg["sectors"].items():
    if name not in set(s2["sector"]):
        print(f"[{name}] pas encore de série Sentinel-2, ignoré", flush=True)
        continue
    pts = sites[sites["site"].isin(sec["ceva_sites"])]
    lat, lon = pts["lat"].mean(), pts["lon"].mean()
    g = sec["tide_gauge"]
    if g not in gauges:
        print(f"[{name}] marégraphe {g}", flush=True)
        gauges[g] = Dl.tide_daily(Dl.tide_hourly(g, start, today.strftime("%Y-%m-%d")))
    print(f"[{name}] météo observée + prévisions Météo-France", flush=True)
    wh = Dl.weather_history(lat, lon, start, hist_end)
    wf = Dl.weather_forecast(lat, lon, days=10)
    weather = pd.concat([wh, wf[~wf.index.isin(wh.index)]]).sort_index()

    # marée future : marnage prévu (modèle) recalé sur Brest observé sur les jours communs
    tf = Dl.tide_forecast(48.38, -4.50, days=10)
    common = tf.index.intersection(brest.index)
    ratio = (brest.loc[common, "tidal_range_m"] / tf.loc[common, "tidal_range_m"]).median() if len(common) else 1.0
    coef_future = (100 * tf["tidal_range_m"] * ratio / Dl.BREST_SPRING_RANGE_M).round()
    coef = brest["coef"].combine_first(coef_future)

    days = weather.index
    for site in sec["ceva_sites"]:
        level = gauges[g]["level_11utc_m"]
        st_ = Dl.stock_daily(s2[s2["site"] == site], days, level_11utc=level, profile=PROFILE)
        d = weather.copy()
        d["stock_ha"] = st_.values
        d["coef"] = coef.reindex(days)
        d["level_11utc_m"] = level.reindex(days)
        r = Dl.risk(d, sec["seaward_deg"])
        r.insert(0, "site", site)
        r.insert(0, "sector", name)
        r["is_forecast"] = r.index > today
        hist_all.append(r)
    print(f"[{name}] ok ({len(sec['ceva_sites'])} plages)", flush=True)

H = pd.concat(hist_all).reset_index().rename(columns={"index": "date"})
for name, g in H.groupby("sector"):
    g.to_csv(out / f"{name}.csv", index=False)
fc = H[H["date"] >= today - pd.Timedelta(days=1)]
fc.to_csv(out / "forecast.csv", index=False)

# ----------------------------------------------------------------------------- vérification
# Sur les jours d'image Sentinel-2 (mars-oct.), la surface vue sur la plage dépend-elle
# de la marée au passage, du vent de mer et des grandes marées des jours précédents ?
obs = s2[s2["date"].dt.month.between(3, 10)].merge(
    H[["date", "site", "level_11utc_m", "onshore_3j", "coef_3j", "temperature_2m_max"]], on=["date", "site"])
obs = obs.dropna(subset=["level_11utc_m", "onshore_3j", "coef_3j"])
obs["y"] = np.log1p(obs["ulva_ha_corr"])
obs["grp"] = obs["site"] + "_" + obs["date"].dt.strftime("%Y-%m")  # même plage, même mois : stock ~ constant
X_cols = ["level_11utc_m", "onshore_3j", "coef_3j"]
dm = obs[["y"] + X_cols] - obs.groupby("grp")[["y"] + X_cols].transform("mean")
X, y = dm[X_cols].values, dm["y"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - len(X_cols) - obs["grp"].nunique()
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / max(dof, 1))
val = {c: {"effet_log_par_unite": float(b), "t": float(b / s)} for c, b, s in zip(X_cols, beta, se)}
val["n_images"] = int(len(obs))
val["n_plage_mois"] = int(obs["grp"].nunique())
val["lecture"] = {
    "level_11utc_m": "hauteur d'eau au passage du satellite (m) : négatif attendu (algues cachées à marée haute)",
    "onshore_3j": "vent de mer moyen des 3 jours précédents (km/h) : positif attendu (algues poussées à la côte)",
    "coef_3j": "plus grand coefficient de marée des 3 jours précédents : positif attendu (dépôt haut sur l'estran)",
}
(out / "validation.json").write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in val.items() if k != "lecture"}, indent=1, ensure_ascii=False))
print(fc[fc["is_forecast"]].groupby(["date"])["niveau"].max().to_string())
