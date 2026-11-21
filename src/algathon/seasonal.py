"""Modèle saisonnier : prévoir la surface cumulée d'ulves de l'année (CEVA) par plage.

Prévision émise le 1er juin de l'année Y (avant le pic de juillet-août), à partir de :
- hiver (1er nov. Y-1 -> 31 mars Y) : température, pluie, jours de coup de vent, rayonnement
- printemps (avril-mai Y) : température, pluie, rayonnement, vent
- débits des rivières à < 25 km (indice = débit / normale mensuelle de la station), hiver et printemps
- nitrates des rivières à < 25 km (médiane hiver et printemps) et flux relatif = nitrates x indice de débit
- report : écart de l'année précédente (stock d'ulves qui passe l'hiver)
- élevages ICPE à < 20 km : agissent via les nitrates (élasticité estimée entre plages)

Le modèle prédit l'ÉCART à la moyenne habituelle de la plage (anomalie de log-surface) ;
la prévision = niveau habituel + écart prédit.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ingest.regional import haversine_km

STORM_KMH = 55  # vent max journalier MERRA-2 (km/h) : ~5 % des jours d'hiver
RADIUS_KM = 25
LIVESTOCK_RADIUS_KM = 20

WEATHER = ["w_temp", "w_precip", "w_storm_days", "w_rad", "autumn_temp",
           "sp_temp", "sp_precip", "sp_rad", "sp_wind"]
HYDRO = ["q_winter", "q_spring", "no3_winter", "no3_spring", "nflux_spring"]
DYNAMIC = WEATHER + HYDRO
# Sélection par validation croisée (une année laissée de côté) : avec 21 années, les 15 variables
# sur-apprennent ; seul le report a un effet robuste (corr. des écarts ~0.25). On garde en plus le flux
# de nitrates, le débit et les températures de printemps/automne (meilleur score régional, 0.18),
# ce qui permet les scénarios « nitrates ». La météo d'hiver (MERRA-2, ~50 km) n'apportait rien.
FEATURES = ["prev_anom", "nflux_spring_a", "q_spring_a", "sp_temp_a", "autumn_temp_a"]
LABELS = {
    "w_temp": "T° hiver", "w_precip": "pluie hiver", "w_storm_days": "coups de vent hiver",
    "w_rad": "ensoleillement hiver", "autumn_temp": "T° automne précédent", "sp_temp": "T° printemps",
    "sp_precip": "pluie printemps", "sp_rad": "ensoleillement printemps", "sp_wind": "vent printemps",
    "q_winter": "débit rivières hiver", "q_spring": "débit rivières printemps",
    "no3_winter": "nitrates hiver", "no3_spring": "nitrates printemps",
    "nflux_spring": "flux de nitrates printemps", "prev": "année précédente",
}


def _between(df, lo, hi):
    return df[(df["date"] >= lo) & (df["date"] <= hi)]


def weather_features(weather: pd.DataFrame, years) -> pd.DataFrame:
    rows = []
    for site, w in weather.groupby("site"):
        for y in years:
            win = _between(w, pd.Timestamp(y - 1, 11, 1), pd.Timestamp(y, 3, 31))
            aut = _between(w, pd.Timestamp(y - 1, 9, 1), pd.Timestamp(y - 1, 10, 31))
            spr = _between(w, pd.Timestamp(y, 4, 1), pd.Timestamp(y, 5, 31))
            if len(win) < 120 or len(spr) < 55:
                continue
            rows.append({"site": site, "year": y,
                         "w_temp": win["temperature_2m_mean"].mean(),
                         "w_precip": win["precipitation_sum"].sum(),
                         "w_storm_days": (win["wind_speed_10m_max"] >= STORM_KMH).sum(),
                         "w_rad": win["shortwave_radiation_sum"].sum(),
                         "autumn_temp": aut["temperature_2m_mean"].mean(),
                         "sp_temp": spr["temperature_2m_mean"].mean(),
                         "sp_precip": spr["precipitation_sum"].sum(),
                         "sp_rad": spr["shortwave_radiation_sum"].sum(),
                         "sp_wind": spr["wind_speed_10m_max"].mean()})
    return pd.DataFrame(rows)


def _near(sites, lon, lat, radius):
    return {s.site: np.where(haversine_km(lon, lat, s.lon, s.lat) <= radius)[0] for s in sites.itertuples()}


def hydro_features(sites, flows: pd.DataFrame, no3: pd.DataFrame, years) -> pd.DataFrame:
    # indice de débit : débit mensuel / normale du même mois à la station
    f = flows.copy()
    f["month"] = f["date"].dt.month
    f["q_idx"] = f["q_m3s"] / f.groupby(["code_station", "month"])["q_m3s"].transform("mean")
    fst = f.groupby("code_station")[["lon", "lat"]].first()
    nst = no3.groupby("code_station")[["longitude", "latitude"]].first()
    near_q = _near(sites, fst["lon"].values, fst["lat"].values, RADIUS_KM)
    near_n = _near(sites, nst["longitude"].values, nst["latitude"].values, RADIUS_KM)
    rows = []
    for s in sites.itertuples():
        fq = f[f["code_station"].isin(fst.index[near_q[s.site]])]
        fn = no3[no3["code_station"].isin(nst.index[near_n[s.site]])]
        for y in years:
            qw = _between(fq, pd.Timestamp(y - 1, 11, 1), pd.Timestamp(y, 3, 31))["q_idx"].mean()
            qs = _between(fq, pd.Timestamp(y, 4, 1), pd.Timestamp(y, 5, 31))["q_idx"].mean()
            nw = _between(fn, pd.Timestamp(y - 1, 11, 1), pd.Timestamp(y, 3, 31))["resultat"].median()
            ns = _between(fn, pd.Timestamp(y, 2, 1), pd.Timestamp(y, 5, 31))["resultat"].median()
            rows.append({"site": s.site, "year": y, "q_winter": qw, "q_spring": qs,
                         "no3_winter": nw, "no3_spring": ns, "nflux_spring": ns * qs,
                         "n_q_stations": len(near_q[s.site]), "n_no3_stations": len(near_n[s.site])})
    return pd.DataFrame(rows)


def livestock_features(sites, lv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in sites.itertuples():
        d = haversine_km(s.lon, s.lat, lv["longitude"].values, lv["latitude"].values)
        near = lv[d <= LIVESTOCK_RADIUS_KM]
        rows.append({"site": s.site, "n_porcs": int(near["porcs"].sum()),
                     "n_bovins": int(near["bovins"].sum()), "n_volailles": int(near["volailles"].sum())})
    out = pd.DataFrame(rows)
    out["n_elevages"] = out[["n_porcs", "n_bovins", "n_volailles"]].sum(axis=1)
    return out


def build_table(sites, ceva_annual, weather, flows, no3, lv, years) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([sites["site"], years], names=["site", "year"]).to_frame(index=False)
    t = (grid.merge(sites[["site", "lat", "lon", "area_ha", "dep", "secteur"]], on="site")
             .merge(weather_features(weather, years), on=["site", "year"], how="left")
             .merge(hydro_features(sites, flows, no3, years), on=["site", "year"], how="left")
             .merge(livestock_features(sites, lv), on="site", how="left")
             .merge(ceva_annual[["site", "year", "cumul_ha"]], on=["site", "year"], how="left"))
    t["y_log"] = np.log1p(t["cumul_ha"])
    return t


def add_anomalies(t: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Anomalies des variables par rapport à la moyenne de la plage (calculée sur ref)."""
    ref = t if ref is None else ref
    t = t.copy()
    for c in DYNAMIC:
        mean = ref.groupby("site")[c].mean()
        t[c + "_a"] = (t[c] - t["site"].map(mean)).fillna(0.0)  # pas de station -> pas d'anomalie
    return t


def livestock_elasticity(t: pd.DataFrame) -> float:
    """Élasticité nitrates / élevages estimée entre plages : d log(NO3) / d log(1 + élevages)."""
    s = t.groupby("site").agg(no3=("no3_spring", "mean"), lv=("n_elevages", "first")).dropna()
    s = s[s["no3"] > 0]
    x, y = np.log1p(s["lv"]), np.log(s["no3"])
    return float(np.polyfit(x, y, 1)[0])


def _model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))


def _prepare(train: pd.DataFrame, rows: pd.DataFrame, prev_source: pd.DataFrame):
    """site_hist et prev_anom calculés uniquement à partir de train."""
    hist = train.groupby("site")["y_log"].mean()
    rows = rows.copy()
    rows["site_hist"] = rows["site"].map(hist)
    prev = prev_source.set_index(["site", "year"])["y_log"]
    key = list(zip(rows["site"], rows["year"] - 1))
    rows["prev_anom"] = (pd.Series(prev.reindex(key).values, index=rows.index) - rows["site_hist"]).fillna(0.0)
    return rows


def cross_validate(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chaque année est prédite par un modèle entraîné sans elle (niveau de plage compris)."""
    obs = t.dropna(subset=["y_log"]).copy()
    preds = []
    for y in sorted(obs["year"].unique()):
        tr_raw, te_raw = obs[obs["year"] != y], obs[obs["year"] == y]
        tr = add_anomalies(_prepare(tr_raw, tr_raw, obs), tr_raw)
        te = add_anomalies(_prepare(tr_raw, te_raw, obs), tr_raw)
        m = _model().fit(tr[FEATURES], tr["y_log"] - tr["site_hist"])
        te["pred_log"] = te["site_hist"] + m.predict(te[FEATURES])
        preds.append(te)
    p = pd.concat(preds).dropna(subset=["site_hist"])
    a_true, a_pred = p["y_log"] - p["site_hist"], p["pred_log"] - p["site_hist"]
    reg_true = a_true.groupby(p["year"]).mean()
    reg_pred = a_pred.groupby(p["year"]).mean()
    per_year = p.groupby("year").apply(lambda g: spearmanr(g["y_log"], g["pred_log"]).statistic,
                                       include_groups=False)
    metrics = {
        "r2_log_model": r2_score(p["y_log"], p["pred_log"]),
        "r2_log_baseline_site_mean": r2_score(p["y_log"], p["site_hist"]),
        "corr_anomalies": float(np.corrcoef(a_true, a_pred)[0, 1]),
        "corr_regional_years": float(np.corrcoef(reg_true, reg_pred)[0, 1]),
        "spearman_sites_per_year": float(per_year.mean()),
        "resid_std_log": float((p["y_log"] - p["pred_log"]).std()),
        "n": len(p),
    }
    return p, metrics


def fit_final(t: pd.DataFrame):
    obs = t.dropna(subset=["y_log"])
    tr = add_anomalies(_prepare(obs, obs, obs), obs)
    m = _model().fit(tr[FEATURES], tr["y_log"] - tr["site_hist"])
    coefs = pd.Series(m[-1].coef_, index=FEATURES)  # coefficients standardisés (comparables)
    return m, obs.groupby("site")["y_log"].mean(), coefs


def scenario_anomalies(t: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Hivers/printemps types (quantiles de l'historique de chaque plage), en anomalies."""
    def q(c, p):
        g = t.groupby("site")[c]
        return g.quantile(p) - g.mean()

    mild = {"w_temp": .85, "w_storm_days": .15, "w_precip": .7, "w_rad": .7, "autumn_temp": .85,
            "sp_temp": .8, "sp_precip": .6, "sp_rad": .75, "sp_wind": .2, "q_winter": .7, "q_spring": .7}
    harsh = {"w_temp": .15, "w_storm_days": .85, "w_precip": .5, "w_rad": .3, "autumn_temp": .15,
             "sp_temp": .2, "sp_precip": .2, "sp_rad": .4, "sp_wind": .8, "q_winter": .3, "q_spring": .2}
    out = {"Hiver et printemps médians": {c: q(c, .5) for c in mild}}
    out["Hiver doux et calme, printemps humide"] = {c: q(c, p) for c, p in mild.items()}
    out["Hiver tempétueux, printemps sec"] = {c: q(c, p) for c, p in harsh.items()}
    return out


# ----------------------------------------------------------------------------- prévision
def _contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Contribution de chaque variable à l'écart prédit (modèle linéaire -> exact)."""
    sc, rg = model[0], model[-1]
    return ((X[FEATURES] - sc.mean_) / sc.scale_) * rg.coef_


def top_factors(contrib: pd.DataFrame, n: int = 3) -> pd.Series:
    """Texte des n facteurs les plus influents par ligne, ex. « débit rivières printemps ↑ (+0.21) »."""
    def fmt(row):
        top = row.abs().sort_values(ascending=False).index[:n]
        return " · ".join(f"{LABELS[c.removesuffix('_a').removesuffix('_anom')]} "
                          f"{'↑' if row[c] > 0 else '↓'} ({row[c]:+.2f})" for c in top)
    return contrib.apply(fmt, axis=1)


def hindcast(bundle: dict, t: pd.DataFrame, years) -> pd.DataFrame:
    """Prévisions avec la météo/hydrologie observées, en chaînant le report d'une année sur l'autre."""
    m, hist, means = bundle["model"], bundle["site_hist"], bundle["site_means"]
    prev = t.dropna(subset=["y_log"]).set_index(["site", "year"])["y_log"].to_dict()
    out = []
    for y in years:
        X = t[t["year"] == y].copy()
        X["site_hist"] = X["site"].map(hist)
        X = X.dropna(subset=["site_hist"])
        for c in DYNAMIC:
            X[c + "_a"] = (X[c] - X["site"].map(means[c])).fillna(0.0)
        X["prev_anom"] = [prev.get((s, y - 1), np.nan) for s in X["site"]]
        X["prev_anom"] = (X["prev_anom"] - X["site_hist"]).fillna(0.0)
        X["pred_log"] = X["site_hist"] + m.predict(X[FEATURES])
        X["facteurs"] = top_factors(_contributions(m, X)).values
        prev.update({(s, y): v for s, v in zip(X["site"], X["pred_log"])})
        out.append(X)
    return pd.concat(out)


def predict_scenario(bundle: dict, scenario: str, r_no3: float = 0.0, r_lv: float = 0.0) -> pd.DataFrame:
    """Prévision pour l'année cible : scénario météo/débits + variation des nitrates et des élevages (%)."""
    m, hist, means = bundle["model"], bundle["site_hist"], bundle["site_means"]
    last, lv = bundle["last_obs"], bundle["n_elevages"]
    X = pd.DataFrame({"site": hist.index, "site_hist": hist.values})
    for c, anom in bundle["scenarios"][scenario].items():
        X[c + "_a"] = X["site"].map(anom).fillna(0.0)
    # élevages -> nitrates via l'élasticité estimée entre plages
    n_lv = X["site"].map(lv).fillna(0)
    lv_factor = ((1 + n_lv * (1 + r_lv / 100)) / (1 + n_lv)) ** bundle["elasticity"]
    factor = (1 + r_no3 / 100) * lv_factor
    for c in ["no3_winter", "no3_spring"]:
        X[c + "_a"] = (X["site"].map(last[c]) * factor - X["site"].map(means[c])).fillna(0.0)
    q_sp = X["site"].map(means["q_spring"]) + X["q_spring_a"]
    X["nflux_spring_a"] = (X["site"].map(last["no3_spring"]) * factor * q_sp
                           - X["site"].map(means["nflux_spring"])).fillna(0.0)
    X["prev_anom"] = X["site"].map(bundle["prev_anom"]).fillna(0.0)
    pred_log = X["site_hist"] + m.predict(X[FEATURES])
    s = bundle["resid_std_log"]
    return pd.DataFrame({"site": X["site"], "ha": np.expm1(pred_log).clip(0),
                         "ha_lo": np.expm1(pred_log - s).clip(0), "ha_hi": np.expm1(pred_log + s).clip(0),
                         "facteurs": top_factors(_contributions(m, X)).values,
                         "nitrate_factor": factor.values})
