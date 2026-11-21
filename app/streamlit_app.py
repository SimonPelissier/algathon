"""Observatoire des marées vertes — Bretagne et façade atlantique.

Lancer :  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algathon.seasonal import LABELS, predict_scenario  # noqa: E402

FIRST_YEAR, LAST_OBS, FORECAST_YEAR, LAST_YEAR = 2002, 2022, 2027, 2028
MONTHS = ["Année entière", "janv.", "févr.", "mars", "avr.", "mai", "juin",
          "juil.", "août", "sept.", "oct.", "nov.", "déc."]
RISK = [  # (seuil ha de cumul annuel, libellé, couleur RGB)
    (50, "Très élevé", [180, 30, 45]),
    (15, "Élevé", [230, 110, 40]),
    (3, "Modéré", [240, 190, 60]),
    (0, "Faible", [90, 170, 110]),
]
METEO_YEAR = {"sp_precip": ("Pluie du printemps (avril-mai)", "mm"), "w_precip": ("Pluie de l'hiver (nov.-mars)", "mm"),
              "q_spring": ("Débit des rivières au printemps", "× normale"),
              "sp_temp": ("Température du printemps", "°C"), "w_temp": ("Température de l'hiver", "°C"),
              "w_storm_days": ("Jours de coup de vent en hiver", "jours"),
              "autumn_temp": ("Température de l'automne précédent", "°C")}
METEO_MONTH = {"precip": ("Pluie du mois", "mm"), "temp": ("Température du mois", "°C"),
               "storm_days": ("Jours de coup de vent du mois", "jours"), "wind": ("Vent max moyen du mois", "km/h")}
WARM = {"sp_temp", "w_temp", "autumn_temp", "w_storm_days", "temp", "storm_days", "wind"}

st.set_page_config(page_title="Observatoire des marées vertes", page_icon="🌊", layout="wide")


# ----------------------------------------------------------------------------- données
@st.cache_data
def load():
    d = {
        "sites": pd.read_csv(DATA / "sites.csv"),
        "ceva": pd.read_csv(DATA / "ceva_annual.csv"),
        "table": pd.read_parquet(DATA / "table.parquet"),
        "cv": pd.read_csv(DATA / "cv_predictions.csv"),
        "hindcast": pd.read_csv(DATA / "hindcast.csv"),
        "metrics": json.loads((DATA / "metrics.json").read_text(encoding="utf-8")),
        "no3": pd.read_parquet(DATA / "no3_station_year.parquet"),
        "no3_m": pd.read_parquet(DATA / "no3_station_month.parquet"),
        "wm": pd.read_parquet(DATA / "weather_site_month.parquet"),
        "livestock": pd.read_csv(DATA / "livestock.csv"),
        "hydro": {}, "hydro_st": {}, "s2": {}, "s2_all": None,
    }
    for p in (DATA / "hydro").glob("*.csv"):
        key = p.stem.replace("_stations", "")
        target = d["hydro_st"] if p.stem.endswith("_stations") else d["hydro"]
        target[key] = pd.read_csv(p, parse_dates=None if p.stem.endswith("_stations") else ["date"])
    daily = DATA / "s2" / "s2_daily_corrected.csv"
    if daily.exists():
        s2 = pd.read_csv(daily, parse_dates=["date"])
        d["s2_all"] = s2
        d["s2"] = {k: g for k, g in s2.groupby("sector")}
    d["s2_annual"] = pd.read_csv(DATA / "s2" / "s2_annual.csv") if (DATA / "s2" / "s2_annual.csv").exists() else None
    dr = DATA / "daily" / "daily_risk.parquet"
    d["daily"] = pd.read_parquet(dr) if dr.exists() else None
    d["daily_val"] = json.loads((DATA / "daily" / "validation.json").read_text(encoding="utf-8"))         if (DATA / "daily" / "validation.json").exists() else None
    tp = DATA / "daily" / "tide_brest_pred.csv"
    d["tide_pred"] = pd.read_csv(tp, parse_dates=["date"]).set_index("date")["coef_pred"] if tp.exists() else None
    tv = DATA / "daily" / "tide_validation.json"
    d["tide_val"] = json.loads(tv.read_text(encoding="utf-8")) if tv.exists() else None
    cal = DATA / "s2" / "calibration.json"
    d["s2_cal"] = json.loads(cal.read_text()) if cal.exists() else {"k": 6.3}
    return d


@st.cache_resource
def load_model():
    return joblib.load(DATA / "model.joblib")


@st.cache_data
def month_profile(_s2: pd.DataFrame) -> pd.Series:
    """Surface présente chaque mois / moyenne mai-sept., médiane des plages-années pilotes (Sentinel-2)."""
    s = _s2.assign(year=_s2["date"].dt.year, month=_s2["date"].dt.month)
    mm = s.groupby(["site", "year", "month"])["ulva_ha_corr"].mean().reset_index()
    ref = mm[mm["month"].between(5, 9)].groupby(["site", "year"])["ulva_ha_corr"].mean()
    mm = mm.join(ref.rename("ref"), on=["site", "year"])
    mm = mm[mm["ref"] > 1]
    prof = (mm["ulva_ha_corr"] / mm["ref"]).groupby(mm["month"]).median().reindex(range(1, 13)).interpolate()
    return prof.fillna(0.0)


def risk_of(ha, scale=1.0):
    for thr, label, color in RISK:
        if ha >= thr * scale:
            return label, color
    return RISK[-1][1], RISK[-1][2]


def forecast(M, year, scenario, r_no3, r_lv):
    p27 = predict_scenario(M, scenario, r_no3, r_lv)
    if year == FORECAST_YEAR:
        p27["source"] = "Prévision 2027 (scénario)"
        return p27
    prev = (p27.set_index("site")["pred_log"] - M["site_hist"]).rename("prev")
    p28 = predict_scenario(M, scenario, r_no3, r_lv, prev_anom=prev, std_factor=1.25)
    p28["source"] = "Prévision 2028 (scénario, 2 ans)"
    return p28


def year_values(D, M, year, scenario, r_no3, r_lv):
    if year <= LAST_OBS:
        v = D["ceva"][D["ceva"]["year"] == year][["site", "cumul_ha"]].rename(columns={"cumul_ha": "ha"})
        v["ha_lo"], v["ha_hi"], v["source"], v["facteurs"] = v["ha"], v["ha"], "Observé (CEVA)", ""
    elif year < FORECAST_YEAR:
        v = D["hindcast"][D["hindcast"]["year"] == year][["site", "pred_ha", "pred_log", "facteurs"]] \
            .rename(columns={"pred_ha": "ha"})
        s = M["resid_std_log"]
        v["ha_lo"], v["ha_hi"] = np.expm1(v["pred_log"] - s).clip(0), np.expm1(v["pred_log"] + s).clip(0)
        v["source"] = "Estimé (modèle + météo réelle)"
        v = v.drop(columns="pred_log")
        if D["s2_annual"] is not None:  # baies pilotes : mesure satellite calibrée sur le CEVA
            sat = D["s2_annual"][(D["s2_annual"]["year"] == year) & (D["s2_annual"]["n_images"] >= 5)]
            sat = sat.set_index("site")["cumul_equiv_ha"]
            hit = v["site"].isin(sat.index)
            v.loc[hit, "ha"] = v.loc[hit, "site"].map(sat).values
            v.loc[hit, "ha_lo"], v.loc[hit, "ha_hi"] = v.loc[hit, "ha"] * 0.6, v.loc[hit, "ha"] * 1.4
            v.loc[hit, "facteurs"] = "mesure Sentinel-2 calibrée sur le CEVA"
    else:
        v = forecast(M, year, scenario, r_no3, r_lv).drop(columns=["pred_log", "nitrate_factor"])
    return v


def to_month(D, v, year, m, prof, k):
    """Cumul annuel -> surface présente en moyenne le mois m ; mesure satellite du mois si elle existe."""
    v = v.copy()
    f = prof.loc[m] / k
    for c in ["ha", "ha_lo", "ha_hi"]:
        v[c] = v[c] * f
    v["source"] = v["source"].str.replace("Observé (CEVA)", "Reconstitué (CEVA annuel × profil)", regex=False)
    if D["s2_all"] is not None and 3 <= m <= 10:  # hiver : soleil bas, détection peu fiable -> pas de mesure
        s = D["s2_all"]
        g = s[(s["date"].dt.year == year) & (s["date"].dt.month == m)].groupby("site")["ulva_ha_corr"]
        sm = g.mean()[g.size() >= 2]  # au moins 2 images dans le mois
        hit = v["site"].isin(sm.index)
        v.loc[hit, "ha"] = v.loc[hit, "site"].map(sm).values
        v.loc[hit, "ha_lo"], v.loc[hit, "ha_hi"] = v.loc[hit, "ha"] * 0.7, v.loc[hit, "ha"] * 1.3
        v.loc[hit, "facteurs"] = "mesure Sentinel-2 du mois (fond retiré)"
    return v


def to_sea(df: pd.DataFrame, radius_m: pd.Series) -> pd.DataFrame:
    """Décale le centre du cercle vers le large (sea_deg, calculé sur le trait de côte OSM) pour que le
    cercle soit posé sur la mer devant la plage, sans dépasser la place disponible en mer."""
    df = df.copy()
    if "sea_deg" not in df:
        return df
    off = np.minimum(radius_m.values + 300, df["sea_room_km"].fillna(0).values * 1000 * 0.6)
    a = np.radians(df["sea_deg"].fillna(0).values)
    df["lat"] = df["lat"] + off * np.cos(a) / 110570
    df["lon"] = df["lon"] + off * np.sin(a) / (111320 * np.cos(np.radians(df["lat"])))
    return df


HEX_KM = 7.0  # rayon des zones hexagonales


def _hex_ids(lon, lat):
    """Hexagone (axial q, r) de chaque point ; projection simple en km autour de 48° N."""
    x, y = np.asarray(lon) * 111.32 * np.cos(np.radians(48)), np.asarray(lat) * 110.57
    q, r = (np.sqrt(3) / 3 * x - y / 3) / HEX_KM, (2 / 3 * y) / HEX_KM
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    rq = np.where((dq > dr) & (dq > ds), -rr - rs, rq)
    rr = np.where(~((dq > dr) & (dq > ds)) & (dr > ds), -rq - rs, rr)
    return rq.astype(int), rr.astype(int)


def _hex_poly(q, r):
    cx, cy = HEX_KM * np.sqrt(3) * (q + r / 2), HEX_KM * 1.5 * r
    ang = np.radians(30 + 60 * np.arange(6))
    xs, ys = cx + HEX_KM * np.cos(ang), cy + HEX_KM * np.sin(ang)
    return [[float(a / (111.32 * np.cos(np.radians(48)))), float(b / 110.57)] for a, b in zip(xs, ys)]


def hex_zones(no3: pd.DataFrame, livestock: pd.DataFrame) -> pd.DataFrame:
    """Nitrates moyens des rivières par zone : blanc (< 10 mg/L) -> jaune vif (>= 50 mg/L)."""
    q, r = _hex_ids(no3["lon"], no3["lat"])
    z = no3.assign(q=q, r=r).groupby(["q", "r"]).agg(no3=("no3", "mean"), n_st=("no3", "size")).reset_index()
    lq, lr = _hex_ids(livestock["longitude"], livestock["latitude"])
    n_lv = pd.Series(1, index=pd.MultiIndex.from_arrays([lq, lr])).groupby(level=[0, 1]).sum()
    z["n_lv"] = [int(n_lv.get((a, b), 0)) for a, b in zip(z["q"], z["r"])]
    t = ((z["no3"] - 10) / 40).clip(0, 1).values
    z["c"] = [[255, int(255 - 40 * v), int(255 - 255 * v), int(70 + 150 * v)] for v in t]
    z["poly"] = [_hex_poly(a, b) for a, b in zip(z["q"], z["r"])]
    return z


LEVEL_TXT = {1: "1 · faible", 2: "2 · modéré", 3: "3 · élevé", 4: "4 · très élevé"}
LEVEL_RGB = {1: [90, 170, 110], 2: [240, 190, 60], 3: [230, 110, 40], 4: [180, 30, 45]}
LEVEL_THR = [2, 8, 20]  # seuils de risque H2S (ha équivalents) entre niveaux, comme scripts/build_daily.py


def level_of(r):
    return 1 + int(np.searchsorted(LEVEL_THR, r, side="right"))


def day_view(D, v, typical, date):
    """Risque d'un jour donné pour les plages pilotes.

    - jour de la série calculée (2017 -> J+10) : indice tel quel (météo et marées observées ou prévues) ;
    - jour de 2027-2028 : risque ATTENDU = stock prévu (prévision annuelle du scénario x profil du mois)
      x marée astronomique prédite (exacte) x chaleur et calme des mêmes dates (±7 j) de 2017-2025.
      La dispersion de la météo passée donne la probabilité d'un niveau >= 3.
    """
    dr = D["daily"]
    if dr is None:
        return None, ""
    g = dr[dr["date"] == date]
    if len(g):
        out = g[["site", "niveau", "risque_h2s", "stock_ha", "coef", "temperature_2m_max", "wind_speed_10m_max"]].copy()
        out["p_eleve"] = np.nan
        src = ("Prévision Météo-France / ECMWF + marées" if g["is_forecast"].any()
               else "Indice calculé (satellite, météo et marées observées)")
        return out, src
    if date.year < FORECAST_YEAR or D["tide_pred"] is None:
        return None, ""
    tp = D["tide_pred"]
    coef, coef3 = tp.get(date, np.nan), tp.rolling(3, min_periods=1).max().get(date, np.nan)
    maree = 0.4 + 0.6 * np.clip((coef3 - 45) / 60, 0, 1) if pd.notna(coef3) else 0.7
    hist = dr[~dr["is_forecast"]]
    dist = (hist["date"].dt.dayofyear - date.dayofyear).abs()
    win = hist[np.minimum(dist, 365 - dist) <= 7]
    # stock attendu = stock satellite habituel de la plage à cette date x (prévision / année typique)
    ratio = (v.set_index("site")["ha"] / typical).replace([np.inf, -np.inf], np.nan)
    rows = []
    for site, w in win.groupby("site"):
        if pd.isna(ratio.get(site, np.nan)):
            continue
        stock = w["stock_ha"].median() * ratio[site]
        r = stock * maree * (0.2 + 0.8 * w["chaleur"]) * w["calme"]
        rows.append({"site": site, "niveau": level_of(r.median()), "risque_h2s": r.median(),
                     "stock_ha": stock, "coef": coef, "temperature_2m_max": w["temperature_2m_max"].median(),
                     "wind_speed_10m_max": w["wind_speed_10m_max"].median(), "p_eleve": (r >= LEVEL_THR[1]).mean()})
    tv = D["tide_val"]
    err = f" (erreur moyenne ±{tv['erreur_moyenne_coef']:.1f} point testée sur un an)" if tv else ""
    return pd.DataFrame(rows), ("Risque attendu : marée astronomique prédite" + err + " · stock satellite habituel "
                                "de la plage × prévision du scénario · météo : 2017-2025 aux mêmes dates (±7 j)")


# ----------------------------------------------------------------------------- interface
D, M = load(), load_model()
K = D["s2_cal"]["k"]
PROF = month_profile(D["s2_all"]) if D["s2_all"] is not None else pd.Series(1.0, index=range(1, 13))

st.sidebar.title("🌊 Marées vertes")
year = st.sidebar.slider("Année", FIRST_YEAR, LAST_YEAR, FORECAST_YEAR)
month = st.sidebar.select_slider("Mois", MONTHS, value="Année entière")
m_idx = MONTHS.index(month)  # 0 = année entière
day = None
if D["daily"] is not None:
    if m_idx and year >= 2017:
        nd = pd.Timestamp(year, m_idx, 1).days_in_month
        dsel = st.sidebar.select_slider("📅 Jour (baies pilotes)", ["Mois entier"] + list(range(1, nd + 1)),
                                        value="Mois entier")
        if dsel != "Mois entier":
            day = pd.Timestamp(year, m_idx, dsel)
    else:
        st.sidebar.select_slider("📅 Jour (baies pilotes)", ["Mois entier", "—"], value="Mois entier", disabled=True)
        st.sidebar.caption("👆 Choisissez un **mois**" + (" et une année ≥ 2017" if year < 2017 else "")
                           + " pour naviguer jour par jour.")
scenario, r_no3, r_lv = next(iter(M["scenarios"])), 0, 0
if year >= FORECAST_YEAR:
    st.sidebar.markdown("**Scénarios 2027-2028**")
    scenario = st.sidebar.radio("Hiver et printemps", list(M["scenarios"]), index=0)
    r_no3 = st.sidebar.slider("Nitrates dans les rivières (%)", -60, 20, 0, step=5)
    r_lv = st.sidebar.slider("Élevages (%)", -60, 20, 0, step=5)
st.sidebar.markdown("**Couches**")
show_sites = st.sidebar.checkbox("Plages (surface d'ulves)", True)
show_no3 = st.sidebar.checkbox("Nitrates des rivières (zones)", True)
show_lv = st.sidebar.checkbox("Densité d'élevages (ICPE)", False)
show_meteo = st.sidebar.checkbox("Météo et débits", False)
METEO = METEO_MONTH if m_idx else METEO_YEAR
meteo_var = st.sidebar.selectbox("Variable météo", list(METEO), format_func=lambda c: METEO[c][0]) if show_meteo else None

v = year_values(D, M, year, scenario, r_no3, r_lv)
typical = np.expm1(M["site_hist"])  # année typique (médiane géométrique 2002-2022)
scale = 1.0
if m_idx:
    v = to_month(D, v, year, m_idx, PROF, K)
    typical = typical * PROF.loc[m_idx] / K
    scale = 1 / K  # seuils de risque ramenés à une surface présente
v = v.merge(D["sites"], on="site")
v["risque"], v["color"] = zip(*v["ha"].map(lambda h: risk_of(h, scale)))
v = v.sort_values("ha", ascending=False)
hist_total = typical.sum()

st.title("Observatoire des marées vertes")
unit_txt = (f"surface d'ulves présente en moyenne en {month} {year} (ha)" if m_idx
            else f"surface cumulée d'ulves par plage en {year} (ha/an)")
st.caption("Bretagne et façade atlantique · " + unit_txt)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Surface présente (moyenne du mois)" if m_idx else "Surface cumulée totale",
          f"{v['ha'].sum():,.0f} ha".replace(",", " "),
          f"{(v['ha'].sum() / hist_total - 1) * 100:+.0f} % vs " + ("même mois, année typique" if m_idx else "année typique"))
c2.metric("Plages à risque très élevé", int((v["risque"] == "Très élevé").sum()))
c3.metric("Plages à risque élevé", int((v["risque"] == "Élevé").sum()))
c4.metric("Source", v["source"].mode().iloc[0])

dv = None
if day is not None:
    dv, dsrc = day_view(D, v, typical, day)
    if dv is None or dv.empty:
        dv = None
        st.warning(f"Pas d'indice quotidien pour le {day:%d/%m/%Y} : il existe de 2017 à aujourd'hui + 10 jours "
                   "(calculé), puis pour 2027-2028 (risque attendu).")
    else:
        dv = dv.merge(D["sites"][["site", "lon", "lat"] + [c for c in ("sea_deg", "sea_room_km")
                                                             if c in D["sites"]]], on="site").sort_values("risque_h2s", ascending=False)
        jour = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"][day.dayofweek]
        st.markdown(f"#### 📅 {jour} {day:%d/%m/%Y} — risque d'échouage et d'H₂S, plages des baies pilotes")
        st.caption(dsrc + f" · coefficient de marée (Brest) : {dv['coef'].iloc[0]:.0f}")
        tab_d = pd.DataFrame({"plage": dv["site"], "niveau": dv["niveau"].map(lambda x: LEVEL_TXT.get(int(x), "")),
                              "indice": dv["risque_h2s"].round(1), "stock d'ulves (ha)": dv["stock_ha"].round(1),
                              "T° max (°C)": dv["temperature_2m_max"].round(0),
                              "vent max (km/h)": dv["wind_speed_10m_max"].round(0)})
        if dv["p_eleve"].notna().any():
            tab_d["proba. niveau ≥ 3"] = (dv["p_eleve"] * 100).round(0).astype(int).astype(str) + " %"
        st.dataframe(tab_d.style.map(lambda x: {"1": "background-color:#d9efdf", "2": "background-color:#fbefc4",
                                                "3": "background-color:#f9cfa9", "4": "background-color:#f1a7a7"}
                                     .get(str(x)[:1], ""), subset=["niveau"]), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- carte
layers = []
if show_lv:
    lv = D["livestock"].copy()
    lv["w"] = lv["porcs"] * 1.0 + lv["volailles"] * 0.7 + lv["bovins"] * 0.5
    layers.append(pdk.Layer("HeatmapLayer", lv, get_position=["longitude", "latitude"], get_weight="w",
                            radius_pixels=40, opacity=0.45,
                            color_range=[[255, 247, 236], [253, 212, 158], [252, 141, 89],
                                         [215, 48, 31], [127, 0, 0]]))
if show_no3:
    future = year >= FORECAST_YEAR
    if m_idx:
        nm = D["no3_m"]
        if future:  # climatologie des 3 dernières années pour ce mois
            last = int(nm["year"].max())
            no3 = (nm[(nm["month"] == m_idx) & (nm["year"] > last - 3)]
                   .groupby("code_station").agg(no3=("no3", "median"), lon=("lon", "first"), lat=("lat", "first"))
                   .reset_index())
            when = f"{month}, normale {last - 2}-{last}"
        else:
            no3 = nm[(nm["year"] == year) & (nm["month"] == m_idx)].copy()
            when = f"{month} {year}"
    else:
        ny = int(D["no3"]["year"].max()) if future else year
        no3 = D["no3"][(D["no3"]["year"] == ny) & (D["no3"]["n"] >= 3)].copy()
        when = f"médiane {ny}"
    if future:
        no3["no3"] = no3["no3"] * (1 + r_no3 / 100)
    zones = hex_zones(no3.dropna(subset=["no3"]), D["livestock"])
    zones["label"] = ("Zone : " + zones["no3"].round(0).astype(int).astype(str) + f" mg NO₃/L en moyenne ({when}), "
                      + zones["n_st"].astype(str) + " point(s) de mesure · " + zones["n_lv"].astype(str)
                      + " élevage(s) classé(s)")
    layers.append(pdk.Layer("PolygonLayer", zones, get_polygon="poly", get_fill_color="c", stroked=True,
                            get_line_color=[255, 255, 255, 120], line_width_min_pixels=1, pickable=True))
if show_meteo:
    name, unit = METEO[meteo_var]
    pos = D["table"].groupby("site")[["lon", "lat"]].first()
    scen = pd.DataFrame(M["scenarios"][scenario])  # écarts du scénario par plage (variables saisonnières)
    if m_idx:
        wm = D["wm"]
        clim_df = wm[(wm["month"] == m_idx) & (wm["year"] <= LAST_OBS)].groupby("site")[meteo_var]
        clim, spread = clim_df.mean(), clim_df.std()
        if year < FORECAST_YEAR:
            cur = wm[(wm["year"] == year) & (wm["month"] == m_idx)].set_index("site")[meteo_var]
            mv = pd.DataFrame({"val": cur})
            mv["anom"] = mv["val"] - clim.reindex(mv.index)
        else:  # mois futur : normale + écart du scénario réparti sur la saison concernée
            winter, spring, autumn = m_idx in (11, 12, 1, 2, 3), m_idx in (4, 5), m_idx in (9, 10)
            col = {"temp": "w_temp" if winter else "sp_temp" if spring else "autumn_temp" if autumn else None,
                   "precip": "w_precip" if winter else "sp_precip" if spring else None,
                   "storm_days": "w_storm_days" if winter else None, "wind": None}[meteo_var]
            div = {"w_precip": 5, "sp_precip": 2, "w_storm_days": 5}.get(col, 1)
            anom = (scen[col] / div) if col in scen else pd.Series(0.0, index=clim.index)
            mv = pd.DataFrame({"anom": anom.reindex(clim.index).fillna(0.0)})
            mv["val"] = clim + mv["anom"]
        mv["spread"] = spread.reindex(mv.index)
        ref_txt = f"normale de {month} 2002-2022"
    else:
        t = D["table"]
        clim = t[t["year"] <= LAST_OBS].groupby("site")[meteo_var].mean()
        if year < FORECAST_YEAR:
            mv = t[t["year"] == year].set_index("site")[[meteo_var]].rename(columns={meteo_var: "val"}).dropna()
            mv["anom"] = mv["val"] - clim.reindex(mv.index)
        else:
            anom = scen[meteo_var] if meteo_var in scen else pd.Series(0.0, index=clim.index)
            mv = pd.DataFrame({"anom": anom.reindex(clim.index).fillna(0.0)})
            mv["val"] = clim + mv["anom"]
        mv["spread"] = t.groupby("site")[meteo_var].std().reindex(mv.index)
        ref_txt = "moyenne 2002-2022"
    mv = mv.join(pos).dropna(subset=["lon", "lat", "val"]).reset_index(names="site")
    sd = mv["spread"].where(mv["spread"] > 0, 1).fillna(1)
    z = (mv["anom"].fillna(0) / sd).clip(-2, 2).values / 2
    hi, lo = ([215, 60, 50], [50, 110, 200]) if meteo_var in WARM else ([40, 100, 200], [200, 140, 50])
    for i, ch in enumerate("rgb"):
        mv[ch] = np.where(z > 0, 240 + (hi[i] - 240) * z, 240 + (lo[i] - 240) * -z).astype(int)
    mv["label"] = (mv["site"] + " — " + name + " : " + mv["val"].round(1).astype(str) + " " + unit
                   + " (" + mv["anom"].fillna(0).round(1).map(lambda x: f"{x:+}") + f" vs {ref_txt})")
    layers.append(pdk.Layer("ScatterplotLayer", mv, get_position=["lon", "lat"], get_radius=7000,
                            get_fill_color="[r, g, b, 150]", stroked=True, get_line_color=[90, 90, 90, 120],
                            line_width_min_pixels=1, pickable=True))
if show_sites:
    sv = v.copy()
    sv["label"] = (sv["site"] + " — " + sv["ha"].round(1).astype(str) + " ha (" + sv["risque"] + ") · "
                   + sv["source"])
    # cercle vert, surface du disque proportionnelle à la surface d'ulves (même échelle année / mois)
    ha_year = sv["ha"] * (K / PROF.max() if m_idx else 1)
    sv["radius"] = 500 * np.sqrt(ha_year.clip(lower=0))
    sv = to_sea(sv, sv["radius"])
    layers.append(pdk.Layer("ScatterplotLayer", sv, get_position=["lon", "lat"], get_radius="radius",
                            radius_min_pixels=3, get_fill_color=[30, 150, 60, 170], stroked=True,
                            get_line_color=[10, 90, 30, 230], line_width_min_pixels=1,
                            pickable=True, auto_highlight=True))
if dv is not None:  # jour choisi : anneau coloré par niveau autour des plages pilotes
    ring = dv.merge(sv[["site", "radius"]], on="site", how="left") if show_sites else dv.assign(radius=2500.0)
    ring["radius"] = ring["radius"].fillna(2500.0)
    ring = to_sea(ring, ring["radius"])
    ring["c"] = ring["niveau"].map(lambda x: LEVEL_RGB.get(int(x), [150, 150, 150]) + [255])
    ring["label"] = ring["site"] + f" — {day:%d/%m/%Y} : niveau " + ring["niveau"].map(lambda x: LEVEL_TXT.get(int(x), ""))
    ring["ring_r"] = ring["radius"] + 900
    layers.append(pdk.Layer("ScatterplotLayer", ring, get_position=["lon", "lat"], get_radius="ring_r",
                            filled=False, stroked=True, get_line_color="c", line_width_min_pixels=4, pickable=True))

st.pydeck_chart(pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                         initial_view_state=pdk.ViewState(latitude=48.0, longitude=-3.2, zoom=6.6, pitch=0),
                         tooltip={"text": "{label}"}), width="stretch", height=560)
st.caption("🟢 Cercles verts : marée verte par plage — plus le cercle est grand, plus la surface d'ulves est grande ("
           + ("présente en moyenne ce mois-ci" if m_idx else "cumul annuel")
           + ") · ⬡ Zones : nitrates des rivières, blanc < 10 mg/L → jaune vif ≥ 50 mg/L (moyenne des points de "
           "mesure de la zone ; la concentration reflète l'azote lessivé par l'eau qui draine les sols)"
           + (" · ⭕ Anneaux : niveau de risque du jour (vert 1 → rouge 4)" if dv is not None else "")
           + " · grands disques (météo) : écart à la normale.")
if m_idx:
    st.caption("Mois : cumul annuel ÷ " + f"{K:.1f}" + " × profil saisonnier mesuré par Sentinel-2 sur les baies "
               "pilotes ; quand le satellite a mesuré la plage au moins 2 fois ce mois-là, c'est sa mesure qui s'affiche. "
               "De novembre à février (soleil bas, peu d'images), la mesure satellite n'est pas utilisée : estimation seule.")

# ----------------------------------------------------------------------------- onglets
tab1, tab5, tab2, tab3, tab4 = st.tabs(["🎯 Plages ciblées", "📅 Risque jour par jour", "📈 Détail d'une plage",
                                         "🛰️ Baies pilotes", "🧠 Modèle"])

with tab1:
    show = v[["site", "secteur", "dep", "ha", "ha_lo", "ha_hi", "risque", "facteurs", "source"]].copy()
    show["année typique"] = show["site"].map(typical)
    show["écart vs typique"] = ((show["ha"] / show["année typique"] - 1).mul(100).round(0)
                                .astype("Int64").astype(str) + " %")
    show = show.rename(columns={"ha": "surface (ha)", "ha_lo": "bas", "ha_hi": "haut", "dep": "dép.",
                                "facteurs": "principaux facteurs (effet sur log-surface)"})
    st.dataframe(show.round(1), width="stretch", hide_index=True, height=420)
    st.download_button("Télécharger la liste (CSV)", show.to_csv(index=False).encode("utf-8"),
                       f"plages_ciblees_{year}_{m_idx:02d}.csv", "text/csv")

with tab5:
    dr = D["daily"]
    if dr is None:
        st.info("Indice quotidien non calculé (scripts/build_daily.py).")
    else:
        LV = {1: "1 · faible", 2: "2 · modéré", 3: "3 · élevé", 4: "4 · très élevé"}
        st.markdown("**Indice expérimental de risque d'échouage et d'H₂S, baies pilotes** — "
                    "stock d'ulves (satellite, corrigé de la marée) × grandes marées × chaleur × calme.")
        bay = st.selectbox("Baie", sorted(dr["sector"].unique()), key="bay_daily")
        b = dr[dr["sector"] == bay]
        fc = b[b["is_forecast"]]
        if len(fc):
            st.markdown(f"**Prévision des 10 prochains jours** (marées SHOM/Open-Meteo, météo Météo-France puis ECMWF)")
            piv = fc.pivot_table(index="site", columns="date", values="niveau").rename(columns=lambda c: c.strftime("%a %d/%m"))
            st.dataframe(piv.map(lambda x: LV.get(int(x), "") if pd.notna(x) else "")
                         .style.map(lambda x: {"1": "background-color:#d9efdf", "2": "background-color:#fbefc4",
                                               "3": "background-color:#f9cfa9", "4": "background-color:#f1a7a7"}
                                    .get(str(x)[:1], "")), width="stretch")
            drivers = fc.groupby("date")[["coef", "temperature_2m_max", "wind_speed_10m_max"]].first().round(0)
            drivers.index = drivers.index.strftime("%a %d/%m")
            st.dataframe(drivers.rename(columns={"coef": "coefficient de marée", "temperature_2m_max": "T° max (°C)",
                                                 "wind_speed_10m_max": "vent max (km/h)"}).T, width="stretch")
        yr = st.slider("Historique : année", 2017, int(b["date"].dt.year.max()), min(year, int(b["date"].dt.year.max())),
                       key="yr_daily")
        hb = b[b["date"].dt.year == yr]
        st.markdown(f"**{yr} — risque H₂S quotidien par plage** (indice, ha équivalents)")
        st.line_chart(hb.pivot_table(index="date", columns="site", values="risque_h2s"))
        st.markdown("**Stock d'ulves estimé (ha) et coefficient de marée**")
        st.line_chart(hb.pivot_table(index="date", columns="site", values="stock_ha").join(
            hb.groupby("date")["coef"].first()))
        val = D["daily_val"]
        if val:
            st.markdown(f"""
**Vérification sur {val['n_images']} images Sentinel-2** (même plage, même mois) :
- hauteur d'eau au passage du satellite : **{(np.exp(val['level_11utc_m']['effet_log_par_unite']) - 1) * 100:+.0f} %**
  de surface vue par mètre d'eau (t = {val['level_11utc_m']['t']:.0f}) → le stock est corrigé de la marée ;
- grandes marées des 3 jours précédents : **{(np.exp(10 * val['coef_3j']['effet_log_par_unite']) - 1) * 100:+.1f} %**
  par 10 points de coefficient (t = {val['coef_3j']['t']:.1f}) → retenu, effet modeste ;
- vent de mer des 3 jours précédents : **{(np.exp(10 * val['onshore_3j']['effet_log_par_unite']) - 1) * 100:+.0f} %**
  par 10 km/h (t = {val['onshore_3j']['t']:.0f}) → contraire à l'hypothèse « le vent pousse les algues » :
  il n'entre pas dans le potentiel ; il intervient dans le risque H₂S (le vent disperse le gaz).

*Indice non encore validé jour par jour : il manque une mesure quotidienne du danger
(piste : capteurs H₂S d'Air Breizh, tonnages de ramassage des communes).*
""")

with tab2:
    site = st.selectbox("Plage", v["site"].tolist())
    obs = D["ceva"][D["ceva"]["site"] == site].set_index("year")["cumul_ha"].rename("Observé (CEVA)")
    cv = (D["cv"][D["cv"]["site"] == site].set_index("year")["pred_log"].pipe(np.expm1)
          .rename("Modèle (validation croisée)"))
    hc = D["hindcast"][D["hindcast"]["site"] == site].set_index("year")["pred_ha"].rename("Estimé 2023-2026")
    sat = (D["s2_annual"][D["s2_annual"]["site"] == site].set_index("year")["cumul_equiv_ha"]
           .rename("Satellite calibré") if D["s2_annual"] is not None else pd.Series(dtype=float, name="Satellite calibré"))
    fut = {y: forecast(M, y, scenario, r_no3, r_lv).set_index("site") for y in (FORECAST_YEAR, LAST_YEAR)}
    pf = pd.Series({y: f.loc[site, "ha"] for y, f in fut.items() if site in f.index}, name="Prévision (scénario)")
    st.markdown(f"**{site} — surface cumulée par an (ha)**")
    st.line_chart(pd.concat([obs, cv, hc, sat, pf], axis=1))
    for y, f in fut.items():
        if site in f.index:
            st.markdown(f"**{y} — {scenario}** : {f.loc[site, 'ha']:.0f} ha "
                        f"(fourchette {f.loc[site, 'ha_lo']:.0f}–{f.loc[site, 'ha_hi']:.0f}) · {f.loc[site, 'facteurs']}")
    # calendrier mois par mois de l'année choisie
    vy = year_values(D, M, year, scenario, r_no3, r_lv)
    if site in set(vy["site"]):
        base = vy.set_index("site").loc[site, "ha"] / K
        cal_m = pd.DataFrame({"mois": MONTHS[1:], "Estimé (profil saisonnier)": (PROF * base).values})
        if D["s2_all"] is not None:
            s = D["s2_all"]
            sm = s[(s["site"] == site) & (s["date"].dt.year == year)].groupby(s["date"].dt.month)["ulva_ha_corr"].mean()
            cal_m["Mesuré (Sentinel-2)"] = sm.reindex(range(1, 13)).values
        st.markdown(f"**{site} — surface présente mois par mois en {year} (ha)**")
        st.bar_chart(cal_m.set_index("mois").reindex(MONTHS[1:]), stack=False)
    row = D["table"][D["table"]["site"] == site]
    st.markdown("**Conditions de l'hiver et du printemps** (variables du modèle)")
    vars_ = ["q_spring", "no3_spring", "w_temp", "w_storm_days"]
    st.line_chart(row.set_index("year")[vars_].rename(columns={c: LABELS[c] for c in vars_}))

with tab3:
    if not D["s2"]:
        st.info("Séries Sentinel-2 en cours de calcul.")
    else:
        sector = st.selectbox("Baie pilote", sorted(D["s2"]))
        s2 = D["s2"][sector]
        piv = s2.pivot_table(index="date", columns="site", values="ulva_ha_corr")
        st.markdown("**Surface d'ulves détectée par Sentinel-2** (ha, images sans nuages, fond de la plage retiré)")
        st.scatter_chart(piv)
        st.markdown("**Profil saisonnier moyen** (surface du mois / moyenne mai-sept., toutes baies pilotes)")
        st.bar_chart(pd.Series(PROF.values, index=MONTHS[1:], name="profil").reindex(MONTHS[1:]))
        if D["s2_annual"] is not None:
            an = D["s2_annual"][D["s2_annual"]["site"].isin(piv.columns)]
            st.markdown("**Cumul annuel : satellite calibré** (ha)")
            st.bar_chart(an.pivot_table(index="year", columns="site", values="cumul_equiv_ha"), stack=False)
            cal = D["s2_cal"]
            st.caption(f"Calibration 2017-2022 : cumul CEVA ≈ {cal['k']:.1f} × surface moyenne mai-sept. "
                       f"(corrélation {cal.get('pearson', float('nan')):.2f} ; "
                       f"{cal.get('intra_site_corr', float('nan')):.2f} d'une année sur l'autre).")
        if sector in D["hydro"] and D["hydro"][sector]["no3_flux_kgd"].notna().any():
            st.markdown("**Flux de nitrates vers la baie (kg NO₃/jour, moyenne glissante 7 j)**")
            st.line_chart(D["hydro"][sector].set_index("date")["no3_flux_kgd"].rolling(7).mean())
        st.caption("Le satellite passe à heure fixe : à marée haute, les ulves immergées sont sous-détectées "
                   "(correction marée prévue).")

with tab4:
    m = D["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("R² (log surface)", f"{m['r2_log_model']:.2f}",
             f"{m['r2_log_model'] - m['r2_log_baseline_site_mean']:+.2f} vs niveau habituel")
    b.metric("Corrélation des écarts annuels", f"{m['corr_anomalies']:.2f}")
    c.metric("Bonne/mauvaise année (régional)", f"{m['corr_regional_years']:.2f}")
    d.metric("Classement des plages (Spearman)", f"{m['spearman_sites_per_year']:.2f}")
    st.markdown(f"""
**Méthode.** Le modèle (régression ridge) prédit l'**écart** de chaque plage à son niveau habituel :
report de l'année précédente (stock d'ulves qui passe l'hiver), flux de nitrates et débit des rivières
au printemps (Hub'Eau, < 25 km), températures du printemps et de l'automne (NASA POWER / MERRA-2).
Prévision émise au **1er juin**. Entraînement sur les surfaces CEVA 2002-2022 de 95 baies sableuses ;
validation : chaque année prédite par un modèle qui ne l'a jamais vue.

**Test en conditions réelles** (entraîné sur le passé seul, 2012-2022) : le modèle classe bien les plages
(0,73) mais l'erreur sur une plage et une année reste d'environ ×1,6 ; la règle « comme l'an dernier »
suit mieux les écarts annuels (0,37 contre 0,25).

**2028** : enchaîne deux années de scénario (2027 sert de report à 2028) ; fourchette élargie de 25 %.
**Mois par mois** : cumul annuel ÷ {K:.1f} × profil saisonnier mesuré par Sentinel-2.

**Élevages → nitrates.** Entre plages, +10 % d'élevages ICPE à moins de 20 km ≈
{(1.1 ** m['livestock_elasticity'] - 1) * 100:+.1f} % de nitrates (élasticité {m['livestock_elasticity']:.2f}).

**Limites.** Élevages = situation actuelle ; pas de CEVA public après 2022 ; 2027-2028 dépendent
d'hivers à venir, d'où les scénarios.
""")
    coefs = pd.Series(m["coefficients"]).rename(lambda c: LABELS[c.removesuffix("_a").removesuffix("_anom")])
    st.markdown("**Effet de chaque variable** (coefficient standardisé sur l'écart de log-surface)")
    st.bar_chart(coefs.sort_values(), horizontal=True)
