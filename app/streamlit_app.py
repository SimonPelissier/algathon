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
FIRST_YEAR, LAST_OBS, FORECAST_YEAR = 2002, 2022, 2027

RISK = [  # (seuil ha, libellé, couleur RGB)
    (50, "Très élevé", [180, 30, 45]),
    (15, "Élevé", [230, 110, 40]),
    (3, "Modéré", [240, 190, 60]),
    (0, "Faible", [90, 170, 110]),
]

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
        "livestock": pd.read_csv(DATA / "livestock.csv"),
        "hydro": {}, "hydro_st": {}, "s2": {},
    }
    for p in (DATA / "hydro").glob("*.csv"):
        key = p.stem.replace("_stations", "")
        target = d["hydro_st"] if p.stem.endswith("_stations") else d["hydro"]
        target[key] = pd.read_csv(p, parse_dates=None if p.stem.endswith("_stations") else ["date"])
    daily = DATA / "s2" / "s2_daily_corrected.csv"
    if daily.exists():
        s2 = pd.read_csv(daily, parse_dates=["date"])
        d["s2"] = {k: g for k, g in s2.groupby("sector")}
    d["s2_annual"] = pd.read_csv(DATA / "s2" / "s2_annual.csv") if (DATA / "s2" / "s2_annual.csv").exists() else None
    d["s2_cal"] = json.loads((DATA / "s2" / "calibration.json").read_text()) if (DATA / "s2" / "calibration.json").exists() else None
    return d


@st.cache_resource
def load_model():
    return joblib.load(DATA / "model.joblib")


def risk_of(ha):
    for thr, label, color in RISK:
        if ha >= thr:
            return label, color
    return RISK[-1][1], RISK[-1][2]


def predict_2027(D, M, scenario, r_no3, r_lv):
    p = predict_scenario(M, scenario, r_no3, r_lv)
    p["source"] = "Prévision 2027"
    return p


def year_values(D, M, year, scenario, r_no3, r_lv):
    if year <= LAST_OBS:
        v = D["ceva"][D["ceva"]["year"] == year][["site", "cumul_ha"]].rename(columns={"cumul_ha": "ha"})
        v["ha_lo"], v["ha_hi"], v["source"] = v["ha"], v["ha"], "Observé (CEVA)"
        v["facteurs"] = ""
    elif year < FORECAST_YEAR:
        v = D["hindcast"][D["hindcast"]["year"] == year][["site", "pred_ha", "pred_log", "facteurs"]].rename(columns={"pred_ha": "ha"})
        s = M["resid_std_log"]
        v["ha_lo"], v["ha_hi"] = np.expm1(v["pred_log"] - s).clip(0), np.expm1(v["pred_log"] + s).clip(0)
        v["source"] = "Estimé (modèle + météo réelle)"
        v = v.drop(columns="pred_log")
        if D["s2_annual"] is not None:  # baies pilotes : mesure satellite calibrée sur le CEVA
            sat = D["s2_annual"][(D["s2_annual"]["year"] == year) & (D["s2_annual"]["n_images"] >= 5)]
            sat = sat.set_index("site")["cumul_equiv_ha"]
            hit = v["site"].isin(sat.index)
            v.loc[hit, "ha"] = v.loc[hit, "site"].map(sat).values
            v.loc[hit, "ha_lo"] = v.loc[hit, "ha"] * 0.6
            v.loc[hit, "ha_hi"] = v.loc[hit, "ha"] * 1.4
            v.loc[hit, "facteurs"] = "mesure Sentinel-2 calibrée sur le CEVA"
    else:
        v = predict_2027(D, M, scenario, r_no3, r_lv)
    v = v.merge(D["sites"], on="site")
    v["risque"], v["color"] = zip(*v["ha"].map(risk_of))
    return v.sort_values("ha", ascending=False)


# ----------------------------------------------------------------------------- interface
D, M = load(), load_model()

st.sidebar.title("🌊 Marées vertes")
year = st.sidebar.slider("Année", FIRST_YEAR, FORECAST_YEAR, FORECAST_YEAR)
scenario, r_no3, r_lv = next(iter(M["scenarios"])), 0, 0
if year == FORECAST_YEAR:
    st.sidebar.markdown("**Scénarios 2027**")
    scenario = st.sidebar.radio("Hiver 2026-2027", list(M["scenarios"]), index=0)
    r_no3 = st.sidebar.slider("Nitrates dans les rivières (%)", -60, 20, 0, step=5)
    r_lv = st.sidebar.slider("Élevages (%)", -60, 20, 0, step=5)
st.sidebar.markdown("**Couches**")
show_sites = st.sidebar.checkbox("Plages (surface d'ulves)", True)
show_no3 = st.sidebar.checkbox("Nitrates des rivières", True)
show_lv = st.sidebar.checkbox("Densité d'élevages (ICPE)", False)
show_flux = st.sidebar.checkbox("Flux de nitrates — baies pilotes", True)

v = year_values(D, M, year, scenario, r_no3, r_lv)
typical = np.expm1(M["site_hist"])  # année typique de chaque plage (médiane géométrique 2002-2022)
hist_total = typical.sum()

st.title("Observatoire des marées vertes")
st.caption("Bretagne et façade atlantique · surfaces cumulées d'ulves par plage (ha/an) · "
           + {True: "observations CEVA", False: "estimations du modèle"}[year <= LAST_OBS])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Surface cumulée totale", f"{v['ha'].sum():,.0f} ha".replace(",", " "),
          f"{(v['ha'].sum() / hist_total - 1) * 100:+.0f} % vs année typique")
c2.metric("Plages à risque très élevé", int((v["risque"] == "Très élevé").sum()))
c3.metric("Plages à risque élevé", int((v["risque"] == "Élevé").sum()))
c4.metric("Source", v["source"].iloc[0])

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
    ny = year if year < FORECAST_YEAR else int(D["no3"]["year"].max())
    no3 = D["no3"][(D["no3"]["year"] == ny) & (D["no3"]["n"] >= 3)].copy()
    if year == FORECAST_YEAR:
        no3["no3"] *= 1 + r_no3 / 100
    x = (no3["no3"].clip(0, 50) / 50).values
    no3["r"], no3["g"], no3["b"] = (60 + 190 * x).astype(int), (140 - 80 * x).astype(int), (200 - 170 * x).astype(int)
    no3["label"] = "Station " + no3["code_station"].astype(str) + " : " + no3["no3"].round(0).astype(int).astype(str) \
        + f" mg NO₃/L (médiane {ny})"
    layers.append(pdk.Layer("ScatterplotLayer", no3, get_position=["lon", "lat"], get_radius=900,
                            get_fill_color="[r, g, b, 170]", pickable=True))
if show_flux and D["hydro"]:
    arcs = []
    for sector, h in D["hydro"].items():
        hy = h[h["date"].dt.year == min(max(year, h["date"].dt.year.min()), h["date"].dt.year.max())]
        flux = hy["no3_flux_kgd"].mean() * (1 + r_no3 / 100)
        bay = D["sites"][D["sites"]["pilot_sector"] == sector][["lon", "lat"]].mean()
        for _, s in D["hydro_st"].get(sector, pd.DataFrame()).iterrows():
            arcs.append({"from_lon": s["longitude_station"], "from_lat": s["latitude_station"],
                         "to_lon": bay["lon"], "to_lat": bay["lat"], "flux": flux,
                         "label": f"{s['libelle_cours_eau']} → baie de {sector.replace('_', '-')} "
                                  f"(flux moyen du secteur : {flux:,.0f} kg NO₃/j)".replace(",", " ")})
    if arcs:
        layers.append(pdk.Layer("ArcLayer", pd.DataFrame(arcs), get_source_position=["from_lon", "from_lat"],
                                get_target_position=["to_lon", "to_lat"], get_width="flux / 400",
                                get_source_color=[230, 120, 40, 220], get_target_color=[40, 150, 90, 220],
                                pickable=True))
if show_sites:
    sv = v.copy()
    sv["label"] = (sv["site"] + " — " + sv["ha"].round(1).astype(str) + " ha (" + sv["risque"] + ")")
    layers.append(pdk.Layer("ColumnLayer", sv, get_position=["lon", "lat"], get_elevation="ha",
                            elevation_scale=60, radius=900, get_fill_color="color", pickable=True,
                            auto_highlight=True, extruded=True))

st.pydeck_chart(pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                         initial_view_state=pdk.ViewState(latitude=48.0, longitude=-3.2, zoom=6.6, pitch=45),
                         tooltip={"text": "{label}"}), width="stretch", height=560)
st.caption("Colonnes : surface cumulée d'ulves par plage · points : nitrates médians des rivières "
           "(bleu < 10 mg/L → rouge > 50 mg/L) · arcs : rivières → baies pilotes (épaisseur ∝ flux).")

# ----------------------------------------------------------------------------- onglets
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Plages ciblées", "📈 Détail d'une plage", "🛰️ Baies pilotes", "🧠 Modèle"])

with tab1:
    show = v[["site", "secteur", "dep", "ha", "ha_lo", "ha_hi", "risque", "facteurs"]].copy()
    show["année typique"] = show["site"].map(typical)
    show["écart vs typique"] = (show["ha"] / show["année typique"] - 1).mul(100).round(0).astype("Int64").astype(str) + " %"
    show = show.rename(columns={"ha": "surface (ha)", "ha_lo": "bas", "ha_hi": "haut", "dep": "dép.",
                                "facteurs": "principaux facteurs (effet sur log-surface)"})
    st.dataframe(show.round(1), width="stretch", hide_index=True, height=420)
    st.download_button("Télécharger la liste (CSV)", show.to_csv(index=False).encode("utf-8"),
                       f"plages_ciblees_{year}.csv", "text/csv")

with tab2:
    site = st.selectbox("Plage", v["site"].tolist())
    obs = D["ceva"][D["ceva"]["site"] == site].set_index("year")["cumul_ha"].rename("Observé (CEVA)")
    cv = D["cv"][D["cv"]["site"] == site].set_index("year")["pred_log"].pipe(np.expm1).rename("Modèle (validation croisée)")
    hc = D["hindcast"][D["hindcast"]["site"] == site].set_index("year")["pred_ha"].rename("Estimé 2023-2026")
    sat = (D["s2_annual"][D["s2_annual"]["site"] == site].set_index("year")["cumul_equiv_ha"].rename("Satellite calibré")
           if D["s2_annual"] is not None else pd.Series(dtype=float, name="Satellite calibré"))
    f27 = predict_2027(D, M, scenario, r_no3, r_lv).set_index("site")
    p27 = pd.Series({FORECAST_YEAR: f27.loc[site, "ha"]} if site in f27.index else {}, name="Prévision 2027")
    st.line_chart(pd.concat([obs, cv, hc, sat, p27], axis=1))
    row = D["table"][D["table"]["site"] == site]
    st.markdown("**Conditions de l'hiver et du printemps** (variables du modèle)")
    if site in f27.index:
        st.markdown(f"**2027 — {scenario}** : {f27.loc[site, 'ha']:.0f} ha "
                    f"(fourchette {f27.loc[site, 'ha_lo']:.0f}–{f27.loc[site, 'ha_hi']:.0f}) · {f27.loc[site, 'facteurs']}")
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
        if D["s2_annual"] is not None:
            an = D["s2_annual"][D["s2_annual"]["site"].isin(piv.columns)]
            st.markdown("**Cumul annuel : CEVA (observé) vs satellite calibré** (ha)")
            st.bar_chart(an.pivot_table(index="year", columns="site", values="cumul_equiv_ha"), stack=False)
            cal = D["s2_cal"]
            st.caption(f"Calibration 2017-2022 : cumul CEVA ≈ {cal['k']:.1f} × surface moyenne mai-sept. "
                       f"(corrélation {cal['pearson']:.2f} ; {cal['intra_site_corr']:.2f} d'une année sur l'autre).")
        if sector in D["hydro"]:
            st.markdown("**Flux de nitrates vers la baie (kg NO₃/jour)**")
            st.line_chart(D["hydro"][sector].set_index("date")["no3_flux_kgd"].rolling(7).mean())
        st.caption("Le satellite passe à heure fixe : à marée haute, les ulves immergées sont sous-détectées "
                   "(correction marée prévue).")

with tab4:
    m = D["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("R² (log surface)", f"{m['r2_log_model']:.2f}", f"{m['r2_log_model'] - m['r2_log_baseline_site_mean']:+.2f} vs niveau habituel")
    b.metric("Corrélation des écarts annuels", f"{m['corr_anomalies']:.2f}")
    c.metric("Bonne/mauvaise année (régional)", f"{m['corr_regional_years']:.2f}")
    d.metric("Classement des plages (Spearman)", f"{m['spearman_sites_per_year']:.2f}")
    st.markdown(f"""
**Méthode.** Le modèle (régression ridge) prédit l'**écart** de chaque plage à son niveau habituel,
à partir de l'hiver et du printemps qui précèdent : température, pluie, coups de vent, ensoleillement
(NASA POWER / MERRA-2), débits et nitrates des rivières à moins de 25 km (Hub'Eau), et l'écart de
l'année précédente (stock d'ulves qui passe l'hiver). Prévision émise au **1er juin**.
Entraînement sur les surfaces CEVA 2002-2022 de 95 baies sableuses ; validation : chaque année
est prédite par un modèle qui ne l'a jamais vue.

**Élevages → nitrates.** Entre plages, +10 % d'élevages ICPE à moins de 20 km ≈
{(1.1 ** m['livestock_elasticity'] - 1) * 100:+.1f} % de nitrates dans les rivières (élasticité estimée
{m['livestock_elasticity']:.2f}). Le curseur « élevages » passe par cette relation.

**Limites.** Élevages = situation actuelle (pas d'historique, effectifs non connus). Après 2022, pas de
donnée CEVA publique : surfaces estimées. 2027 dépend de l'hiver et du printemps à venir, d'où les scénarios.
""")
    coefs = pd.Series(m["coefficients"]).rename(lambda c: LABELS[c.removesuffix("_a").removesuffix("_anom")])
    st.markdown("**Effet de chaque variable** (coefficient standardisé sur l'écart de log-surface)")
    st.bar_chart(coefs.sort_values(), horizontal=True)
