"""Données régionales pour toutes les plages CEVA : météo (NASA POWER / MERRA-2),
nitrates (Hub'Eau, toutes stations de la zone), élevages ICPE (Géorisques)."""
import time

import numpy as np
import pandas as pd
import requests

from .hubeau import BASE as HUBEAU, NITRATE, _get

GEORISQUES = "https://georisques.gouv.fr/api/v1/installations_classees"


def site_points(ceva: dict, typologie="baie sableuse") -> pd.DataFrame:
    """Un point représentatif (centre de l'enveloppe) par site CEVA."""
    rows = []
    for f in ceva["features"]:
        p = f["properties"]
        if typologie and p["typologie"] != typologie:
            continue
        coords = np.array([c for poly in f["geometry"]["coordinates"] for ring in poly for c in ring])
        lon, lat = coords.mean(axis=0)
        rows.append({"site": p["site"], "typologie": p["typologie"], "dep": p["Dep"],
                     "secteur": p["GRAND_SECT"], "lon": lon, "lat": lat,
                     "area_ha": p["Shape_Area"] / 1e4})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- météo
NASA_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_VARS = {"T2M": "temperature_2m_mean", "PRECTOTCORR": "precipitation_sum",
              "ALLSKY_SFC_SW_DWN": "shortwave_radiation_sum", "WS10M_MAX": "wind_speed_10m_max",
              "WD10M": "wind_direction_10m"}


def weather(points: pd.DataFrame, start: str, end: str, grid_deg: float = 0.5) -> pd.DataFrame:
    """Météo journalière NASA POWER (réanalyse MERRA-2, ~0.5°) par site.

    Les sites d'une même maille partagent une requête. Vent en km/h, rayonnement en kWh/m²/j.
    (Open-Meteo/ERA5 serait plus fin mais sa limite de débit est trop basse pour 26 ans x 50 points.)
    """
    pts = points.assign(glat=(points["lat"] / grid_deg).round() * grid_deg,
                        glon=(points["lon"] / grid_deg).round() * grid_deg)
    grid = pts[["glat", "glon"]].drop_duplicates().reset_index(drop=True)
    frames = []
    for i, g in grid.iterrows():
        params = {"parameters": ",".join(POWER_VARS), "community": "AG", "format": "JSON",
                  "latitude": g["glat"], "longitude": g["glon"],
                  "start": start.replace("-", ""), "end": end.replace("-", "")}
        for attempt in range(5):
            try:
                r = requests.get(NASA_POWER, params=params, timeout=300)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        par = r.json()["properties"]["parameter"]
        d = pd.DataFrame({POWER_VARS[k]: pd.Series(v) for k, v in par.items()})
        d.index = pd.to_datetime(d.index, format="%Y%m%d")
        d = d.replace(-999.0, np.nan).rename_axis("date").reset_index()
        d["wind_speed_10m_max"] *= 3.6
        d["glat"], d["glon"] = g["glat"], g["glon"]
        frames.append(d)
        print(f"  météo {i + 1}/{len(grid)} mailles", flush=True)
    w = pd.concat(frames)
    return pts[["site", "glat", "glon"]].merge(w, on=["glat", "glon"]).drop(columns=["glat", "glon"])


# ----------------------------------------------------------------------------- nitrates
def nitrates_region(bbox, start: str, end: str) -> pd.DataFrame:
    """Toutes les analyses de nitrates de la zone, mois par mois (limite de profondeur Hub'Eau)."""
    rows = []
    months = pd.date_range(start, end, freq="MS")
    for m in months:
        m_end = min(pd.Timestamp(end), m + pd.offsets.MonthEnd(0))
        page = 1
        while True:
            js = _get(f"{HUBEAU}/qualite_rivieres/analyse_pc",
                      {"bbox": ",".join(map(str, bbox)), "code_parametre": NITRATE,
                       "date_debut_prelevement": m.strftime("%Y-%m-%d"),
                       "date_fin_prelevement": m_end.strftime("%Y-%m-%d"),
                       "size": 5000, "page": page,
                       "fields": "code_station,date_prelevement,resultat,longitude,latitude"})
            rows += js["data"]
            if not js.get("next") or page * 5000 >= 20000:
                break
            page += 1
        if m.month == 12:
            print(f"  nitrates {m.year} : {len(rows)} analyses cumulées", flush=True)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.pop("date_prelevement"))
    return df[(df["resultat"] >= 0) & (df["resultat"] < 300)].dropna(subset=["longitude", "latitude"])


# ----------------------------------------------------------------------------- élevages
def livestock(departements: list[str]) -> pd.DataFrame:
    """Installations classées (ICPE) d'élevage géolocalisées : bovins, porcs, volailles."""
    rows = []
    for dep in departements:
        page = 1
        while True:
            js = None
            for attempt in range(4):
                try:
                    r = requests.get(GEORISQUES, params={"departement": dep, "page": page,
                                                         "page_size": 1000}, timeout=120)
                    if r.status_code == 200:
                        js = r.json()
                        break
                except requests.RequestException:
                    pass
                time.sleep(5 * (attempt + 1))
            if js is None:
                print(f"  ICPE {dep} page {page} : échec, on passe", flush=True)
                break
            for d in js["data"]:
                if d.get("bovins") or d.get("porcs") or d.get("volailles"):
                    rows.append({k: d.get(k) for k in ("codeAIOT", "raisonSociale", "commune", "codeInsee",
                                                       "longitude", "latitude", "bovins", "porcs",
                                                       "volailles", "regime", "etatActivite")})
            if page >= js.get("total_pages", 1):
                break
            page += 1
        print(f"  ICPE dép. {dep} : {len(rows)} élevages cumulés", flush=True)
    return pd.DataFrame(rows).dropna(subset=["longitude", "latitude"]).drop_duplicates("codeAIOT")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- débits
def monthly_flows(points: pd.DataFrame, bbox, start: str, end: str, radius_km: float = 25) -> pd.DataFrame:
    """Débits mensuels (QmM, m3/s) des stations Hub'Eau à moins de radius_km d'une plage."""
    st = _get(f"{HUBEAU}/hydrometrie/referentiel/stations",
              {"bbox": ",".join(map(str, bbox)), "size": 2000,
               "fields": "code_station,libelle_cours_eau,longitude_station,latitude_station"})["data"]
    st = pd.DataFrame(st).dropna(subset=["longitude_station", "latitude_station"])
    dmin = np.array([haversine_km(st["longitude_station"].values, st["latitude_station"].values,
                                  p.lon, p.lat) for p in points.itertuples()]).min(axis=0)
    st = st[dmin <= radius_km]
    frames = []
    for i, s in enumerate(st.itertuples()):
        rows, url = [], f"{HUBEAU}/hydrometrie/obs_elab"
        params = {"code_entite": s.code_station, "grandeur_hydro_elab": "QmM",
                  "date_debut_obs_elab": start, "date_fin_obs_elab": end, "size": 20000,
                  "fields": "date_obs_elab,resultat_obs_elab"}
        try:
            while url:
                js = _get(url, params)
                rows += js["data"]
                url, params = js.get("next"), None
        except RuntimeError as e:
            print(f"  {s.code_station} : {e}", flush=True)
            continue
        if rows:
            d = pd.DataFrame(rows)
            frames.append(pd.DataFrame({"code_station": s.code_station, "lon": s.longitude_station,
                                        "lat": s.latitude_station,
                                        "date": pd.to_datetime(d["date_obs_elab"]),
                                        "q_m3s": d["resultat_obs_elab"] / 1000.0}))
        if (i + 1) % 25 == 0:
            print(f"  débits {i + 1}/{len(st)} stations", flush=True)
    return pd.concat(frames, ignore_index=True)
