"""Indice quotidien de risque d'échouage d'ulves (baies pilotes), J-historique et J+1..J+10.

Idée : un échouage dangereux demande
  1. un STOCK d'ulves dans la baie (lent, mesuré par Sentinel-2 tous les ~5 jours, interpolé) ;
  2. des CONDITIONS qui poussent les algues sur la plage (rapides) : grande marée (coefficient) et
     vent de mer des jours précédents ;
  3. de la CHALEUR et du soleil, qui accélèrent la fermentation (H2S).

  potentiel d'échouage P = stock (ha) x conditions (0..1)
  risque H2S          R = P x chaleur (0.2..1)

Sources : SHOM (marégraphes, hauteur d'eau horaire validée), Open-Meteo (météo observée, prévisions
Météo-France à 10 jours, niveau de la mer prévu), Sentinel-2 (stock).
"""
import time

import numpy as np
import pandas as pd
import requests

SHOM = "https://services.data.shom.fr/maregraphie/observation/json/{id}"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
MARINE = "https://marine-api.open-meteo.com/v1/marine"
BREST = 3
BREST_SPRING_RANGE_M = 6.10  # marnage de vive-eau moyenne à Brest (coefficient 100)
DAILY = "wind_speed_10m_max,wind_direction_10m_dominant,temperature_2m_max,shortwave_radiation_sum,precipitation_sum"


def _get(url, params, timeout=120, retries=5):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"échec {url} {params}")


# ----------------------------------------------------------------------------- marées
def _tide_month(gauge: int, lo: str, hi: str) -> pd.Series | None:
    for src in (4, 3):  # 4 = horaire validé ; 3 = 10 min (données récentes pas encore validées)
        js = _get(SHOM.format(id=gauge), {"sources": src, "dtStart": f"{lo}T00:00:00Z", "dtEnd": f"{hi}T23:59:59Z"})
        d = pd.DataFrame(js.get("data", []))
        if len(d) >= 24:  # au moins un jour de données
            s = pd.Series(d["value"].values, index=pd.to_datetime(d["timestamp"], format="%Y/%m/%d %H:%M:%S"))
            return s.resample("1h").mean()
    return None


def tide_hourly(gauge: int, start: str, end: str, workers: int = 8) -> pd.Series:
    """Hauteur d'eau horaire (m), requêtes mois par mois (l'API tronque les longues périodes)."""
    from concurrent.futures import ThreadPoolExecutor
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    spans = [(m.strftime("%Y-%m-%d"), min(pd.Timestamp(end), m + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"))
             for m in months]
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda sp: _tide_month(gauge, *sp), spans) if p is not None]
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]


def tide_daily(h: pd.Series, gauge_is_brest: bool = False) -> pd.DataFrame:
    """Marnage du jour, hauteur à 11 h UTC (passage Sentinel-2) et coefficient (si Brest)."""
    day = h.groupby(h.index.normalize())
    out = pd.DataFrame({"tidal_range_m": day.max() - day.min(),
                        "level_11utc_m": h[h.index.hour == 11].groupby(h[h.index.hour == 11].index.normalize()).mean()})
    if gauge_is_brest:
        out["coef"] = (100 * out["tidal_range_m"] / BREST_SPRING_RANGE_M).round()
    out.index.name = "date"
    return out


def tide_forecast(lat: float, lon: float, days: int = 10) -> pd.DataFrame:
    """Marée prévue (modèle Open-Meteo Marine) : marnage journalier, relatif (à recaler sur le marégraphe)."""
    js = _get(MARINE, {"latitude": lat, "longitude": lon, "hourly": "sea_level_height_msl",
                       "forecast_days": days, "past_days": 7})
    h = pd.Series(js["hourly"]["sea_level_height_msl"], index=pd.to_datetime(js["hourly"]["time"])).astype(float)
    return tide_daily(h)


# ----------------------------------------------------------------------------- météo
def _daily_frame(js) -> pd.DataFrame:
    d = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")


def weather_history(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Météo journalière observée (NASA POWER / MERRA-2, sans limite de débit), mêmes colonnes qu'Open-Meteo."""
    js = _get("https://power.larc.nasa.gov/api/temporal/daily/point",
              {"parameters": "WS10M_MAX,WD10M,T2M_MAX,ALLSKY_SFC_SW_DWN,PRECTOTCORR", "community": "AG",
               "format": "JSON", "latitude": round(lat, 3), "longitude": round(lon, 3),
               "start": start.replace("-", ""), "end": end.replace("-", "")}, timeout=300)
    p = js["properties"]["parameter"]
    d = pd.DataFrame({k: pd.Series(v) for k, v in p.items()}).replace(-999.0, np.nan)
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    d.index.name = "date"
    return pd.DataFrame({"wind_speed_10m_max": d["WS10M_MAX"] * 3.6, "wind_direction_10m_dominant": d["WD10M"],
                         "temperature_2m_max": d["T2M_MAX"], "shortwave_radiation_sum": d["ALLSKY_SFC_SW_DWN"] * 3.6,
                         "precipitation_sum": d["PRECTOTCORR"]}).dropna(how="all")


def weather_forecast(lat: float, lon: float, days: int = 10) -> pd.DataFrame:
    return _daily_frame(_get(FORECAST, {"latitude": lat, "longitude": lon, "daily": DAILY, "forecast_days": days,
                                        "past_days": 10, "timezone": "Europe/Paris"}))
    # modèle choisi automatiquement par Open-Meteo : Météo-France (AROME/ARPEGE) à courte échéance,
    # puis ECMWF/GFS jusqu'à J+10


# ----------------------------------------------------------------------------- indice
def onshore(speed_kmh, dir_from_deg, seaward_deg):
    """Composante du vent qui souffle du large vers la plage (km/h, >= 0)."""
    return np.clip(speed_kmh * np.cos(np.radians(dir_from_deg - seaward_deg)), 0, None)


TIDE_EFFECT = -0.3675  # effet mesuré : log(1+surface vue) par mètre d'eau au passage (t = -28, 3 524 images)


def stock_daily(s2_site: pd.DataFrame, days: pd.DatetimeIndex, level_11utc: pd.Series | None = None,
                profile: pd.Series | None = None, max_gap: int = 20, persist: int = 21,
                window: int = 60, shrink: float = 0.5) -> pd.Series:
    """Stock d'ulves (ha, fond retiré) chaque jour.

    - chaque image (mars-oct.) est ramenée à la hauteur d'eau moyenne au passage (correction de marée) ;
    - interpolation entre images (au plus max_gap jours) ;
    - jusqu'à `persist` jours après la dernière image : normale du jour x écart de l'année à la normale
      (médiane des images des `window` derniers jours, rapprochée de 1 par l'exposant `shrink`).
      Test (2 976 images, 12 plages, coupures mai-sept. 2018-2025, horizon 21 j) : erreur moyenne
      20,4 -> 19,4 ha ; prolonger les 3 dernières images telles quelles faisait pire (21,8 ha) ;
    - au-delà (hiver) : moyenne mai-sept. de la plage x profil saisonnier du mois.
    """
    o = s2_site[s2_site["date"].dt.month.between(3, 10)][["date", "ulva_ha_corr"]].copy()
    if level_11utc is not None:
        lv = level_11utc.reindex(o["date"]).values
        ref = np.nanmean(level_11utc.values)
        adj = np.where(np.isnan(lv), 0.0, -TIDE_EFFECT * (lv - ref))
        o["ulva_ha_corr"] = np.expm1(np.log1p(o["ulva_ha_corr"]) + adj)
    obs = o.groupby("date")["ulva_ha_corr"].mean()
    s = obs.reindex(days.union(obs.index)).interpolate(method="time", limit=max_gap, limit_area="inside")
    s = s.reindex(days)
    if profile is not None and len(obs):
        base = obs[obs.index.month.isin([5, 6, 7, 8, 9])].mean()
        clim = pd.Series(days.month, index=days).map(profile) * base
        # après la dernière image (et dans les longues séries nuageuses) : on garde l'écart de l'année
        # à la normale mesuré sur les 3 dernières images, pendant au plus `persist` jours, puis la normale
        clim_obs = obs.index.month.map(profile).values * base
        r = (obs / np.where(clim_obs > 0.5, clim_obs, np.nan)).clip(0.2, 5)
        r = r.rolling(f"{window}D", min_periods=1).median() ** shrink  # images bruitées : médiane, rapprochée de 1
        grid = days.union(r.index)
        rr = r.reindex(grid).ffill()
        last = pd.Series(r.index, index=r.index).reindex(grid).ffill()
        fresh = ((grid.to_series() - last).dt.days <= persist).values
        rr = rr.where(fresh).reindex(days)
        s = s.fillna(clim * rr).fillna(clim)
    return s


def risk(df: pd.DataFrame, seaward_deg: float) -> pd.DataFrame:
    """Potentiel d'échouage = stock x grande marée ; risque H2S = potentiel x chaleur x calme.

    Choix fondés sur la vérification Sentinel-2 : le coefficient des 3 jours précédents augmente la
    surface sur l'estran (faiblement) ; le vent de mer la DIMINUE (dispersion) -> il n'entre pas dans
    le potentiel. Le vent intervient dans le risque H2S : le gaz s'accumule par temps chaud et calme.
    """
    d = df.copy()
    d["onshore_kmh"] = onshore(d["wind_speed_10m_max"], d["wind_direction_10m_dominant"], seaward_deg)
    d["onshore_3j"] = d["onshore_kmh"].rolling(3, min_periods=1).mean()
    d["coef_3j"] = d["coef"].rolling(3, min_periods=1).max()
    d["maree"] = (0.4 + 0.6 * ((d["coef_3j"] - 45) / (105 - 45)).clip(0, 1)).fillna(0.7)
    d["chaleur"] = (((d["temperature_2m_max"] - 12) / 14).clip(0, 1) * 0.7
                    + (d["shortwave_radiation_sum"] / 28).clip(0, 1) * 0.3)
    d["calme"] = 1 - ((d["wind_speed_10m_max"] - 15) / 35).clip(0, 0.7)
    d["potentiel_ha"] = d["stock_ha"] * d["maree"]
    d["risque_h2s"] = d["potentiel_ha"] * (0.2 + 0.8 * d["chaleur"]) * d["calme"]
    d["niveau"] = pd.cut(d["risque_h2s"], [-np.inf, 2, 8, 20, np.inf], labels=[1, 2, 3, 4]).astype("float")
    return d
