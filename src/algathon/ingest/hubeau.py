"""Débits journaliers et nitrates des rivières via les API Hub'Eau (sans compte).

- Hydrométrie : https://hubeau.eaufrance.fr/page/api-hydrometrie  (QmnJ = débit moyen journalier, L/s)
- Qualité des cours d'eau : https://hubeau.eaufrance.fr/page/api-qualite-cours-deau  (paramètre 1340 = nitrates)
"""
import re
import time
import unicodedata

import pandas as pd
import requests

BASE = "https://hubeau.eaufrance.fr/api/v2"
NITRATE = 1340


def _get(url, params, retries=3):
    """GET avec nouvelles tentatives (Hub'Eau renvoie parfois des réponses vides ou des 5xx)."""
    err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code in (200, 206):
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.RequestException, ValueError) as e:
            err = repr(e)
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"Hub'Eau inaccessible ({url}) : {err}")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z]+", " ", s)


def _matches(label: str, rivers: list[str] | None) -> bool:
    """Le nom de rivière apparaît comme mot entier dans le libellé (sans accents)."""
    if not rivers:
        return True
    words = f" {_norm(label)} "
    return any(f" {_norm(r).strip()} " in words for r in rivers)


def _bbox(b):
    return ",".join(str(x) for x in b)


def flow_stations(bbox) -> pd.DataFrame:
    js = _get(f"{BASE}/hydrometrie/referentiel/stations",
              {"bbox": _bbox(bbox), "en_service": "true", "size": 200,
               "fields": "code_station,libelle_station,libelle_cours_eau,longitude_station,latitude_station"})
    return pd.DataFrame(js["data"])


def daily_flow(code_station, start, end) -> pd.DataFrame:
    """Débit moyen journalier (m3/s). Pagination par curseur."""
    rows, url = [], f"{BASE}/hydrometrie/obs_elab"
    params = {"code_entite": code_station, "grandeur_hydro_elab": "QmnJ",
              "date_debut_obs_elab": start, "date_fin_obs_elab": end, "size": 20000,
              "fields": "date_obs_elab,resultat_obs_elab"}
    while url:
        js = _get(url, params)
        rows += js["data"]
        url, params = js.get("next"), None
    df = pd.DataFrame(rows, columns=["date_obs_elab", "resultat_obs_elab"])
    return pd.DataFrame({"date": pd.to_datetime(df["date_obs_elab"]),
                         "flow_m3s": df["resultat_obs_elab"] / 1000.0})


def nitrate_samples(bbox, start, end) -> pd.DataFrame:
    """Analyses de nitrates (mg NO3/L) de toutes les stations de la bbox, année par année."""
    rows = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        page = 1
        while True:
            js = _get(f"{BASE}/qualite_rivieres/analyse_pc",
                      {"bbox": _bbox(bbox), "code_parametre": NITRATE,
                       "date_debut_prelevement": max(start, f"{y}-01-01"),
                       "date_fin_prelevement": min(end, f"{y}-12-31"),
                       "size": 5000, "page": page,
                       "fields": "code_station,libelle_station,date_prelevement,resultat,longitude,latitude"})
            rows += js["data"]
            if not js.get("next"):
                break
            page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date_prelevement"])
    df = df[(df["resultat"] >= 0) & (df["resultat"] < 300)]  # valeurs aberrantes
    return df.drop(columns="date_prelevement")


def sector_hydro(bbox, start, end, rivers=None, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Série journalière par secteur : débit total, nitrate médian, flux de nitrates.

    Pour éviter de compter deux fois un même cours d'eau, on garde par rivière
    la station au plus fort débit moyen (la plus en aval).
    Retourne (série journalière du secteur, table des stations retenues).
    """
    st = flow_stations(bbox)
    st = st[st["libelle_cours_eau"].fillna("").map(lambda x: _matches(x, rivers))]
    flows = {}
    for _, s in st.iterrows():
        q = daily_flow(s["code_station"], start, end)
        if len(q) and q["flow_m3s"].notna().any():
            flows[s["code_station"]] = q.set_index("date")["flow_m3s"]
            st.loc[st["code_station"] == s["code_station"], "mean_flow"] = q["flow_m3s"].mean()
            st.loc[st["code_station"] == s["code_station"], "n_days"] = len(q)
    if "mean_flow" not in st:  # aucune station de débit exploitable (ex. Guissény)
        st["mean_flow"] = float("nan")
    st = st.dropna(subset=["mean_flow"])
    keep = st.sort_values("mean_flow").groupby("libelle_cours_eau").tail(1)
    if verbose:
        print(f"  {len(keep)} stations de débit retenues : {', '.join(keep['libelle_station'])}")

    days = pd.date_range(start, end, freq="D")
    q = pd.DataFrame({c: flows[c] for c in keep["code_station"]}, index=days).reindex(days)
    out = pd.DataFrame(index=days)
    out["flow_m3s"] = q.sum(axis=1, min_count=1)
    out["n_flow_stations"] = q.notna().sum(axis=1)

    no3 = nitrate_samples(bbox, start, end)
    if len(no3):
        no3 = no3[no3["libelle_station"].fillna("").map(lambda x: _matches(x, rivers))]
    if verbose:
        n = no3["code_station"].nunique() if len(no3) else 0
        print(f"  {len(no3)} analyses de nitrates sur {n} stations")
    if len(no3):
        daily = no3.groupby("date")["resultat"].median().reindex(days)
        out["no3_mgl_obs"] = daily
        out["no3_mgl"] = daily.interpolate(method="time", limit=45, limit_direction="both")
        # flux de nitrates : m3/s * mg/L = g/s -> kg/jour
        out["no3_flux_kgd"] = out["flow_m3s"] * out["no3_mgl"] * 86.4
    out.index.name = "date"
    return out.reset_index(), keep
