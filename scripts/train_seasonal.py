"""Entraîne le modèle saisonnier (CEVA 2002-2022) et prépare les prévisions 2023-2027.

Sorties (data/processed/seasonal/) :
  table.parquet       plage x année : variables + cible CEVA
  cv_predictions.csv  prédictions en validation croisée (chaque année prédite sans elle)
  hindcast.csv        estimations 2023-2026 (météo et rivières observées ; CEVA non publié après 2022)
  metrics.json        scores + coefficients
  model.joblib        modèle + tout ce qu'il faut pour les scénarios 2027 (utilisé par l'application)
"""
import json

import joblib
import numpy as np
import pandas as pd
from _common import ROOT

from algathon import seasonal as S
from algathon.sites import ceva_annual_table, load_ceva

reg, out = ROOT / "data/processed/regional", ROOT / "data/processed/seasonal"
out.mkdir(parents=True, exist_ok=True)

sites = pd.read_csv(reg / "sites.csv")
ceva = ceva_annual_table(load_ceva(ROOT / "data/raw/ceva/prolif_AV_site.geojson"))
weather = pd.read_parquet(reg / "weather.parquet")
flows = pd.read_parquet(reg / "flows_monthly.parquet")
no3 = pd.concat([pd.read_parquet(p) for p in sorted(reg.glob("nitrates_*.parquet"))], ignore_index=True)
lv = pd.read_csv(reg / "livestock_icpe.csv")

years = list(range(2002, 2027))
t = S.build_table(sites, ceva, weather, flows, no3, lv, years)
t.to_parquet(out / "table.parquet", index=False)
print(f"table : {len(t)} lignes, {t['cumul_ha'].notna().sum()} avec cible CEVA")

cv, metrics = S.cross_validate(t)
cv[["site", "year", "y_log", "pred_log", "site_hist"]].to_csv(out / "cv_predictions.csv", index=False)

model, site_hist, coefs = S.fit_final(t)
obs = t.dropna(subset=["y_log"])
bundle = {
    "model": model, "site_hist": site_hist, "features": S.FEATURES,
    "site_means": {c: obs.groupby("site")[c].mean() for c in S.DYNAMIC},
    "resid_std_log": metrics["resid_std_log"],
    "elasticity": S.livestock_elasticity(t),
    "n_elevages": t.groupby("site")["n_elevages"].first(),
    "scenarios": S.scenario_anomalies(obs),
}
hc = S.hindcast(bundle, t, range(2023, 2027))
hc["pred_ha"] = np.expm1(hc["pred_log"]).clip(lower=0)
hc[["site", "year", "pred_ha", "pred_log", "facteurs"] + S.DYNAMIC].to_csv(out / "hindcast.csv", index=False)

# point de départ 2027 : dernières valeurs connues de nitrates + écart estimé 2026
last = t.sort_values("year").groupby("site")[["no3_winter", "no3_spring"]].last()
bundle["last_obs"] = {c: last[c] for c in last}
h26 = hc[hc["year"] == 2026].set_index("site")
bundle["prev_anom"] = h26["pred_log"] - h26.index.map(site_hist)
joblib.dump(bundle, out / "model.joblib")

metrics["coefficients"] = {k: float(v) for k, v in coefs.sort_values(key=abs, ascending=False).items()}
metrics["livestock_elasticity"] = bundle["elasticity"]
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=1,
                 ensure_ascii=False))
