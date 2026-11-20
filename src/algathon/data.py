"""Chargement des données. Inclut un générateur synthétique pour tester le pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

from .indices import compute_all


def make_synthetic(n_sites: int = 30, n_days: int = 730, seed: int = 0) -> pd.DataFrame:
    """Série journalière par site : bandes S2, variables locales, et label bloom.

    Le bloom est favorisé par eau chaude + nutriments élevés + vent faible + ensoleillement.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values
    rows = []
    for s in range(n_sites):
        base_t = rng.normal(14, 2)
        sst = base_t + 6 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 0.8, n_days)
        rain = rng.gamma(0.6, 4, n_days)
        nitrate = pd.Series(rain).rolling(10, min_periods=1).mean().values * 2 + rng.normal(5, 1, n_days)
        wind = np.abs(rng.normal(6, 3, n_days))
        par = 30 + 20 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 4, n_days)

        drive = 0.35 * (sst - 16) + 0.25 * (nitrate - 8) - 0.3 * (wind - 6) + 0.05 * (par - 30)
        biomass = pd.Series(drive).ewm(span=7).mean().values + rng.normal(0, 0.5, n_days)
        bloom = (biomass > np.quantile(biomass, 0.85)).astype(int)

        chl = np.clip(0.02 + 0.01 * biomass, 0, None)
        b4 = 0.03 - 0.3 * chl + rng.normal(0, 0.002, n_days)
        b5 = 0.03 + 0.5 * chl + rng.normal(0, 0.002, n_days)
        b6 = 0.02 + 0.2 * chl + rng.normal(0, 0.002, n_days)
        b8 = 0.01 + 0.4 * chl + rng.normal(0, 0.002, n_days)
        b11 = 0.005 + rng.normal(0, 0.001, n_days)

        # couverture nuageuse : ~40 % des jours sans observation satellite
        cloudy = rng.random(n_days) < 0.4
        idx = compute_all({"B4": b4, "B5": b5, "B6": b6, "B8": b8, "B11": b11})
        for k in idx:
            idx[k] = np.where(cloudy, np.nan, idx[k])

        rows.append(pd.DataFrame({
            "site": s, "date": dates, "sst": sst, "rain": rain, "nitrate": nitrate,
            "wind": wind, "par": par, **idx, "bloom": bloom,
        }))
    return pd.concat(rows, ignore_index=True)


def load_table(path: str | Path) -> pd.DataFrame:
    """Charge une table (site, date, variables..., bloom) en CSV ou Parquet."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
