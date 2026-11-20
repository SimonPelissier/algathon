"""Construction des features temporelles et de la cible à horizon h."""
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, variables: list[str], lags: list[int],
                   windows: list[int], horizon: int) -> pd.DataFrame:
    df = df.sort_values(["site", "date"]).copy()
    g = df.groupby("site", group_keys=False)

    feats = {}
    for v in variables:
        # dernière observation disponible (gère les trous nuageux)
        filled = g[v].ffill()
        feats[f"{v}_last"] = filled
        for L in lags:
            feats[f"{v}_lag{L}"] = filled.groupby(df["site"]).shift(L)
        for w in windows:
            feats[f"{v}_mean{w}"] = g[v].transform(lambda s: s.rolling(w, min_periods=1).mean())
        # anomalie par rapport à la climatologie du site
        feats[f"{v}_anom"] = df[v] - g[v].transform("mean")

    out = pd.concat([df[["site", "date"]], pd.DataFrame(feats, index=df.index)], axis=1)
    doy = df["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["bloom_now"] = df["bloom"]
    # cible : bloom à J+horizon
    out["target"] = g["bloom"].shift(-horizon)
    return out.dropna(subset=["target"]).reset_index(drop=True)
