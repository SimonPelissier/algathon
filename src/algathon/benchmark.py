"""Benchmark de familles de modèles pour la surface d'échouage annuelle par baie.

Tous les modèles reçoivent exactement les mêmes variables et la même validation
temporelle que `baseline_annuel` (cible et variables d'échelle en log1p, indicatrices
de baie, imputation par la médiane du pli d'entraînement). Seule la forme de la
relation change : linéaire, additive non linéaire, arbres, noyaux, réseau de neurones.

L'incertitude sur les scores est estimée par bootstrap sur les années testées
(on retire des années entières, pas des lignes isolées, pour respecter la dépendance
entre baies une même année).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVR

# famille de chaque modèle, pour l'affichage
FAMILLES = {
    "annee_precedente": "référence naïve",
    "moyenne": "référence naïve",
    "ridge": "linéaire",
    "lasso": "linéaire",
    "elasticnet": "linéaire",
    "gam_splines": "additif non linéaire (type GAM)",
    "random_forest": "arbres",
    "extra_trees": "arbres",
    "gradient_boosting": "arbres (boosting)",
    "svr_rbf": "noyaux",
    "processus_gaussien": "noyaux (bayésien)",
    "knn": "plus proches voisins",
    "reseau_neurones": "réseau de neurones (MLP)",
}


def _prep(*etapes):
    """Imputation + standardisation avant un modèle qui ne gère pas les trous."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), *etapes)


def tous_les_modeles() -> Dict[str, object]:
    """Modèles réglés pour ~100-140 lignes d'entraînement (fortement régularisés)."""
    alphas = np.logspace(-2, 3, 21)
    return {
        "ridge": _prep(RidgeCV(alphas=alphas)),
        "lasso": _prep(LassoCV(cv=5, max_iter=20000, random_state=0)),
        "elasticnet": _prep(ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=20000,
                                         random_state=0)),
        # GAM approché : chaque variable passe par une base de splines (effet courbe),
        # les effets s'additionnent, et la Ridge empêche les courbes trop tordues.
        "gam_splines": _prep(SplineTransformer(n_knots=4, degree=3), RidgeCV(alphas=np.logspace(-1, 4, 21))),
        "random_forest": _prep(RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                     max_features=0.5, random_state=0)),
        "extra_trees": _prep(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                                 max_features=0.5, random_state=0)),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=2, max_iter=150, learning_rate=0.05, min_samples_leaf=8,
            l2_regularization=1.0, random_state=0),
        "svr_rbf": _prep(SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
        "processus_gaussien": _prep(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(0.5),
            normalize_y=True, random_state=0)),
        "knn": _prep(KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "reseau_neurones": _prep(MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0,
                                              max_iter=5000, random_state=0)),
    }


def bootstrap_annees(pred: pd.DataFrame, n: int = 1000, graine: int = 0) -> pd.DataFrame:
    """Intervalle à 90 % de la MAE relative et du gain sur « année précédente »,
    en rééchantillonnant les années testées (blocs d'années)."""
    cols = [c for c in pred.columns if c.startswith("pred_")]
    p = pred.dropna(subset=cols + ["vrai"]).copy()
    moy = p.groupby("baie")["vrai"].mean()
    for c in cols:
        p["err_" + c] = (p[c] - p["vrai"]).abs() / p["baie"].map(moy)
    annees = p["annee"].unique()
    par_annee = {a: g for a, g in p.groupby("annee")}
    rng = np.random.default_rng(graine)
    tirages = []
    for _ in range(n):
        echantillon = pd.concat([par_annee[a] for a in rng.choice(annees, len(annees))])
        mae = {c[5:]: echantillon["err_" + c].mean() for c in cols}
        tirages.append(mae)
    t = pd.DataFrame(tirages)
    gain = 1 - t.div(t["annee_precedente"], axis=0)
    return pd.DataFrame({
        "mae_rel": {c[5:]: p["err_" + c].mean() for c in cols},
        "mae_rel_p05": t.quantile(0.05),
        "mae_rel_p95": t.quantile(0.95),
        "gain_vs_n_1": 1 - pd.Series({c[5:]: p["err_" + c].mean() for c in cols})
        / p["err_pred_annee_precedente"].mean(),
        "gain_p05": gain.quantile(0.05),
        "gain_p95": gain.quantile(0.95),
        "p_bat_n_1": (gain > 0).mean(),
    }).assign(famille=lambda d: d.index.map(FAMILLES)).sort_values("mae_rel")
