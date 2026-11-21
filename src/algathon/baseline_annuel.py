"""Modèles de référence pour la surface d'échouage annuelle par baie.

Un seul modèle pour toutes les baies (environ 20 années par baie, trop peu pour un modèle
par baie) : chaque baie a son propre niveau (indicatrice), et les effets des variables
sont communs. Cible et variables de débit et de flux en log1p : un effet devient un
facteur multiplicatif, comparable d'une petite baie à une grande.

Validation, toujours découpée par année, jamais au hasard :
  - « loyo » (leave-one-year-out) : on prédit l'année N avec toutes les autres, sauf
    N+1, exclue car sa variable surface_n_1 vaut exactement la cible de N (fuite).
  - « glissante » : on prédit N avec les seules années < N. C'est la situation réelle
    d'une prévision ; c'est le score à retenir.

Références naïves, calculées dans le même pli :
  - « annee_precedente » : surface de N-1
  - « moyenne » : moyenne de la baie sur les années d'entraînement du pli
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CIBLE = "surface_ha"

JEUX_DE_VARIABLES: Dict[str, List[str]] = {
    # tout est connu au 1er mai : vraie prévision de la saison
    "prevision_1er_mai": [
        "surface_n_1", "debit_jan_avr", "no3_jan_avr", "flux_n_jan_avr_t",
        "ray_mars_avr", "temp_mars_avr", "pluie_mars_avr",
    ],
    # variables de la saison elle-même : explique la saison, ne la prévoit pas
    "diagnostic_mai_aout": [
        "surface_n_1", "debit_mai_aout", "no3_mai_aout", "flux_n_mai_aout_t",
        "ray_mars_mai", "temp_mars_mai", "pluie_mars_mai",
    ],
}
# variables passées en log1p (positives, très asymétriques, d'échelles très différentes)
VARIABLES_LOG = {"surface_n_1", "debit_jan_avr", "flux_n_jan_avr_t",
                 "debit_mai_aout", "flux_n_mai_aout_t"}


# ---------------------------------------------------------------- découpages

def plis_loyo(annees: List[int], tampon: int = 1) -> Iterator[Tuple[List[int], int]]:
    """(années d'entraînement, année de test). Exclut les `tampon` années suivant le test."""
    annees = sorted(set(annees))
    for a in annees:
        exclues = set(range(a, a + tampon + 1))
        yield [x for x in annees if x not in exclues], a


def plis_glissants(annees: List[int], min_train: int = 8) -> Iterator[Tuple[List[int], int]]:
    """(années strictement antérieures, année de test), avec au moins `min_train` années."""
    annees = sorted(set(annees))
    for i, a in enumerate(annees):
        if i >= min_train:
            yield annees[:i], a


# ---------------------------------------------------------------- modèles

def matrice(df: pd.DataFrame, variables: List[str], baies: List[str]) -> pd.DataFrame:
    """Variables (log1p si besoin) + indicatrices de baie, colonnes dans un ordre fixe."""
    X = pd.DataFrame(index=df.index)
    for v in variables:
        X[v] = np.log1p(df[v].clip(lower=0)) if v in VARIABLES_LOG else df[v]
    for b in baies:
        X["baie_" + b] = (df["baie"] == b).astype(float)
    return X


def random_forest() -> object:
    """Forêt aléatoire : 300 arbres entraînés chacun sur un tirage des lignes et une moitié
    des variables, puis moyennés. Feuilles d'au moins 5 lignes (~140 lignes au total)."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                              random_state=0),
    )


def extra_trees() -> object:
    """Extra Trees : comme la forêt aléatoire, mais les seuils de coupure sont tirés au
    hasard. Plus de diversité entre arbres, donc des prédictions plus lisses et moins de
    surapprentissage sur petit jeu de données (meilleur modèle du benchmark)."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                            random_state=0),
    )


def fabriquer_modeles() -> Dict[str, object]:
    return {
        # référence linéaire, lisible (effet de chaque variable = un coefficient)
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 21)),
        ),
        # modèles non linéaires retenus après le benchmark (scripts/benchmark_modeles.py)
        "random_forest": random_forest(),
        "extra_trees": extra_trees(),
        # boosting gardé pour comparaison : moins bon sur ~140 lignes
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=2, max_iter=150, learning_rate=0.05,
            min_samples_leaf=8, l2_regularization=1.0, random_state=0,
        ),
    }


def importance_hors_echantillon(table: pd.DataFrame, variables: List[str], nom_modele: str,
                                n_repetitions: int = 10) -> pd.Series:
    """Importance par permutation, mesurée sur chaque année laissée de côté (plis loyo).

    Pour chaque année : on entraîne sans elle (ni N+1), on mélange une variable dans
    l'année testée et on mesure la hausse d'erreur. Les indicatrices de baie sont
    regroupées en « baie (niveau propre) ». Renvoie des % (somme des parts positives = 100).
    """
    df = table.dropna(subset=[CIBLE])
    baies = sorted(df["baie"].unique())
    X, y = matrice(df, variables, baies), np.log1p(df[CIBLE].values)
    parts = []
    for annees_train, annee_test in plis_loyo(df["annee"].tolist()):
        tr, te = df["annee"].isin(annees_train).values, (df["annee"] == annee_test).values
        modele = fabriquer_modeles()[nom_modele].fit(X[tr], y[tr])
        pi = permutation_importance(modele, X[te], y[te], n_repeats=n_repetitions,
                                    random_state=0, scoring="neg_mean_absolute_error")
        parts.append(pd.Series(pi.importances_mean, index=X.columns))
    imp = pd.concat(parts, axis=1).mean(axis=1)
    imp = imp.groupby(lambda c: "baie (niveau propre)" if c.startswith("baie_") else c).sum()
    return (100 * imp / imp.clip(lower=0).sum()).sort_values(ascending=False)


def evaluer(table: pd.DataFrame, variables: List[str], methode: str = "glissante",
            modeles=None) -> pd.DataFrame:
    """Prédictions hors échantillon de chaque modèle et des deux références naïves.

    Renvoie une ligne par (baie, année testée) avec les colonnes vrai, pred_<méthode>.
    `modeles` : fonction sans argument renvoyant {nom: modèle sklearn} (défaut : Ridge et
    gradient boosting). Un modèle neuf est créé à chaque pli.
    """
    modeles = modeles or fabriquer_modeles
    df = table.dropna(subset=[CIBLE]).copy()
    baies = sorted(df["baie"].unique())
    plis = plis_glissants if methode == "glissante" else plis_loyo
    resultats = []
    for annees_train, annee_test in plis(df["annee"].tolist()):
        train = df[df["annee"].isin(annees_train)]
        test = df[df["annee"] == annee_test]
        if test.empty:
            continue
        sortie = test[["baie", "annee", CIBLE]].rename(columns={CIBLE: "vrai"}).copy()
        sortie["pred_annee_precedente"] = test["surface_n_1"].values
        moyennes = train.groupby("baie")[CIBLE].mean()
        sortie["pred_moyenne"] = test["baie"].map(moyennes).values
        Xtr, Xte = matrice(train, variables, baies), matrice(test, variables, baies)
        ytr = np.log1p(train[CIBLE].values)
        for nom, modele in modeles().items():
            modele.fit(Xtr, ytr)
            sortie["pred_" + nom] = np.expm1(modele.predict(Xte)).clip(min=0)
        resultats.append(sortie)
    return pd.concat(resultats, ignore_index=True)


def scores(pred: pd.DataFrame) -> pd.DataFrame:
    """Erreurs comparées, sur les lignes où toutes les méthodes ont une prédiction.

    - mae_ha : erreur absolue moyenne, en ha (dominée par Saint-Brieuc)
    - mae_relative : moyenne sur les baies de MAE_baie / surface moyenne de la baie
    - gain_vs_n_1 : 1 - mae_relative / mae_relative(année précédente) ; > 0 = mieux
    """
    cols = [c for c in pred.columns if c.startswith("pred_")]
    p = pred.dropna(subset=cols + ["vrai"])
    lignes = []
    for c in cols:
        err = (p[c] - p["vrai"]).abs()
        par_baie = err.groupby(p["baie"]).mean() / p.groupby("baie")["vrai"].mean()
        lignes.append({"methode": c[5:], "n": len(p), "mae_ha": err.mean(),
                       "mae_relative": par_baie.mean()})
    s = pd.DataFrame(lignes).set_index("methode")
    s["gain_vs_n_1"] = 1 - s["mae_relative"] / s.loc["annee_precedente", "mae_relative"]
    return s.sort_values("mae_relative")


def scores_par_baie(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE relative par baie et par méthode (mêmes lignes que `scores`)."""
    cols = [c for c in pred.columns if c.startswith("pred_")]
    p = pred.dropna(subset=cols + ["vrai"])
    moy = p.groupby("baie")["vrai"].mean()
    return pd.DataFrame({
        c[5:]: (p[c] - p["vrai"]).abs().groupby(p["baie"]).mean() / moy for c in cols
    }).assign(n=p.groupby("baie").size())
