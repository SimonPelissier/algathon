# Algathon — Prédiction des proliférations d'algues

Modèle de machine learning qui estime le **risque d'apparition ou de prolifération de micro‑algues et de macro‑algues**. Il combine des **images satellitaires** (Sentinel‑2, Sentinel‑3, Landsat) et des **données locales** (température de l'eau, nutriments, météo, débits des rivières, marées).

## À quoi sert le projet

Les proliférations d'algues coûtent cher. Elles touchent la santé, l'économie et les écosystèmes, et on les détecte souvent trop tard, quand l'algue est déjà là. Le projet vise à **anticiper** ces épisodes de quelques jours à quelques semaines, pour pouvoir agir avant.

| Phénomène | Exemples | Enjeux | Qui en a besoin |
|---|---|---|---|
| **Micro‑algues toxiques (HAB)** | *Alexandrium*, *Pseudo‑nitzschia*, *Dinophysis* | Fermeture de la conchyliculture, toxines (ASP/PSP/DSP), santé publique | Conchyliculteurs, Ifremer (REPHY), ARS |
| **Cyanobactéries d'eau douce** | *Microcystis*, *Planktothrix* | Baignade interdite, eau potable, mortalité animale | ARS, syndicats d'eau, collectivités |
| **Marées vertes (macro‑algues)** | *Ulva* (Bretagne) | Échouages, H₂S toxique, coûts de ramassage, tourisme | Communes, CEVA, préfectures |
| **Sargasses** | *Sargassum* (Antilles, Guyane) | Échouages massifs, H₂S, pêche, tourisme | Collectivités d'outre‑mer, Météo‑France |

**Ce que le modèle produit** : pour chaque zone (pixel, maille ou site) et chaque date, une **probabilité de bloom à J+h**, avec des cartes de risque et une analyse des facteurs explicatifs (rôle de la température, des nutriments, etc.).

**Utilisations concrètes** :
- **Alerte précoce** : déclencher plus tôt les prélèvements de contrôle sanitaire.
- **Logistique** : prévoir les moyens de ramassage sur les plages.
- **Aide à la décision** : relier les blooms aux apports agricoles (nitrates), ce qui sert aux politiques bassin versant.
- **Valorisation** : repérer la biomasse récoltable (biostimulants, bioplastiques, méthanisation).

## Données envisagées

| Source | Variables | Accès |
|---|---|---|
| Sentinel‑2 MSI (10–20 m) | Bandes B2–B12, puis indices NDVI, FAI, NDCI, MCI | Copernicus Data Space, Google Earth Engine |
| Sentinel‑3 OLCI (300 m) | Chlorophylle‑a, MCI, phycocyanine | Copernicus Marine / EUMETSAT |
| SST (MODIS, OSTIA) | Température de surface | Copernicus Marine |
| Météo (ERA5) | Vent, rayonnement, précipitations | Copernicus CDS |
| Hydrologie | Débits (Hydroportail), nitrates (Naïades) | Open data FR |
| Vérité terrain | REPHY/REPHYTOX (Ifremer), contrôles ARS, survols CEVA | Open data |

## Architecture

```
configs/         paramètres (zone, horizon, modèle)
data/raw/        données brutes (non versionnées)
data/processed/  table de features
src/algathon/
  indices.py     indices spectraux (NDVI, FAI, NDCI, MCI…)
  data.py        chargement des données + générateur synthétique pour démo
  features.py    features temporelles (lags, moyennes glissantes, anomalies)
  model.py       entraînement, validation temporelle, évaluation
scripts/train.py point d'entrée
tests/           tests unitaires
```

**Approche de base** : gradient boosting sur des features tabulaires (indices spectraux et variables locales, avec leurs lags), validé par un **découpage temporel** pour éviter la fuite de données (on n'entraîne jamais sur le futur). Des extensions sont prévues : ConvLSTM ou U‑Net spatio‑temporel sur des séries d'images.

## Démarrage rapide

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

Sans données réelles, `train.py` génère un **jeu synthétique** pour tester le pipeline de bout en bout.

## Feuille de route

- [ ] Choisir le cas d'usage prioritaire et la zone d'étude
- [ ] Ingestion Sentinel‑2/3 (Earth Engine ou openEO)
- [ ] Récupérer la vérité terrain (REPHY, ARS…) et l'aligner avec les images
- [ ] Baseline gradient boosting, avec métriques PR‑AUC et F1 par horizon
- [ ] Modèle spatio‑temporel (images)
- [ ] Cartes de risque et tableau de bord
