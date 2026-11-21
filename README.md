# Algathon — Prédire les proliférations d'algues en Bretagne

Modèle de machine learning qui estime le **risque d'apparition ou de prolifération d'algues sur le littoral breton**, quelques jours à quelques semaines à l'avance. Il combine des **images satellitaires** (Sentinel‑2, Sentinel‑3) et des **données locales** (température de l'eau, nitrates des rivières, météo, débits, marées).

## À quoi sert le projet

La Bretagne est la région de France la plus touchée par deux phénomènes :

### 1. Les marées vertes (macro‑algues, *Ulva*)
- **Le problème** : chaque printemps et chaque été, des milliers de tonnes d'ulves s'échouent dans les baies sableuses peu profondes : Saint‑Brieuc, Lannion, Douarnenez, la Forêt‑Fouesnant, etc. En se décomposant, elles dégagent de l'**H₂S**, un gaz potentiellement mortel (décès d'animaux et humains documentés). Le ramassage coûte plusieurs millions d'euros par an aux communes, et le tourisme en pâtit.
- **La cause** : les apports de **nitrates d'origine agricole** par les rivières, combinés à l'ensoleillement, à la température de l'eau et au confinement des baies.
- **Ce que le modèle apporte** : anticiper les surfaces d'échouage par baie pour planifier le ramassage et protéger le public. Il permet aussi de quantifier l'effet des flux de nitrates, ce qui éclaire les Plans de lutte contre les algues vertes (PLAV).

### 2. Les micro‑algues toxiques (HAB)
- **Le problème** : *Pseudo‑nitzschia* (toxine ASP), *Alexandrium* (PSP) et *Dinophysis* (DSP) contaminent les coquillages. Leurs blooms entraînent des **fermetures de zones conchylicoles** (rade de Brest, baie de Concarneau, baie de Morlaix…) et des pertes importantes pour la filière. On observe aussi des « eaux vertes » à *Lepidodinium* en Bretagne Sud.
- **Ce que le modèle apporte** : une alerte précoce pour cibler les prélèvements du réseau REPHY et prévenir les producteurs.

**Ce que le modèle produit** : pour chaque baie (ou pixel) et chaque date, une **probabilité de bloom à J+h**, des cartes de risque et l'importance des facteurs explicatifs.

**Utilisateurs potentiels** : communes et intercommunalités, CEVA, préfectures, ARS Bretagne, Ifremer, comités régionaux conchylicoles, et filières de valorisation des ulves (biostimulants, alimentation animale, méthanisation).

## Données envisagées

| Source | Variables | Usage | Accès |
|---|---|---|---|
| Sentinel‑2 MSI (10 m, revisite ~5 j) | Bandes B2–B12, puis indices NDVI, FAI | Surfaces d'ulves en baie | Copernicus Data Space, Earth Engine |
| Sentinel‑3 OLCI (300 m) | Chlorophylle‑a, MCI | Blooms de phytoplancton au large | Copernicus Marine |
| SST (OSTIA / MODIS) | Température de surface | Les deux | Copernicus Marine |
| ERA5 / Météo‑France | Vent, rayonnement, pluie | Les deux | Copernicus CDS, meteo.data.gouv.fr |
| Hydroportail, Naïades | Débits et nitrates des rivières bretonnes | Moteur des marées vertes | Open data FR |
| SHOM | Coefficients de marée | Échouages, observabilité | data.shom.fr |
| **CEVA** | Surfaces d'échouage d'ulves (survols) | **Vérité terrain macro‑algues** | Bilans CEVA |
| **REPHY / REPHYTOX (Ifremer)** | Comptages de phytoplancton, toxines | **Vérité terrain micro‑algues** | SEANOE, Quadrige |

Attention : la couverture nuageuse est fréquente en Bretagne, donc beaucoup de jours n'ont pas d'image. Le pipeline gère ces trous (dernière valeur observée, moyennes glissantes).

## Architecture

```
configs/         paramètres (zone, horizon, modèle)
data/raw/        données brutes (non versionnées)
data/processed/  table de features
src/algathon/
  indices.py     indices spectraux (NDVI, FAI, NDCI, MCI)
  data.py        chargement des données + générateur synthétique pour démo
  features.py    features temporelles (lags, moyennes glissantes, anomalies)
  model.py       entraînement, validation temporelle, évaluation
scripts/train.py point d'entrée
tests/           tests unitaires
```

**Approche de base** : gradient boosting sur des features tabulaires (indices spectraux et variables locales, avec leurs lags), validé par un **découpage temporel** (on n'entraîne jamais sur le futur). Des extensions sont prévues : U‑Net pour segmenter les ulves sur les images Sentinel‑2, et un modèle spatio‑temporel de type ConvLSTM.

## Démarrage rapide

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

Sans données réelles, `train.py` génère un **jeu synthétique** pour tester le pipeline de bout en bout.

## Marées vertes : prévision annuelle des échouages par baie

Premier maillon réel du projet, sur des données publiques et sans aucune clé d'API.
Les 7 baies suivies (Saint-Brieuc, Lieue de Grève, Fresnaye, Horn-Guillec, Guissény,
Douarnenez, Concarneau et La Forêt-Fouesnant) sont décrites dans `configs/baies.yaml`.

```bash
python scripts/ingest.py           # télécharge CEVA, Hub'Eau, Open-Meteo dans data/raw/ (cache)
python scripts/baseline_annuel.py  # Ridge et gradient boosting contre deux références naïves
pytest tests/
```

| Fichier | Rôle |
|---|---|
| `src/algathon/ingestion/` | téléchargements avec cache : `ceva.py` (WFS GéoSAS), `hubeau.py` (débits, nitrates), `meteo.py` (Open-Meteo) |
| `src/algathon/series_annuelles.py` | table baie × année : surface CEVA, volumes ramassés, débits, nitrates, flux d'azote, météo |
| `src/algathon/baseline_annuel.py` | modèles et validation par année (leave-one-year-out et fenêtre glissante) |
| `notebooks/01_exploration_annuelle.ipynb` | séries, corrélations, nombre de points par baie |
| `src/algathon/detection/` | surfaces d'ulves Sentinel-2 par scène (`sentinel2.py`), choix de la scène de basse mer et comparaison au CEVA (`calibration.py`) |
| `notebooks/02_detection_sentinel2.ipynb` | disponibilité des scènes, effet de la marée, calibration du seuil NDVI |
| `src/algathon/detection/affinage.py`, `scripts/affiner_s2.py` | exclusion des algues fixées et du liseré côtier, cartes de fréquence |
| `scripts/prevision_5j_experimentale.py` | expérimental : prévision tous les 5 jours (image claire + marée prévue + météo), cumuls comparés au CEVA |
| `notebooks/05_prevision_5j.ipynb` | qualité de la prévision à 5 jours, saison 2019, cumuls de saison contre CEVA ; houle hivernale testée et écartée |
| `notebooks/03_cartes_et_evolutions.ipynb` | carte des baies et stations, cartes d'échouage, saisonnalité, série 2002-2025, écarts au CEVA |
| `data/README.md` | sources, formats, périodes, comment régénérer les données |

Ce qu'il faut savoir : le CEVA ne publie qu'**une surface cumulée par site et par an**
(2002-2022). On a donc environ 20 points par baie, d'où un modèle unique pour toutes les baies,
validé uniquement par découpage temporel. Une prévision mensuelle demandera la détection
satellite (Sentinel-2) pour obtenir des surfaces datées.

## Feuille de route

- [x] Squelette du pipeline (indices, features, baseline, validation temporelle)
- [ ] Choisir la cible prioritaire : marées vertes (*Ulva*) ou micro‑algues toxiques
- [ ] Choisir les baies pilotes (ex. Saint‑Brieuc, Lannion, Douarnenez)
- [ ] Ingestion Sentinel‑2 sur ces baies (Earth Engine ou openEO), avec masque nuages
- [ ] Récupérer la vérité terrain (CEVA et/ou REPHY) et l'aligner avec les images
- [ ] Ajouter les variables locales : nitrates/débits des rivières, SST, météo, marées
- [ ] Baseline avec métriques PR‑AUC et F1 par horizon
- [ ] Cartes de risque et tableau de bord
