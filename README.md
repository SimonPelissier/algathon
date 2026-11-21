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

## L'Observatoire (application cartographique)

Carte interactive de la Bretagne et de la façade atlantique (Cotentin → Vendée) :
- **curseur temporel 2002 → 2027** : surfaces d'ulves observées (CEVA) jusqu'en 2022, estimées ensuite, prévues pour 2027 ;
- **couches** : surface d'ulves par plage (colonnes 3D), nitrates des rivières, densité d'élevages (ICPE), flux de nitrates vers les baies pilotes ;
- **scénarios 2027** : type d'hiver (doux / médian / rude), variation des nitrates et des élevages ;
- **plages ciblées** : classement exportable des plages à risque ;
- **suivi Sentinel-2** des baies pilotes.

```bash
streamlit run app/streamlit_app.py
```

Pipeline complet pour régénérer les données de l'application :
```bash
python scripts/fetch_ceva.py
python scripts/fetch_regional.py            # météo NASA POWER, nitrates Hub'Eau, élevages Géorisques
python scripts/fetch_hubeau.py              # flux de nitrates des baies pilotes
python scripts/fetch_s2.py --backend cdse   # surfaces d'ulves Sentinel-2 (compte CDSE)
python scripts/train_seasonal.py            # modèle saisonnier + prévisions
python scripts/export_app_data.py           # données légères pour app/data/
```

## Démarrage rapide

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
# Python du Microsoft Store : créer plutôt le venv dans un chemin court
# (ex. %USERPROFILE%\.venvs\algathon), sinon l'installation de streamlit échoue
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

Sans données réelles, `train.py` génère un **jeu synthétique** pour tester le pipeline de bout en bout.

Récupération des données réelles (comptes, quotas, dépannage SSL) : voir [docs/acces_donnees.md](docs/acces_donnees.md).

```bash
python scripts/fetch_ceva.py                    # sites + cumuls annuels CEVA
python scripts/fetch_hubeau.py                  # débits + nitrates par baie
python scripts/fetch_s2.py --backend cdse       # surfaces d'ulves Sentinel-2
```

## Feuille de route

- [x] Squelette du pipeline (indices, features, baseline, validation temporelle)
- [x] Cible : **marées vertes, prévision des échouages à court terme (J+3 à J+10)** pour les communes
- [x] Baies pilotes : Saint‑Brieuc (Morieux, Yffiniac, Binic), Lannion (Saint‑Michel‑en‑Grève), Douarnenez — voir [configs/sectors.yaml](configs/sectors.yaml)
- [x] Vérité terrain CEVA ouverte : contours des sites + cumuls annuels 2002‑2022 (`scripts/fetch_ceva.py`)
- [x] Débits + nitrates des rivières par baie via Hub'Eau (`scripts/fetch_hubeau.py`)
- [ ] Surfaces d'ulves Sentinel‑2 par site et par date — code prêt (`scripts/fetch_s2.py`, CDSE ou Earth Engine), à lancer avec les comptes
- [ ] Calibrer le seuil NDVI avec les cumuls annuels CEVA ; demander au CEVA les données par inventaire
- [ ] Ajouter SST, météo (vent, rayonnement), coefficients de marée
- [ ] Baseline avec métriques PR‑AUC et F1 par horizon
- [ ] Cartes de risque et tableau de bord
