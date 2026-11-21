# Données

Rien dans `data/raw/` ni dans `data/processed/` n'est versionné (voir `.gitignore`).
Tout se régénère avec :

```bash
pip install -r requirements.txt
python scripts/ingest.py           # ~2 min la première fois, quelques secondes ensuite (cache)
python scripts/ingest.py --force   # retélécharge tout (pour prolonger les séries)
```

Aucun compte ni aucune clé n'est nécessaire. Le cache pèse environ 9 Mo.

## Contenu de `data/raw/` (cache brut, une réponse d'API par fichier)

| Dossier | Source | Contenu | Période | Granularité |
|---|---|---|---|---|
| `ceva/prolif_AV_site.geojson` | CEVA via WFS GéoSAS, couche `greenseas:prolif_AV_site`, Licence Ouverte | 129 sites (95 baies sableuses, 34 vasières), champs `cumul_2002` … `cumul_2022` | 2002-2022 | **Un cumul annuel par site** (somme des survols de mai, juillet et septembre), en ha |
| `ceva/enquete_communale_AV_bretagne.geojson` | CEVA via WFS GéoSAS, couche `greenseas:enquete_communale_AV_bretagne` | 279 communes, champs `VOL_1997` … `VOL_2022` | 1997-2022 | Volume ramassé par commune et par an, en m³ |
| `hubeau/debits/<station>_QmnJ.json` | Hub'Eau hydrométrie, `obs_elab`, grandeur QmnJ | Débit moyen journalier (l/s dans le fichier) | 2001 → date du téléchargement | Journalier |
| `hubeau/nitrates/<station>_NO3.json` | Hub'Eau qualité des cours d'eau, `analyse_pc`, paramètre Sandre 1340 | Concentrations en nitrates, mg(NO3)/L | 2001 → date du téléchargement, selon la station | 15 à 25 prélèvements par an |
| `meteo/openmeteo_<lat>_<lon>.json` | Open-Meteo Archive (réanalyse ERA5) | Température moyenne, rayonnement global, pluie | 2001 → J-7 | Journalier |

Liens directs :
- WFS GéoSAS : `https://geosas.fr/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=greenseas:prolif_AV_site&outputFormat=SHAPE-ZIP`
  (remplacer `typeName` par `greenseas:enquete_communale_AV_bretagne` pour les volumes ramassés).
- Fiche GéoSAS : https://geosas.fr/geonetwork/srv/api/records/4c9c050c-40ac-48e4-8f20-f7cd4b9ed6e1
- Hub'Eau : https://hubeau.eaufrance.fr/page/api-hydrometrie et https://hubeau.eaufrance.fr/page/api-qualite-cours-deau
- Open-Meteo : https://open-meteo.com/en/docs/historical-weather-api

Le jeu PNDB « Enquêtes communales CEVA sur le ramassage des algues vertes »
(pndb.opendatasoft.com) est **vide** (0 enregistrement au 2026-11-21) : on lit la même
donnée sur GéoSAS.

## Contenu de `data/processed/`

- `baies_annuel.csv` : une ligne par baie et par année (2002-2022), construite par
  `src/algathon/series_annuelles.py`. Les baies, leurs sites CEVA, communes et stations sont
  définis dans `configs/baies.yaml`.

  | Colonne | Signification |
  |---|---|
  | `surface_ha` | cible : somme des surfaces CEVA des sites de la baie (NaN si des sites pesant plus de 10 % manquent) |
  | `volume_m3` | volumes ramassés, somme des communes de la baie |
  | `surface_n_1`, `volume_n_1` | valeurs de l'année précédente |
  | `debit_<fenêtre>` | débit moyen, m³/s, somme des rivières de la baie |
  | `no3_<fenêtre>` | nitrates, mg/L, moyenne pondérée par le débit |
  | `flux_n_<fenêtre>_t` | flux d'azote nitrique, t N |
  | `ray_*`, `temp_*`, `pluie_*` | rayonnement (MJ/m², cumul), température (°C, moyenne), pluie (mm, cumul) |
  | `debit_proxy` | `True` si le débit vient d'un bassin voisin (Guissény) |

  Fenêtres : `jan_avr` (connu au 1er mai, utilisable pour prévoir), `mai_aout` (la saison
  elle-même), `mars_avr` et `mars_mai` pour la météo.

- `predictions_<jeu>_<validation>.csv` : prédictions hors échantillon de `scripts/baseline_annuel.py`.

## Limites connues

- Le CEVA ne publie que des **cumuls annuels** : pas de surface par date de survol, et rien après 2022.
- Les stations de débit sont en amont des estuaires : les flux couvrent une partie seulement
  des bassins versants. Ils servent à comparer les années entre elles, pas à faire un bilan.
- Guissény : la station du Quillimadec est fermée depuis 1982 ; on utilise l'Aber Benoît comme substitut.
- Douarnenez (Kerharo, depuis 2013) et Fresnaye (Frémur d'Hénanbihen, depuis 2010) ont des débits courts.
- Valeurs CEVA à vérifier : Fresnaye à 0 ha en 2006 et de 2010 à 2014 ; beaucoup de sites à 0 en 2022.

## Dépannage

Sur un Python installé depuis python.org (macOS), si les téléchargements échouent sur une
erreur de certificat SSL, lancer une fois `Install Certificates.command` du dossier Python.
