# Accès aux données

| Donnée | Script | Compte requis |
|---|---|---|
| Sites et cumuls annuels CEVA (2002-2022) | `python scripts/fetch_ceva.py` | non |
| Débits + nitrates des rivières (Hub'Eau) | `python scripts/fetch_hubeau.py` | non |
| Surfaces d'ulves Sentinel-2 (CDSE) | `python scripts/fetch_s2.py --backend cdse` | oui (gratuit) |
| Surfaces d'ulves Sentinel-2 (Earth Engine) | `python scripts/fetch_s2.py --backend gee --project <ID>` | oui (gratuit) |

## Copernicus Data Space Ecosystem (CDSE)
1. Créer un compte sur https://dataspace.copernicus.eu (immédiat).
2. Au premier lancement de `fetch_s2.py --backend cdse`, un lien s'affiche : l'ouvrir et se connecter.
   Le jeton est ensuite mis en cache.
3. Quota gratuit : crédits openEO mensuels, suffisants pour quelques baies sur plusieurs années.
   Tester d'abord sur une courte période : `--sector lannion --start 2023-05-01 --end 2023-06-30`.

## Google Earth Engine
1. https://earthengine.google.com → *Get started* → créer un projet Google Cloud et l'enregistrer
   pour un usage **non commercial / recherche** (la validation peut prendre du temps).
2. `pip install earthengine-api`, puis lancer `fetch_s2.py --backend gee --project <ID_PROJET>` :
   l'authentification s'ouvre dans le navigateur au premier lancement.

## Données terrain
- **CEVA** : couche `greenseas:prolif_AV_site` sur GéoSAS (Licence Ouverte) = contours des sites
  et **cumuls annuels** de surface (ha). Les données **par inventaire** (survols mai/juillet/septembre
  + mensuels sur les grandes baies) sont dans les rapports PDF ; pour le CSV : algue@ceva.fr.
- Usage : calibrer le seuil NDVI de détection Sentinel-2 en comparant aux cumuls annuels.

## Problème de certificats SSL (Windows)
Si pip ou Python échouent avec `CERTIFICATE_VERIFY_FAILED` :
`pip install --use-feature=truststore -r requirements.txt`. Les scripts activent ensuite
automatiquement `truststore`.
