"""Téléchargements HTTP avec cache disque.

Chaque réponse est écrite une fois dans data/raw/<source>/<fichier>.json.
Un second appel relit le fichier au lieu d'interroger l'API (idempotence).
`force=True` retélécharge et remplace le fichier.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import requests

RACINE = Path(__file__).resolve().parents[3]
DOSSIER_RAW = RACINE / "data" / "raw"

# Codes HTTP pour lesquels une nouvelle tentative a du sens (surcharge, panne passagère).
CODES_A_RETENTER = {429, 500, 502, 503, 504}
ENTETES = {"User-Agent": "algathon-ingestion/0.1 (projet hackathon open source)"}


def get_json(url: str, params: Optional[dict] = None, essais: int = 5,
             timeout: int = 180) -> dict:
    """GET qui renvoie du JSON, avec nouvelles tentatives et attente croissante.

    Hub'Eau est parfois lent ou coupe la connexion : on retente jusqu'à `essais` fois.
    """
    derniere_erreur = None
    for i in range(essais):
        try:
            r = requests.get(url, params=params, headers=ENTETES, timeout=timeout)
            if r.status_code in CODES_A_RETENTER:
                raise requests.HTTPError("HTTP %d sur %s" % (r.status_code, r.url))
            r.raise_for_status()  # Hub'Eau renvoie 206 quand il reste des pages : c'est un succès
            return r.json()
        except (requests.RequestException, ValueError) as e:
            derniere_erreur = e
            if i < essais - 1:
                time.sleep(3 * 2 ** i)
    raise RuntimeError("Échec après %d essais : %s" % (essais, derniere_erreur))


def json_en_cache(chemin: Path, producteur: Callable[[], object], force: bool = False):
    """Renvoie le contenu de `chemin` s'il existe, sinon appelle `producteur` et l'enregistre.

    L'écriture passe par un fichier temporaire : une interruption ne laisse jamais
    un cache à moitié écrit qui serait relu comme valide.
    """
    if chemin.exists() and not force:
        return json.loads(chemin.read_text(encoding="utf-8"))
    donnees = producteur()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.name + ".tmp")
    tmp.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")
    tmp.replace(chemin)
    return donnees
