"""Affinage pixel par pixel : végétation permanente, liseré côtier et cartes de fréquence.

On relit le NDVI des seules scènes de basse mer retenues (une par baie et par mois,
plus celles des fenêtres de survol CEVA). Pour chaque pixel :
  n_valide   nombre de ces scènes où il est visible (sans nuage)
  n_ulve     nombre de ces scènes où NDVI > seuil
  frequence  n_ulve / n_valide

Un pixel détecté dans plus de `frequence_permanente` de ses scènes, d'avril à octobre
et toutes années confondues, n'est pas une ulve libre (elles bougent avec la marée et
d'une année à l'autre) : c'est une algue fixée sur la roche, un herbier ou un pixel mixte
terre/mer. Il est exclu, ainsi qu'un liseré de `erosion_pixels` le long du bord de la zone.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_erosion

from ..ingestion.cache import DOSSIER_RAW
from . import sentinel2 as s2

INVALIDE = -128


def cle_grille(px: dict) -> Tuple:
    return (px["crs"], tuple(round(v, 3) for v in px["transform"]), tuple(px["forme"]))


def pixels_scene(conf: dict, code_baie: str, item: dict, geoms: List[dict], emprise,
                 force: bool = False) -> dict:
    """NDVI pixel par pixel d'une scène (int8, en centièmes), mis en cache en .npz."""
    chemin = DOSSIER_RAW / "sentinel2" / "pixels" / code_baie / (item["id"] + ".npz")
    if chemin.exists() and not force:
        z = np.load(str(chemin), allow_pickle=False)
        return {"ndvi_i8": z["ndvi_i8"], "decouvert": np.unpackbits(z["decouvert"])[
                    :int(np.prod(z["forme"]))].reshape(tuple(z["forme"])).astype(bool),
                "crs": str(z["crs"]), "transform": tuple(z["transform"]), "forme": tuple(z["forme"])}
    px = s2.analyser_scene(conf, code_baie, item, geoms, emprise, pixels=True)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.stem + ".tmp.npz")
    np.savez_compressed(str(tmp), ndvi_i8=px["ndvi_i8"], decouvert=np.packbits(px["decouvert"]),
                        crs=np.array(px["crs"]),
                        transform=np.array(px["transform"]), forme=np.array(px["forme"]))
    tmp.replace(chemin)
    return px


def ramener_sur_grille(masque: np.ndarray, src_px: dict, dst_px: dict) -> np.ndarray:
    """Reprojette un masque booléen d'une grille de scène vers une autre (plus proche voisin)."""
    if cle_grille(src_px) == cle_grille(dst_px):
        return masque
    out = np.zeros(tuple(dst_px["forme"]), dtype="uint8")
    reproject(masque.astype("uint8"), out,
              src_transform=Affine(*src_px["transform"]), src_crs=src_px["crs"],
              dst_transform=Affine(*dst_px["transform"]), dst_crs=dst_px["crs"],
              resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)
    return out.astype(bool)


def cartes_baie(pixels: Dict[str, dict], annees: Dict[str, int], seuil: float,
                conf_aff: dict) -> dict:
    """Fréquences et masque d'exclusion sur la grille la plus fréquente de la baie.

    pixels : {id de scène: pixels}, annees : {id de scène: année}.
    """
    grilles = Counter(cle_grille(p) for p in pixels.values())
    cle = grilles.most_common(1)[0][0]
    ref = next(p for p in pixels.values() if cle_grille(p) == cle)
    forme = tuple(ref["forme"])
    n_valide = np.zeros(forme, dtype="int32")
    n_ulve = np.zeros(forme, dtype="int32")
    n_decouvert = np.zeros(forme, dtype="int32")
    n_ulve_decouvert = np.zeros(forme, dtype="int32")
    par_annee: Dict[int, List[np.ndarray]] = {}
    seuil_i8 = int(round(seuil * 100))
    for sid, p in pixels.items():
        if cle_grille(p) != cle:
            continue  # scènes d'une autre tuile : pas sur la même grille, non cumulées
        v = p["ndvi_i8"] != INVALIDE
        u = v & (p["ndvi_i8"] > seuil_i8)
        n_valide += v
        n_ulve += u
        n_decouvert += p["decouvert"]
        n_ulve_decouvert += u & p["decouvert"]
        a = annees[sid]
        if a not in par_annee:
            par_annee[a] = [np.zeros(forme, "int16"), np.zeros(forme, "int16")]
        par_annee[a][0] += v
        par_annee[a][1] += u
    with np.errstate(invalid="ignore", divide="ignore"):
        frequence = np.where(n_valide > 0, n_ulve / n_valide, np.nan)
        freq_annees = {a: np.where(c[0] > 0, c[1] / c[0], np.nan) for a, c in par_annee.items()}
    zone = n_valide > 0
    permanent = (n_valide >= conf_aff["n_scenes_min"]) & (frequence >= conf_aff["frequence_permanente"])
    # Algues fixées sur roche : détectées presque à chaque fois que la marée les découvre,
    # même si elles ne sont découvertes qu'aux plus basses mers (fréquence globale faible).
    fd = conf_aff.get("fixe_si_decouvert")
    if fd:
        with np.errstate(invalid="ignore", divide="ignore"):
            freq_dec = np.where(n_decouvert > 0, n_ulve_decouvert / n_decouvert, 0.0)
        permanent |= (n_decouvert >= fd["n_decouvert_min"]) & (freq_dec >= fd["frequence_min"])
    # Second critère : un pixel détecté (fréquence annuelle >= seuil) presque chaque année
    # est fixe (algues sur roche visibles seulement aux plus basses mers, herbiers).
    pa = conf_aff.get("persistance_annuelle")
    if pa and len(freq_annees) >= 4:
        empile = np.stack(list(freq_annees.values()))
        vues = np.isfinite(empile).sum(axis=0)
        detectees = (np.nan_to_num(empile, nan=0.0) >= pa["frequence_min"]).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            part = np.where(vues >= 4, detectees / vues, 0.0)
        permanent |= part >= pa["part_annees"]
    e = conf_aff["erosion_pixels"]
    lisere = zone & ~binary_erosion(zone, iterations=e) if e > 0 else np.zeros(forme, bool)
    return {"ref": ref, "zone": zone, "n_valide": n_valide, "frequence": frequence,
            "permanent": permanent, "lisere": lisere, "exclusion": permanent | lisere,
            "freq_annees": freq_annees}


def surfaces_corrigees(px: dict, exclusion: np.ndarray, ref: dict, seuils) -> dict:
    """Surfaces d'ulves d'une scène (ha) hors pixels exclus, pour chaque seuil."""
    excl = ramener_sur_grille(exclusion, ref, px)
    v = (px["ndvi_i8"] != INVALIDE) & ~excl
    res = {"valid_corr_ha": float(v.sum() * s2.HA_PAR_PIXEL)}
    for s in seuils:
        res["ulva_corr_s%03d" % round(s * 100)] = float(
            (v & (px["ndvi_i8"] > int(round(s * 100)))).sum() * s2.HA_PAR_PIXEL)
    return res


def enregistrer_cartes(code_baie: str, cartes: dict, dossier) -> None:
    """Sauve les cartes d'une baie (pour le notebook 03), hors dépôt."""
    dossier.mkdir(parents=True, exist_ok=True)
    annees = sorted(cartes["freq_annees"])
    np.savez_compressed(
        str(dossier / (code_baie + ".npz")),
        frequence=cartes["frequence"].astype("float32"),
        n_valide=cartes["n_valide"].astype("int16"),
        permanent=cartes["permanent"], lisere=cartes["lisere"], zone=cartes["zone"],
        annees=np.array(annees),
        freq_annees=np.stack([cartes["freq_annees"][a].astype("float32") for a in annees]),
        crs=np.array(cartes["ref"]["crs"]), transform=np.array(cartes["ref"]["transform"]),
    )
