"""Indices spectraux pour la détection d'algues (bandes Sentinel-2 en réflectance)."""
import numpy as np

EPS = 1e-6


def ndvi(red, nir):
    """Normalized Difference Vegetation Index — macro-algues flottantes / échouées."""
    return (nir - red) / (nir + red + EPS)


def ndci(red, red_edge1):
    """Normalized Difference Chlorophyll Index (B5, B4) — chlorophylle-a, eaux turbides."""
    return (red_edge1 - red) / (red_edge1 + red + EPS)


def fai(red, nir, swir, wl_red=665.0, wl_nir=842.0, wl_swir=1610.0):
    """Floating Algae Index (Hu, 2009) — sargasses, Ulva flottantes."""
    baseline = red + (swir - red) * (wl_nir - wl_red) / (wl_swir - wl_red)
    return nir - baseline


def mci(red, red_edge1, red_edge2, wl_red=665.0, wl_re1=705.0, wl_re2=740.0):
    """Maximum Chlorophyll Index — blooms denses de phytoplancton."""
    baseline = red + (red_edge2 - red) * (wl_re1 - wl_red) / (wl_re2 - wl_red)
    return red_edge1 - baseline


def compute_all(bands: dict) -> dict:
    """bands: dict avec clés B4 (red), B5, B6, B8 (nir), B11 (swir)."""
    b = {k: np.asarray(v, dtype=float) for k, v in bands.items()}
    return {
        "ndvi": ndvi(b["B4"], b["B8"]),
        "ndci": ndci(b["B4"], b["B5"]),
        "fai": fai(b["B4"], b["B8"], b["B11"]),
        "mci": mci(b["B4"], b["B5"], b["B6"]),
    }
