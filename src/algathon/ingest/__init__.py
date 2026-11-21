"""Ingestion des données : Hub'Eau (débits, nitrates), Sentinel-2 (CDSE openEO ou Earth Engine)."""


def use_system_certificates():
    """Utilise le magasin de certificats de l'OS (proxy/antivirus qui interceptent le HTTPS)."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
