import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algathon.ingest.hubeau import _matches  # noqa: E402
from algathon.ingest.sentinel2 import _parse_openeo_json  # noqa: E402


def test_river_matching_ignores_accents_and_requires_whole_word():
    assert _matches("Le Gouët à Saint-Julien", ["Gouet"])
    assert _matches("L'URNE A MAGENTA", ["Urne"])
    assert not _matches("RISQUE", ["Ris"])
    assert not _matches("Le Douron au Ponthou", ["Yar", "Roscoat"])


def test_parse_openeo_json():
    fc = {"features": [{"properties": {"site": "A"}}, {"properties": {"site": "B"}}]}
    js = {"2023-05-01T00:00:00Z": [[1.0, 10.0, 20.0], []]}
    df = _parse_openeo_json(js, fc)
    assert list(df["site"]) == ["A"] and df["ulva_ha"].iloc[0] == 1.0
