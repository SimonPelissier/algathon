"""Point d'entrée : python scripts/train.py --config configs/default.yaml"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algathon.data import load_table, make_synthetic  # noqa: E402
from algathon.features import build_features  # noqa: E402
from algathon.model import evaluate, temporal_split, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))

    src = cfg["data"].get("path")
    df = load_table(src) if src else make_synthetic(**cfg["data"].get("synthetic", {}))
    print(f"Données : {len(df)} lignes, {df['site'].nunique()} sites"
          f" ({'réelles' if src else 'synthétiques'})")

    f = cfg["features"]
    table = build_features(df, f["variables"], f["lags"], f["windows"], cfg["horizon_days"])
    feature_cols = [c for c in table.columns if c not in ("site", "date", "target")]

    tr, te = temporal_split(table, cfg["split"]["test_start"])
    model = train(tr, feature_cols, cfg["model"])
    for name, part in (("train", tr), ("test", te)):
        m = evaluate(model, part, feature_cols)
        print(f"[{name}] " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
