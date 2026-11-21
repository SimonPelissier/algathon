import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algathon.ingest import use_system_certificates  # noqa: E402

use_system_certificates()
