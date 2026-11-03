"""Central paths, random seed and project-wide assumptions."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SECOM_FILES = ("secom.data", "secom_labels.data", "secom.names")

SEED = 42
TEST_FRACTION = 0.2


@dataclass(frozen=True)
class CostMatrix:
    """Assumed misclassification costs; these are not measured values from a real line."""

    false_negative: float = 10.0
    false_positive: float = 1.0


COSTS = CostMatrix()
