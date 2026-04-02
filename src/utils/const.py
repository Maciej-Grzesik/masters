from pathlib import Path
import sys


SEED = 1410

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "AOD_4"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "resnet_transrate_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT


RESNET_MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]
