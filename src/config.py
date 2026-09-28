"""Shared paths and settings for the whole pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE = "cs50"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / SITE
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Share of the timeline (oldest first) used for training in the prediction
# and recommender evaluations; the newest part is held out as the test set.
TRAIN_FRACTION = 0.8

for _d in (PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
