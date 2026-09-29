"""Shared paths and settings for the whole pipeline.

The community being analysed comes from the LCA_SITE environment variable
(run_pipeline.py sets it from --site), so every module works on one community.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stack Exchange site -> display name
COMMUNITIES = {"datascience": "Data Science", "ai": "Artificial Intelligence"}
SITE = os.getenv("LCA_SITE", "datascience")
COMMUNITY_NAME = COMMUNITIES.get(SITE, SITE)


def site_dirs(site: str) -> dict[str, Path]:
    return {
        "raw": PROJECT_ROOT / "data" / "raw" / site,
        "processed": PROJECT_ROOT / "data" / "processed" / site,
        "outputs": PROJECT_ROOT / "outputs" / site,
        "figures": PROJECT_ROOT / "outputs" / site / "figures",
    }


_dirs = site_dirs(SITE)
RAW_DIR = _dirs["raw"]
PROCESSED_DIR = _dirs["processed"]
OUTPUT_DIR = _dirs["outputs"]
FIGURE_DIR = _dirs["figures"]
COMPARISON_DIR = PROJECT_ROOT / "outputs" / "comparison"

RANDOM_SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Share of the timeline (oldest first) used for training in the prediction
# and recommender evaluations; the newest part is held out as the test set.
TRAIN_FRACTION = 0.8

for _d in (PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR, COMPARISON_DIR):
    _d.mkdir(parents=True, exist_ok=True)
