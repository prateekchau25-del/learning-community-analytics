"""
Run the whole analysis in order, for one community or for all of them.

Usage:
    python src/run_pipeline.py                         # every community + comparison
    python src/run_pipeline.py --site datascience      # one community
    python src/run_pipeline.py --site ai --from network
    python src/run_pipeline.py --site all --only figures

Input is read from PostgreSQL when .env is configured and reachable, otherwise from the
CSV files in data/raw/<site>/. Results go to outputs/<site>/ and outputs/comparison/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("preprocess", "clean text and build question features"),
    ("topics", "LDA + BERTopic topic modeling"),
    ("network", "knowledge network, centrality, communities"),
    ("diffusion", "diffusion tests and IC/LT simulation"),
    ("predict", "answer prediction models"),
    ("recommender", "expert recommender evaluation"),
    ("figures", "report figures"),
]
MODULES = {"preprocess": "preprocess", "topics": "topic_model", "network": "network", "diffusion": "diffusion",
           "predict": "predict", "recommender": "recommender", "figures": "figures"}
SITES = ["datascience", "ai"]


def run_site(steps):
    """Run steps for the community in LCA_SITE (set before this process imported config)."""
    import importlib

    from config import COMMUNITY_NAME
    from data_loader import data_source
    print(f"##### {COMMUNITY_NAME} (data source: {data_source()})\n")
    for name, desc in steps:
        print(f"=== {name}: {desc}", flush=True)
        t = time.time()
        importlib.import_module(MODULES[name]).run()
        print(f"=== {name} done in {time.time() - t:.0f}s\n", flush=True)


def main():
    names = [s[0] for s in STEPS]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="all", choices=SITES + ["all"])
    parser.add_argument("--from", dest="start", choices=names, default=names[0])
    parser.add_argument("--only", choices=names + ["compare"])
    args = parser.parse_args()
    total = time.time()

    if args.site == "all":
        # Each community runs in its own process, because the settings are fixed at import time.
        if args.only != "compare":
            for site in SITES:
                cmd = [sys.executable, __file__, "--site", site, "--from", args.start]
                if args.only:
                    cmd += ["--only", args.only]
                subprocess.run(cmd, check=True, env={**os.environ, "LCA_SITE": site, "PYTHONIOENCODING": "utf-8"})
        if args.only in (None, "compare", "figures"):
            print("=== compare: cross-community comparison", flush=True)
            sys.path.insert(0, str(Path(__file__).parent))
            import compare
            compare.run()
    else:
        os.environ["LCA_SITE"] = args.site
        sys.path.insert(0, str(Path(__file__).parent))
        selected = [s for s in STEPS if s[0] == args.only] if args.only else STEPS[names.index(args.start):]
        run_site(selected)

    print(f"Finished in {(time.time() - total) / 60:.1f} min. Results are in outputs/.")


if __name__ == "__main__":
    main()
