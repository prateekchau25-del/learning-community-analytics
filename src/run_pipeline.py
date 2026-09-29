"""
Run the whole analysis in order.

Usage:
    python src/run_pipeline.py                 # everything
    python src/run_pipeline.py --from network  # start at a later step
    python src/run_pipeline.py --only figures  # a single step

Input is read from PostgreSQL when .env is configured and reachable, otherwise from the
CSV files in data/raw/. Results go to outputs/ (and to PostgreSQL via load_to_postgres.py).
"""

import argparse
import time

import diffusion
import figures
import network
import predict
import preprocess
import recommender
import topic_model
from data_loader import data_source

STEPS = [
    ("preprocess", preprocess.run, "clean text and build question features"),
    ("topics", topic_model.run, "LDA + BERTopic topic modeling (~5 min)"),
    ("network", network.run, "knowledge network, centrality, communities (~1 min)"),
    ("diffusion", diffusion.run, "diffusion tests and IC/LT simulation (~8 min)"),
    ("predict", predict.run, "answer prediction models (~1 min)"),
    ("recommender", recommender.run, "expert recommender evaluation"),
    ("figures", figures.run, "report figures"),
]


def main():
    names = [s[0] for s in STEPS]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=names, default=names[0])
    parser.add_argument("--only", choices=names)
    args = parser.parse_args()

    selected = [s for s in STEPS if s[0] == args.only] if args.only else STEPS[names.index(args.start):]
    print(f"Data source: {data_source()}\n")
    total = time.time()
    for name, func, desc in selected:
        print(f"=== {name}: {desc}")
        t = time.time()
        func()
        print(f"=== {name} done in {time.time() - t:.0f}s\n")
    print(f"Pipeline finished in {(time.time() - total) / 60:.1f} min. Results are in outputs/.")


if __name__ == "__main__":
    main()
