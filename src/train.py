"""Fit all models on ALL ratings and save the router for the app (FR-10).

Run from the project root:  python -m src.train
The evaluation (src/evaluate.py) uses a train/test split; this script does not,
because the demo app should use every rating it has.
"""
import pandas as pd

from src import config
from src.router import HybridRouter


def main():
    ratings = pd.read_parquet(config.PROCESSED / "ratings.parquet")
    items = pd.read_parquet(config.PROCESSED / "items.parquet")
    router = HybridRouter().fit(ratings, items)
    router.save()
    print(f"fitted on {len(ratings)} ratings, {ratings.user_id.nunique()} users, {len(items)} items")
    print("SVD backend:", router.svd.backend)
    print("saved", config.MODELS / "router.joblib")


if __name__ == "__main__":
    main()
