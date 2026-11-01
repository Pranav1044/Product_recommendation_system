"""Step 6: evaluation. Fits every model on the training split and compares them.

Run from the project root:  python -m src.evaluate
Quick test on the first 200 users:  python -m src.evaluate 200

Split: for each user, the latest ratings (about 20%, at least 1) are held out as test data.
Relevant = a test rating of 4 or more. All models are fit on the training part only.
Precision@K and Recall@K are averaged over users with at least one relevant test item.
"""
import copy
import sys

import numpy as np
import pandas as pd

from src import config
from src.item_cf import ItemCF
from src.popularity import PopularityModel
from src.router import HybridRouter

COLD_MAX_TRAIN = 4       # cold-start segment = users with 2 to 4 training ratings


def split_by_time(ratings, fraction):
    """Hold out each user's latest ratings (at least 1) as test data."""
    ratings = ratings.sort_values("timestamp", kind="stable")
    newest_first = ratings.groupby("user_id").cumcount(ascending=False)   # 0 = newest
    n = ratings.groupby("user_id").rating.transform("size")
    n_test = (n * fraction).round().clip(lower=1)
    is_test = newest_first < n_test
    return ratings[~is_test], ratings[is_test]


def cache_recommend(model):
    """Remember each user's recommendations so SVD is only computed once per user."""
    store = {}
    original = model.recommend

    def cached(user_id, n=10):
        key = (user_id, n)
        if key not in store:
            store[key] = original(user_id, n)
        return store[key]

    model.recommend = cached


def list_diversity(items, content):
    """Average (1 - cosine similarity) between the text vectors of the items in one list."""
    idx = [content.pos[i] for i in items if i in content.pos]
    if len(idx) < 2:
        return None
    sim = (content.X[idx] @ content.X[idx].T).toarray()
    n = len(idx)
    return 1 - (sim.sum() - np.trace(sim)) / (n * (n - 1))


def score(fn, users, relevant, k, content, catalog_size):
    """fn(user_id) -> list of item ids. Returns the metrics for these users."""
    precision, recall, lists, diversity = [], [], [], []
    for u in users:
        recs = list(fn(u))[:k]
        lists.append(recs)
        d = list_diversity(recs, content)
        if d is not None:
            diversity.append(d)
        rel = relevant.get(u)
        if rel:
            hits = len(set(recs) & rel)
            precision.append(hits / k)
            recall.append(hits / len(rel))
    covered = set().union(*lists) if lists else set()
    return {
        "users_evaluated": len(precision),
        "precision": np.mean(precision) if precision else np.nan,
        "recall": np.mean(recall) if recall else np.nan,
        "coverage": len(covered) / catalog_size,
        "diversity": np.mean(diversity) if diversity else np.nan,
    }


def show(title, df):
    print(f"\n{title}")
    print(df.round(4).to_string(index=False))


def main():
    k = config.TOP_K
    max_users = int(sys.argv[1]) if len(sys.argv) > 1 else None

    ratings = pd.read_parquet(config.PROCESSED / "ratings.parquet")
    items = pd.read_parquet(config.PROCESSED / "items.parquet")
    catalog = len(items)

    train, test = split_by_time(ratings, config.TEST_FRACTION)
    n_train = train.groupby("user_id").size()
    relevant = (test[test.rating >= config.RELEVANT_RATING]
                .groupby("user_id").parent_asin.agg(set).to_dict())
    users = list(n_train.index)
    if max_users:
        users = users[:max_users]
    print(f"train ratings: {len(train)} | test ratings: {len(test)} | users: {len(users)}")
    print(f"users with at least one relevant test item: {sum(u in relevant for u in users)}")

    print("fitting models on the training split ...")
    router = HybridRouter().fit(train, items)
    cache_recommend(router.svd)
    print("SVD backend:", router.svd.backend)
    history = {u: router.item_cf.history(u) for u in users}

    def ids(recs):
        return [i for i, _ in recs]

    models = {
        "Popularity": lambda u: ids(router.popularity.recommend(u, k)),
        "Item-based CF": lambda u: ids(router.item_cf.recommend(u, k)),
        "SVD": lambda u: ids(router.svd.recommend(u, k)),
        "Content-based": lambda u: ids(router.content.recommend_from(history[u], k)),
        "Hybrid": lambda u: [r["item_id"] for r in router.recommend_for_user(u, k)],
    }
    segments = {
        "All users": users,
        f"Cold-start segment (2 to {COLD_MAX_TRAIN} training ratings)":
            [u for u in users if n_train[u] <= COLD_MAX_TRAIN],
        f"{COLD_MAX_TRAIN + 1} or more training ratings":
            [u for u in users if n_train[u] > COLD_MAX_TRAIN],
    }

    # 1. Model comparison
    rows = []
    for seg, seg_users in segments.items():
        for name, fn in models.items():
            m = score(fn, seg_users, relevant, k, router.content, catalog)
            rows.append({"segment": seg, "model": name, "users": len(seg_users), **m})
    table = pd.DataFrame(rows)
    for seg, seg_users in segments.items():
        part = table[table.segment == seg].drop(columns=["segment", "users"])
        show(f"{seg}: {len(seg_users)} users, K = {k}", part)
    config.ROOT.joinpath("reports").mkdir(exist_ok=True)
    table.to_csv(config.ROOT / "reports" / "model_comparison.csv", index=False)

    # 2. Tuning checks (nothing is changed automatically; edit src/config.py yourself)
    mid = [u for u in users if 3 <= n_train[u] <= COLD_MAX_TRAIN]
    out = []
    for t in (3, 4, 5):
        r = copy.copy(router)            # shares the fitted models
        r.svd_min = t
        fn = lambda u: [x["item_id"] for x in r.recommend_for_user(u, k)]
        all_ = score(fn, users, relevant, k, router.content, catalog)
        mid_ = score(fn, mid, relevant, k, router.content, catalog)
        out.append({"ROUTER_SVD_MIN": t, "P@K all": all_["precision"], "R@K all": all_["recall"],
                    "P@K 3-4 ratings": mid_["precision"], "R@K 3-4 ratings": mid_["recall"]})
    show("Router threshold (hybrid)", pd.DataFrame(out))

    out = []
    for m in (2, 3, 4):
        icf = ItemCF(min_co_rated=m).fit(train)
        s = score(lambda u: ids(icf.recommend(u, k)), users, relevant, k, router.content, catalog)
        out.append({"MIN_CO_RATED": m, "P@K": s["precision"], "R@K": s["recall"], "coverage": s["coverage"]})
    show("Item-based CF: minimum co-rated users", pd.DataFrame(out))

    out = []
    for m in (5, 10, 20):
        pop = PopularityModel(m=m).fit(train)
        s = score(lambda u: ids(pop.recommend(u, k)), users, relevant, k, router.content, catalog)
        out.append({"POPULARITY_M": m, "P@K": s["precision"], "R@K": s["recall"], "coverage": s["coverage"]})
    show("Popularity: m", pd.DataFrame(out))

    print("\nsaved reports/model_comparison.csv")


if __name__ == "__main__":
    main()
