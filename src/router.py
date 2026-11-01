"""Hybrid router: picks a model from how much history a user has.

Rules (thresholds live in src/config.py):
    0 ratings                      -> popularity
    1 up to ROUTER_SVD_MIN - 1     -> item-based CF + content-based, merged by score
    ROUTER_SVD_MIN or more         -> SVD (only for users the SVD model was trained on)
    similar-product lookup         -> item CF + content if the item has ROUTER_ITEM_MIN or more
                                      ratings, otherwise content only
    free-text search               -> content-based, popularity if nothing matches

Every result is a dict: item_id, score, model, reason ("why recommended").
"""
from collections import defaultdict

import joblib

from src import config
from src.content_model import ContentModel
from src.item_cf import ItemCF
from src.popularity import PopularityModel
from src.svd_model import SVDModel

REASONS = {
    "popularity": "Popular with other shoppers",
    "item_cf": "Liked by shoppers who rated similar items",
    "content": "Similar to items you rated",
    "svd": "Picked for you from your rating history",
}


def _pack(recs, model, reason):
    return [{"item_id": i, "score": s, "model": model, "reason": reason} for i, s in recs]


def _merge(lists, n, reason=None):
    """Scale each list to 0-1, add the scores of items found by several models, keep the top n."""
    total = defaultdict(float)
    used = defaultdict(list)
    for name, recs in lists.items():
        if not recs:
            continue
        top = max(s for _, s in recs)
        for item, s in recs:
            total[item] += s / top if top > 0 else 0.0
            used[item].append(name)
    ranked = sorted(total, key=total.get, reverse=True)[:n]
    return [{"item_id": i, "score": total[i], "model": "+".join(used[i]),
             "reason": reason or REASONS[used[i][0]]} for i in ranked]


class HybridRouter:
    def __init__(self, svd_min=config.ROUTER_SVD_MIN, few_min=config.ROUTER_FEW_MIN,
                 item_min=config.ROUTER_ITEM_MIN):
        self.svd_min = svd_min
        self.few_min = few_min
        self.item_min = item_min

    def fit(self, ratings, items):
        self.popularity = PopularityModel().fit(ratings)
        self.item_cf = ItemCF().fit(ratings)
        self.svd = SVDModel().fit(ratings)
        self.content = ContentModel().fit(items)
        self.item_counts = ratings.groupby("parent_asin").size().to_dict()
        return self

    def choose(self, n_ratings, in_training=True):
        """Name of the model used for a user with n_ratings training ratings."""
        if n_ratings < self.few_min:
            return "popularity"
        if n_ratings >= self.svd_min and in_training:
            return "svd"
        return "item_cf+content"

    def recommend_for_user(self, user_id, n=10):
        """Existing user: history comes from the training ratings."""
        return self.recommend_from_ratings(self.item_cf.history(user_id), n, user_id=user_id)

    def recommend_from_ratings(self, rated, n=10, user_id=None):
        """rated is {item_id: rating}. Also used for new users who rate a few items."""
        in_training = user_id is not None and user_id in self.svd.seen
        choice = self.choose(len(rated), in_training)
        if choice == "svd":
            recs = self.svd.recommend(user_id, n)
            if recs:
                return _pack(recs, "svd", REASONS["svd"])
        if choice != "popularity":
            merged = _merge({"item_cf": self.item_cf.recommend_from(rated, n),
                             "content": self.content.recommend_from(rated, n)}, n)
            if merged:
                return merged
        return _pack(self.popularity.recommend(user_id, n), "popularity", REASONS["popularity"])

    def search(self, query, n=10):
        recs = self.content.search(query, n)
        if recs:
            return _pack(recs, "content", "Matches your search")
        return _pack(self.popularity.recommend(None, n), "popularity", REASONS["popularity"])

    def similar_to(self, item_id, n=10):
        if self.item_counts.get(item_id, 0) >= self.item_min:
            return _merge({"item_cf": self.item_cf.similar_items(item_id, n),
                           "content": self.content.similar_items(item_id, n)},
                          n, reason="Similar to this product")
        return _pack(self.content.similar_items(item_id, n), "content", "Similar to this product")

    def save(self, path=None):
        config.MODELS.mkdir(exist_ok=True)
        joblib.dump(self, path or config.MODELS / "router.joblib")

    @staticmethod
    def load(path=None):
        return joblib.load(path or config.MODELS / "router.joblib")
