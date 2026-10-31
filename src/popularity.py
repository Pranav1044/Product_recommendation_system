"""Popularity model: weighted rating, the same list for everyone."""
from src import config


class PopularityModel:
    def __init__(self, m=config.POPULARITY_M):
        self.m = m

    def fit(self, ratings):
        stats = ratings.groupby("parent_asin").rating.agg(["mean", "count"])
        overall = ratings.rating.mean()
        v, r = stats["count"], stats["mean"]
        # (v / (v + m)) * R + (m / (v + m)) * C
        stats["score"] = v / (v + self.m) * r + self.m / (v + self.m) * overall
        self.table = stats.sort_values("score", ascending=False)
        self.seen = ratings.groupby("user_id").parent_asin.agg(set).to_dict()
        return self

    def recommend(self, user_id=None, n=10):
        """Top-n items by weighted rating, skipping items the user already rated."""
        seen = self.seen.get(user_id, set())
        top = self.table[~self.table.index.isin(seen)].head(n)
        return list(zip(top.index, top.score))
