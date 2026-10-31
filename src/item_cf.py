"""Item-based collaborative filtering (adjusted cosine similarity)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src import config


class ItemCF:
    def __init__(self, min_co_rated=config.MIN_CO_RATED):
        self.min_co_rated = min_co_rated

    def fit(self, ratings):
        u_idx, self.user_ids = pd.factorize(ratings.user_id)
        i_idx, self.item_ids = pd.factorize(ratings.parent_asin)
        self.user_pos = {u: k for k, u in enumerate(self.user_ids)}
        self.item_pos = {i: k for k, i in enumerate(self.item_ids)}
        r = ratings.rating.to_numpy(dtype=float)
        shape = (len(self.user_ids), len(self.item_ids))

        # Mean-center each user's ratings, then compare items by cosine similarity
        user_mean = np.bincount(u_idx, weights=r) / np.bincount(u_idx)
        centered = csr_matrix((r - user_mean[u_idx], (u_idx, i_idx)), shape=shape)
        rated = csr_matrix((np.ones_like(r), (u_idx, i_idx)), shape=shape)

        co_rated = (rated.T @ rated).toarray()        # users who rated both items
        items = normalize(centered.T.tocsr())
        sim = (items @ items.T).toarray()
        sim[co_rated < self.min_co_rated] = 0         # ignore pairs with too little overlap
        sim = np.maximum(sim, 0)                      # keep positive similarity only
        np.fill_diagonal(sim, 0)
        self.sim = sim
        self.ratings_matrix = csr_matrix((r, (u_idx, i_idx)), shape=shape)
        return self

    def history(self, user_id):
        """The user's training ratings as {item_id: rating}."""
        if user_id not in self.user_pos:
            return {}
        row = self.ratings_matrix[self.user_pos[user_id]]
        return {self.item_ids[j]: v for j, v in zip(row.indices, row.data)}

    def similar_items(self, item_id, n=10):
        if item_id not in self.item_pos:
            return []
        s = self.sim[self.item_pos[item_id]]
        top = np.argsort(-s)[:n]
        return [(self.item_ids[j], float(s[j])) for j in top if s[j] > 0]

    def recommend_from(self, rated, n=10):
        """Top-n items from a {item_id: rating} dict (works for users with few ratings)."""
        known = [(self.item_pos[i], r) for i, r in rated.items() if i in self.item_pos]
        if not known:
            return []
        v = np.zeros(len(self.item_ids))
        for j, r in known:
            v[j] = r
        scores = self.sim @ v                          # similarity-weighted sum of ratings
        scores[[j for j, _ in known]] = -np.inf        # never recommend already rated items
        top = np.argsort(-scores)[:n]
        return [(self.item_ids[j], float(scores[j])) for j in top if scores[j] > 0]

    def recommend(self, user_id, n=10):
        return self.recommend_from(self.history(user_id), n)
