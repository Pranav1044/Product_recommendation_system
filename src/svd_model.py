"""SVD matrix factorization. Uses Surprise if it is installed, else scikit-learn TruncatedSVD.

Check `model.backend` and state which one you used in the report.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src import config

try:
    from surprise import SVD, Dataset, Reader
    HAVE_SURPRISE = True
except ImportError:
    HAVE_SURPRISE = False


class SVDModel:
    def __init__(self, n_factors=config.SVD_FACTORS):
        self.n_factors = n_factors
        self.backend = "surprise" if HAVE_SURPRISE else "truncated_svd"

    def fit(self, ratings):
        self.seen = ratings.groupby("user_id").parent_asin.agg(set).to_dict()
        if HAVE_SURPRISE:
            self.item_list = list(ratings.parent_asin.unique())
            data = Dataset.load_from_df(
                ratings[["user_id", "parent_asin", "rating"]], Reader(rating_scale=(1, 5)))
            self.algo = SVD(n_factors=self.n_factors, random_state=config.RANDOM_SEED)
            self.algo.fit(data.build_full_trainset())
        else:
            u_idx, user_ids = pd.factorize(ratings.user_id)
            i_idx, item_ids = pd.factorize(ratings.parent_asin)
            self.user_pos = {u: k for k, u in enumerate(user_ids)}
            self.item_list = list(item_ids)
            r = ratings.rating.to_numpy(dtype=float)
            mean = np.bincount(u_idx, weights=r) / np.bincount(u_idx)
            # Missing ratings count as 0 = "average for this user" after centering
            X = csr_matrix((r - mean[u_idx], (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))
            self.svd = TruncatedSVD(
                n_components=min(self.n_factors, X.shape[1] - 1), random_state=config.RANDOM_SEED)
            self.user_factors = self.svd.fit_transform(X)
        return self

    def recommend(self, user_id, n=10):
        """Top-n unseen items by predicted score. Returns [] for users not in training."""
        if user_id not in self.seen:
            return []
        if HAVE_SURPRISE:
            scores = np.array([self.algo.predict(user_id, i).est for i in self.item_list])
        else:
            scores = self.user_factors[self.user_pos[user_id]] @ self.svd.components_
        seen = self.seen[user_id]
        order = np.argsort(-scores)
        out = [(self.item_list[j], float(scores[j])) for j in order if self.item_list[j] not in seen]
        return out[:n]
