"""Content-based model: TF-IDF text similarity between products."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_prep import clean_text


class ContentModel:
    def fit(self, items):
        """items: DataFrame with parent_asin and the cleaned text column."""
        self.ids = list(items.parent_asin)
        self.pos = {a: k for k, a in enumerate(self.ids)}
        self.vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, ngram_range=(1, 2))
        self.X = self.vec.fit_transform(items.text)   # rows are L2-normalised, so dot = cosine
        return self

    def similar_items(self, item_id, n=10):
        if item_id not in self.pos:
            return []
        i = self.pos[item_id]
        s = (self.X @ self.X[i].T).toarray().ravel()
        s[i] = -1
        top = np.argsort(-s)[:n]
        return [(self.ids[j], float(s[j])) for j in top if s[j] > 0]

    def search(self, query, n=10):
        """Free-text search, for visitors with no history."""
        q = self.vec.transform([clean_text(query)])
        s = (self.X @ q.T).toarray().ravel()
        top = np.argsort(-s)[:n]
        return [(self.ids[j], float(s[j])) for j in top if s[j] > 0]

    def recommend_from(self, rated, n=10):
        """Items similar to a {item_id: rating} dict, weighted by the ratings."""
        known = [(self.pos[i], r) for i, r in rated.items() if i in self.pos]
        if not known:
            return []
        rows = [j for j, _ in known]
        w = np.array([r for _, r in known], dtype=float)
        profile = np.asarray(self.X[rows].T @ w).ravel()
        scores = np.asarray(self.X @ profile).ravel()
        scores[rows] = -np.inf
        top = np.argsort(-scores)[:n]
        return [(self.ids[j], float(scores[j])) for j in top if scores[j] > 0]
