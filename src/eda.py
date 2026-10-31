"""Step 3: exploratory analysis. Saves figures to reports/figures and prints key numbers.

Run from the project root:  python -m src.eda
You can also paste each block into a notebook cell (notebooks/01_eda.ipynb).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

config.FIGURES.mkdir(parents=True, exist_ok=True)
ratings = pd.read_parquet(config.PROCESSED / "ratings.parquet")
items = pd.read_parquet(config.PROCESSED / "items.parquet")


def save(name):
    plt.tight_layout()
    plt.savefig(config.FIGURES / name, dpi=150)
    plt.close()


# 1. Rating distribution
counts = ratings.rating.value_counts().sort_index()
plt.bar(counts.index.astype(str), counts.values)
plt.title("Rating distribution")
plt.xlabel("Rating")
plt.ylabel("Number of ratings")
save("rating_distribution.png")
print("share of ratings >= 4:", round((ratings.rating >= config.RELEVANT_RATING).mean() * 100, 1), "%")

# 2. Ratings per user and per item
per_user = ratings.groupby("user_id").size()
per_item = ratings.groupby("parent_asin").size()

plt.hist(per_user, bins=50, log=True)
plt.title("Ratings per user")
plt.xlabel("Ratings")
plt.ylabel("Users (log scale)")
save("ratings_per_user.png")

plt.plot(np.sort(per_item.values)[::-1])
plt.yscale("log")
plt.title("Ratings per item, sorted (long tail)")
plt.xlabel("Item rank")
plt.ylabel("Ratings (log scale)")
save("ratings_per_item.png")

# 3. Most rated items
top = per_item.nlargest(10)
titles = items.set_index("parent_asin").title.reindex(top.index).fillna("").str.slice(0, 40)
plt.barh(titles.values[::-1], top.values[::-1])
plt.title("Ten most rated items")
plt.xlabel("Number of ratings")
save("top_items.png")

# 4. Sparsity and what the router will see
n_users, n_items = per_user.size, per_item.size
print("users:", n_users, "| items:", n_items, "| ratings:", len(ratings))
print("sparsity:", round((1 - len(ratings) / (n_users * n_items)) * 100, 2), "%")
print(per_user.describe())

# Approximate training ratings per user after the split (at least 1 held out per user)
test_n = np.maximum(1, np.round(per_user * config.TEST_FRACTION)).astype(int)
train_n = per_user - test_n
for t in (2, 3, 4, 5, 8, 10):
    print(f"users with at least {t} training ratings: {(train_n >= t).sum()}")

# 5. Text check: do top terms contain HTML leftovers or boilerplate?
vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5)
X = vec.fit_transform(items.text)
terms = np.array(vec.get_feature_names_out())
mean_score = np.asarray(X.mean(axis=0)).ravel()
print("top 30 terms:", list(terms[mean_score.argsort()[::-1][:30]]))
suspects = [t for t in ("nbsp", "amp", "br", "quot", "div", "span", "href", "http", "www") if t in set(terms)]
print("HTML or boilerplate terms found:", suspects)

