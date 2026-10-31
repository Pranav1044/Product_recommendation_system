"""Quick check that all four models run and give sensible results.

Run from the project root:  python -m src.try_models
"""
import pandas as pd

from src import config
from src.content_model import ContentModel
from src.item_cf import ItemCF
from src.popularity import PopularityModel
from src.svd_model import SVDModel

ratings = pd.read_parquet(config.PROCESSED / "ratings.parquet")
items = pd.read_parquet(config.PROCESSED / "items.parquet")
titles = items.set_index("parent_asin").title


def show(name, recs):
    print(f"\n{name}")
    for item, score in recs:
        print(f"  {score:7.3f}  {titles.get(item, '')[:70]}")
    if not recs:
        print("  (no results)")


counts = ratings.groupby("user_id").size()
user = counts[counts >= 5].index[0]
print("sample user:", user, "| ratings:", counts[user])
print("their rated items:")
for item, rating in ratings[ratings.user_id == user][["parent_asin", "rating"]].itertuples(index=False):
    print(f"  {rating}  {titles.get(item, '')[:70]}")

pop = PopularityModel().fit(ratings)
icf = ItemCF().fit(ratings)
svd = SVDModel().fit(ratings)
content = ContentModel().fit(items)
print("\nSVD backend:", svd.backend)

show("Popularity (new visitor)", pop.recommend(None))
show("Item-based CF for the user", icf.recommend(user))
show("SVD for the user", svd.recommend(user))
rated = icf.history(user)
show("Content-based from the user's rated items", content.recommend_from(rated))
item = next(iter(rated))
print("\nSeed item:", titles.get(item, "")[:70])
show("Similar items (item-based CF)", icf.similar_items(item))
show("Similar items (content)", content.similar_items(item))
show("Search: hair oil", content.search("hair oil"))
