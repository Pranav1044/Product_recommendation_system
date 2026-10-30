import pandas as pd

# Load reviews (only the columns you need) and metadata
reviews = pd.read_json("data/raw/All_Beauty.jsonl", lines=True)
reviews = reviews[["user_id", "parent_asin", "rating", "timestamp"]]
meta = pd.read_parquet("data/raw/meta_All_Beauty.parquet")

print("reviews:", reviews.shape, "| meta:", meta.shape)
print("meta columns:", list(meta.columns))
print("users:", reviews.user_id.nunique(), "| items in reviews:", reviews.parent_asin.nunique())
print("duplicate parent_asin in meta:", meta.parent_asin.duplicated().sum())

# 1. The join: share of reviews and items that find a metadata row
in_meta = reviews.parent_asin.isin(set(meta.parent_asin))
print("reviews with a metadata row:", round(in_meta.mean() * 100, 1), "%")
items = reviews.parent_asin.drop_duplicates()
print("items with a metadata row:", round(items.isin(set(meta.parent_asin)).mean() * 100, 1), "%")

# 2. Is there usable text? (description and features may be lists)
def has_text(x):
    return x is not None and len(x) > 0

m = meta.drop_duplicates("parent_asin")
print("meta rows with title:", round(m.title.notna().mean() * 100, 1), "%")
print("meta rows with description:", round(m.description.apply(has_text).mean() * 100, 1), "%")
print("meta rows with features:", round(m.features.apply(has_text).mean() * 100, 1), "%")

# 3. Sparsity: ratings per user, and what survives the filters
print(reviews.groupby("user_id").size().describe())

def kcore(df, k):
    while True:
        u = df.user_id.map(df.user_id.value_counts())
        i = df.parent_asin.map(df.parent_asin.value_counts())
        keep = (u >= k) & (i >= k)
        if keep.all():
            return df
        df = df[keep]

matched = reviews[in_meta]
for k in (3, 5):
    core = kcore(matched, k)
    print(f"k={k}: ratings={len(core)}, users={core.user_id.nunique()}, items={core.parent_asin.nunique()}")