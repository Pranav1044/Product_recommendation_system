"""Step 3: clean the All_Beauty data and save processed files.

Run from the project root:  python -m src.data_prep
"""
import html
import re

import pandas as pd

from src import config

TAG = re.compile(r"<[^>]+>")
NON_WORD = re.compile(r"[^a-z0-9\s]")
SPACE = re.compile(r"\s+")


def to_text(x):
    """Turn a list, array, string or missing value into one plain string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return " ".join(str(i) for i in x)
    except TypeError:  # NaN and other non-iterables
        return ""


def clean_text(s):
    """Unescape HTML entities, drop tags, lowercase, keep letters and digits."""
    s = html.unescape(html.unescape(s))  # twice, in case of double-encoded entities
    s = TAG.sub(" ", s).lower()
    s = NON_WORD.sub(" ", s)
    return SPACE.sub(" ", s).strip()


def kcore(df, k):
    """Repeatedly drop users and items with fewer than k ratings until none are left."""
    while True:
        u = df.user_id.map(df.user_id.value_counts())
        i = df.parent_asin.map(df.parent_asin.value_counts())
        keep = (u >= k) & (i >= k)
        if keep.all():
            return df
        df = df[keep]


def main():
    log = []

    def note(msg):
        print(msg)
        log.append(msg)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Reviews: keep needed columns, drop missing values, keep the latest duplicate
    reviews = pd.read_json(config.REVIEWS_FILE, lines=True)
    reviews = reviews[["user_id", "parent_asin", "rating", "timestamp"]]
    note(f"raw reviews: {len(reviews)}")
    reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating"])
    reviews = reviews.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last")
    note(f"after dropping missing values and duplicates: {len(reviews)}")

    # 2. Metadata joined on parent_asin
    meta = pd.read_parquet(config.META_FILE)
    meta = meta[["parent_asin", "title", "main_category", "categories", "features", "description"]]
    meta = meta.drop_duplicates("parent_asin")
    matched = reviews.parent_asin.isin(set(meta.parent_asin))
    note(f"reviews with a metadata row: {matched.mean() * 100:.1f}% (dropping {(~matched).sum()})")
    reviews = reviews[matched]

    # 3. k-core filter
    core = kcore(reviews, config.MIN_RATINGS).reset_index(drop=True)
    n_users, n_items = core.user_id.nunique(), core.parent_asin.nunique()
    sparsity = 1 - len(core) / (n_users * n_items)
    note(f"k = {config.MIN_RATINGS}: ratings = {len(core)}, users = {n_users}, items = {n_items}")
    note(f"sparsity of the user-item matrix: {sparsity * 100:.2f}%")

    # 4. Item table with one cleaned text field
    items = meta[meta.parent_asin.isin(set(core.parent_asin))].copy()
    parts = {c: items[c].map(to_text) for c in ["title", "categories", "features", "description"]}
    items["text"] = (parts["title"] + " " + parts["categories"] + " "
                     + parts["features"] + " " + parts["description"]).map(clean_text)
    items["title"] = items.title.fillna("")
    items["main_category"] = items.main_category.fillna("Unknown")
    for c, s in parts.items():
        note(f"items with non-empty {c}: {(s.str.len() > 0).mean() * 100:.1f}%")
    note(f"average cleaned text length (words): {items.text.str.split().str.len().mean():.1f}")

    # 5. Save
    core.to_parquet(config.PROCESSED / "ratings.parquet", index=False)
    items[["parent_asin", "title", "main_category", "text"]].to_parquet(
        config.PROCESSED / "items.parquet", index=False)
    (config.ROOT / "reports").mkdir(exist_ok=True)
    (config.ROOT / "reports" / "data_prep_log.txt").write_text("\n".join(log), encoding="utf-8")
    note("saved data/processed/ratings.parquet and items.parquet")


if __name__ == "__main__":
    main()
