"""Router tests. Run from the project root:  python -m pytest tests -v"""
import random

import pandas as pd
import pytest

from src.router import HybridRouter

ITEM_TEXT = {
    "i0": "hair oil", "i1": "hair shampoo", "i2": "hair brush",
    "i3": "skin cream", "i4": "skin serum", "i5": "skin mask",
    "i6": "nail polish", "i7": "nail gel", "i8": "makeup brush", "i9": "makeup kit",
}


@pytest.fixture(scope="module")
def router():
    rng = random.Random(0)
    rows = []
    for u in range(8):                                    # u0..u7: 6 ratings each
        for item in rng.sample(list(ITEM_TEXT), 6):
            rows.append((f"u{u}", item, rng.randint(1, 5)))
    for item, rating in [("i0", 5), ("i1", 4), ("i3", 5)]:   # "few": 3 ratings
        rows.append(("few", item, rating))
    ratings = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating"])
    items = pd.DataFrame({"parent_asin": list(ITEM_TEXT),
                          "title": list(ITEM_TEXT.values()),
                          "text": list(ITEM_TEXT.values())})
    return HybridRouter().fit(ratings, items)


# Branch logic (no models needed)
def test_zero_ratings_use_popularity():
    assert HybridRouter().choose(0) == "popularity"


def test_few_ratings_use_item_cf_and_content():
    r = HybridRouter()
    for n in range(1, 5):
        assert r.choose(n) == "item_cf+content"


def test_many_ratings_use_svd():
    r = HybridRouter()
    assert r.choose(5, in_training=True) == "svd"
    assert r.choose(50, in_training=True) == "svd"


def test_new_user_cannot_use_svd():
    assert HybridRouter().choose(8, in_training=False) == "item_cf+content"


def test_thresholds_come_from_settings():
    assert HybridRouter(svd_min=3).choose(3, in_training=True) == "svd"
    assert HybridRouter(svd_min=7).choose(5, in_training=True) == "item_cf+content"


# End to end on a tiny dataset
def test_unknown_user_gets_popularity(router):
    recs = router.recommend_for_user("ghost", n=5)
    assert recs and all(r["model"] == "popularity" for r in recs)


def test_few_ratings_branch_returns_unseen_items(router):
    recs = router.recommend_for_user("few", n=5)
    assert recs
    assert all(r["model"] in {"item_cf", "content", "item_cf+content"} for r in recs)
    assert not {"i0", "i1", "i3"} & {r["item_id"] for r in recs}


def test_many_ratings_branch_uses_svd(router):
    recs = router.recommend_for_user("u0", n=3)
    assert recs and all(r["model"] == "svd" for r in recs)
    assert not set(router.item_cf.history("u0")) & {r["item_id"] for r in recs}


def test_search_matches_text(router):
    found = {r["item_id"] for r in router.search("hair", n=3)}
    assert found == {"i0", "i1", "i2"}


def test_similar_items_exclude_the_item_itself(router):
    assert all(r["item_id"] != "i0" for r in router.similar_to("i0", n=3))


def test_every_result_has_a_reason(router):
    for recs in (router.recommend_for_user("u0", 3), router.recommend_for_user("few", 3),
                 router.search("hair", 3), router.similar_to("i0", 3)):
        assert all(r["reason"] for r in recs)
