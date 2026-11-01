"""Streamlit demo app. Loads the saved router only, no retraining.

First run:  python -m src.train
Then:       streamlit run app.py
"""
import time

import pandas as pd
import streamlit as st

from src import config
from src.router import HybridRouter

MIN_SEARCH_SCORE = 0.15   # hide weak free-text matches
N_RESULTS = config.TOP_K
N_ONBOARDING = 6          # products shown to a new user to rate

st.set_page_config(page_title="Product Recommendation Engine", layout="centered")


@st.cache_resource
def load():
    router = HybridRouter.load()
    items = pd.read_parquet(config.PROCESSED / "items.parquet")
    titles = items.set_index("parent_asin").title.to_dict()
    return router, titles


try:
    router, titles = load()
except FileNotFoundError:
    st.error("Saved model not found. Run `python -m src.train` first.")
    st.stop()


def title_of(item_id):
    return titles.get(item_id) or item_id


def show(results, took):
    if not results:
        st.info("No results. Try different input.")
        return
    st.caption(f"{len(results)} results in {took:.2f} s")
    for rank, r in enumerate(results, 1):
        st.markdown(f"**{rank}. {title_of(r['item_id'])}**")
        st.caption(f"Why: {r['reason']}  |  model: {r['model']}")


st.title("Product Recommendation Engine")
st.caption(
    "Hybrid recommender for Amazon beauty products. The router picks popularity, item-based CF "
    "plus content, SVD or text similarity depending on how much history there is."
)

mode = st.sidebar.radio("Mode", ["Existing user", "New user", "Similar to a product"])
st.sidebar.caption(f"SVD backend: {router.svd.backend}")
st.sidebar.caption(
    f"Router: 0 ratings -> popularity, {router.few_min} to {router.svd_min - 1} -> item CF + content, "
    f"{router.svd_min}+ -> SVD"
)

# 1. Existing user
if mode == "Existing user":
    users = list(router.item_cf.user_ids)
    counts = {u: len(router.item_cf.history(u)) for u in users}
    users.sort(key=lambda u: -counts[u])
    user = st.selectbox(
        "User ID (most active first)", users,
        format_func=lambda u: f"{u[:14]}...  ({counts[u]} ratings)")
    history = router.item_cf.history(user)
    with st.expander(f"Items this user rated ({len(history)})"):
        for item, rating in history.items():
            st.write(f"{int(rating)} stars: {title_of(item)}")
    st.write(f"Model chosen: **{router.choose(len(history))}**")
    if st.button("Get recommendations"):
        start = time.perf_counter()
        results = router.recommend_for_user(user, N_RESULTS)
        show(results, time.perf_counter() - start)

# 2. New user: search term and/or a few ratings
elif mode == "New user":
    st.write("Type a search term, rate a few products, or do neither to see popular products.")
    with st.form("new_user"):
        query = st.text_input("Search term (for example: hair oil)")
        st.write("Rate products you know (optional, 3 to 5 gives the best results):")
        onboarding = list(router.popularity.table.index[:N_ONBOARDING])
        ratings = {}
        for item in onboarding:
            ratings[item] = st.selectbox(
                title_of(item)[:90], ["Not rated", 1, 2, 3, 4, 5], key=f"rate_{item}")
        submitted = st.form_submit_button("Get recommendations")
    if submitted:
        rated = {i: r for i, r in ratings.items() if r != "Not rated"}
        start = time.perf_counter()
        if rated:
            results = router.recommend_from_ratings(rated, N_RESULTS)
        elif query.strip():
            results = router.search(query, N_RESULTS)
            results = [r for r in results if r["model"] != "content" or r["score"] >= MIN_SEARCH_SCORE]
        else:
            results = router.recommend_from_ratings({}, N_RESULTS)
        show(results, time.perf_counter() - start)
        if rated and query.strip():
            st.caption("Your ratings were used; the search term was ignored.")

# 3. Similar to a product
else:
    item_ids = sorted(router.content.ids, key=lambda i: title_of(i).lower())
    item = st.selectbox("Product", item_ids, format_func=lambda i: title_of(i)[:100])
    n_ratings = router.item_counts.get(item, 0)
    st.write(f"This product has **{n_ratings}** ratings in the data.")
    if st.button("Find similar products"):
        start = time.perf_counter()
        results = router.similar_to(item, N_RESULTS)
        show(results, time.perf_counter() - start)
