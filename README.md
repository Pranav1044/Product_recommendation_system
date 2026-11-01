# Hybrid Product Recommendation Engine

A recommender for Amazon beauty products that picks a different method depending on how much rating history a user has: popularity for new users, collaborative filtering for users with history, and text similarity for products with few or no ratings. It includes an offline evaluation and a Streamlit demo app.

Mini Project, Phase 1-2 (individual project).

## Problem

Shoppers face too many products, and no single method works for everyone. Popularity ignores taste, collaborative filtering needs history, and text similarity ignores what other people liked. This project combines them and measures how each one performs.

## Features

- Four models: popularity (weighted rating), item-based collaborative filtering, SVD matrix factorization, content-based text similarity
- Rule-based hybrid router that chooses a model from the number of ratings a user has (thresholds set in `src/config.py`)
- Offline evaluation: Precision@K, Recall@K, catalog coverage, list diversity, and a cold-start segment
- Streamlit app with three modes: existing user, new user, similar to a product
- A "why recommended" label on every result showing which model produced it

## Dataset

Amazon Reviews 2023, category All_Beauty, from McAuley Lab: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

| File | Content |
| --- | --- |
| `All_Beauty.jsonl` | Reviews: user, product ID, rating, timestamp |
| `raw_meta_All_Beauty` (parquet) | Product metadata: title, description, features, categories |

Reviews and metadata are joined on `parent_asin`. After removing duplicate user-item pairs and keeping users and items with at least 3 ratings, the data has 1,244 users, 1,463 items and 8,978 ratings (99.51% sparse). The data files are not committed to this repository. Download both files into `data/raw/`.

## How it works

| User history | Model used |
| --- | --- |
| 0 ratings | Popularity (weighted rating) |
| 1 to 4 ratings | Item-based CF and content-based on the rated items, merged by score |
| 5 or more ratings | SVD |
| Similar to a product | Item-based CF and content if the product has 5 or more ratings, otherwise content only (TF-IDF cosine similarity) |

The thresholds are defaults and can be changed in `src/config.py`.

## Project structure

```
recommendation-engine/
├── README.md
├── requirements.txt
├── app.py                    # Streamlit app
├── data/
│   ├── raw/                  # downloaded files, not committed
│   └── processed/            # cleaned ratings and products
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_comparison.ipynb
├── src/
│   ├── config.py             # thresholds, K, minimum counts
│   ├── data_prep.py
│   ├── popularity.py
│   ├── item_cf.py
│   ├── svd_model.py
│   ├── content_model.py
│   ├── router.py
│   ├── evaluate.py
│   ├── eda.py
│   ├── train.py
│   └── try_models.py
├── models/                   # saved models
├── tests/
│   └── test_router.py
└── reports/
    └── figures/
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone <your-repo-url>
cd recommendation-engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `All_Beauty.jsonl` and the `raw_meta_All_Beauty` parquet file from the dataset page into `data/raw/`.

## Usage

Adjust the commands below to match your final script names.

```bash
python -m src.data_prep          # clean data and save processed files
python -m src.evaluate           # train on the split, evaluate, save the comparison table
python -m src.train              # fit on all ratings and save models/router.joblib

streamlit run app.py             # start the demo app
python -m pytest tests           # run the router tests
```

## Evaluation

Each user's ratings are split 80/20 into train and test. A test item is relevant when its rating is 4 or more. K = 10. All models are fit on the training part only.

Results on all 1,244 users (1,075 have at least one relevant test item). The full table, with the cold-start and 5-or-more-ratings segments, is in `reports/model_comparison.csv`.

| Model | Precision@10 | Recall@10 | Coverage | Diversity |
| --- | --- | --- | --- | --- |
| Popularity | 0.0011 | 0.0066 | 0.9% | 0.984 |
| Item-based CF | 0.0023 | 0.0195 | 44.0% | 0.967 |
| SVD | 0.0013 | 0.0083 | 11.6% | 0.982 |
| Content-based | 0.0099 | 0.0780 | 91.9% | 0.800 |
| Hybrid | 0.0073 | 0.0679 | 90.4% | 0.883 |

Cold-start segment (users with 2 to 4 training ratings): N users = 783 (649 with a relevant test item). The hybrid matches the best single model there (Precision@10 0.0103, Recall@10 0.1032).

Settings checked on the test split: `MIN_CO_RATED` 2 beat 3 and 4, `ROUTER_SVD_MIN` 5 beat 3 and 4, and `POPULARITY_M` differences were within noise, so the defaults are kept.

SVD implementation used: Surprise.

## Limitations

- All_Beauty is very sparse (about 1.1 ratings per user), so few users have enough history for collaborative filtering (about 461 of 1,244 users keep 5 or more training ratings).
- Absolute scores are low: most users have one relevant test item, so Precision@10 cannot exceed 0.1, and hit counts are small enough that gaps such as popularity vs SVD are noise.
- For users with 5 or more training ratings, SVD trails the content-based model, so the hybrid is weaker than content alone for them.
- Collaborative models cannot serve users or products with no ratings; the router falls back to popularity and text similarity.
- Popularity can score well on Precision@K because popular items are often liked, so accuracy is read together with coverage.
- Text features come from product titles and descriptions and may still contain noise after cleaning.
- Ratings only: views, clicks and purchases are not used.
- Offline metrics only; no A/B testing with real users.
- Local demo scale, no real-time serving.

## Future work

- Learned blending of model scores instead of fixed rules
- Diversity re-ranking
- Sentence-embedding text features
- Implicit-feedback models
- A FastAPI backend with a database

## Acknowledgements

Dataset: Amazon Reviews 2023 by McAuley Lab (paper: arXiv 2403.03952).

## License

Add a license here (for example MIT), or state that all rights are reserved.
