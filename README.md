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

Reviews and metadata are joined on `parent_asin`. The data files are not committed to this repository. Download both files into `data/raw/`.

## How it works

| User history | Model used |
| --- | --- |
| 0 ratings | Popularity (weighted rating, optional category filter) |
| 1 to 4 ratings | Item-based CF and content-based on the rated items, merged by score |
| 5 or more ratings | SVD |
| New or rarely rated product | Content-based (TF-IDF cosine similarity) |

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
│   └── evaluate.py
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
python -m src.evaluate           # train models, evaluate, save the comparison table
streamlit run app.py             # start the demo app
python -m pytest tests           # run the router tests
```

## Evaluation

Each user's ratings are split 80/20 into train and test. A test item is relevant when its rating is 4 or more. K = 10. All models are fit on the training part only.

Fill this table in after running the evaluation.

| Model | Precision@10 | Recall@10 | Coverage | Diversity |
| --- | --- | --- | --- | --- |
| Popularity | | | | |
| Item-based CF | | | | |
| SVD | | | | |
| Content-based | | | | |
| Hybrid | | | | |

Cold-start segment (users with 1 to 4 training ratings): N users = ___

SVD implementation used: Surprise / scikit-learn TruncatedSVD (delete one).

## Limitations

- All_Beauty is very sparse (about 1.1 ratings per user), so few users have enough history for collaborative filtering.
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