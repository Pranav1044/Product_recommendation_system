"""Central settings. Change numbers here, not inside the model files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

REVIEWS_FILE = RAW / "All_Beauty.jsonl"
META_FILE = RAW / "meta_All_Beauty.parquet"

# Data
MIN_RATINGS = 3          # keep users and items with at least this many ratings (k-core)
RANDOM_SEED = 66

# Evaluation
TOP_K = 10
RELEVANT_RATING = 4      # a test rating at or above this counts as relevant
TEST_FRACTION = 0.2      # share of each user's ratings held out (at least 1 per user)

# Models
MIN_CO_RATED = 2         # item-based CF: minimum users who rated both items
POPULARITY_M = 5         # weighted rating: pulls items with few ratings toward the global mean
SVD_FACTORS = 20         # number of latent factors for SVD

# Router: number of TRAINING ratings a user has
ROUTER_FEW_MIN = 1       # 1 up to ROUTER_SVD_MIN - 1  -> item-based CF + content
ROUTER_SVD_MIN = 5       # this many or more           -> SVD (revisit after EDA)
ROUTER_ITEM_MIN = 5      # an item with fewer training ratings uses content only
