from pathlib import Path

for d in ["data/raw", "data/processed", "notebooks", "src", "models", "tests", "reports/figures"]:
    Path(d).mkdir(parents=True, exist_ok=True)

for f in ["README.md", "requirements.txt", "app.py", "src/config.py", "src/data_prep.py",
          "src/popularity.py", "src/item_cf.py", "src/svd_model.py", "src/content_model.py",
          "src/router.py", "src/evaluate.py", "tests/test_router.py"]:
    Path(f).touch()