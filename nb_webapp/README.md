# Naive Bayes Expert System (Flask) — Ready-to-Train

End‑to‑end skeleton for your assignment:
1) Download dataset from the web (each student must use a DIFFERENT dataset).
2) Train a Naive Bayes model (`train.py`) → produces `model_pipeline.joblib` + `meta.json` + `static/cm.png`.
3) Run Flask app (`app.py`) to serve inputs + inference + confusion matrix accuracy.
4) Deploy to PythonAnywhere.

---

## 1) Choose a dataset (different for every student)

Use any **classification** dataset with a categorical target. Recommended sources:
- UCI Machine Learning Repository
- Kaggle (use a direct `raw` CSV URL)
- OpenML

**Rules to stay unique:**
- Pick a different dataset (or at least a different *target* column) than your classmates.
- If the dataset has numeric columns, we discretize them automatically during training.
- Make sure the dataset has **no personally identifiable** data.

Set the dataset URL & target column at the top of `train.py`:
```python
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
TARGET = "class"
```

> Example: For the UCI Car Evaluation dataset you must supply column names:
```python
COLUMN_NAMES = ["buying","maint","doors","persons","lug_boot","safety","class"]
```

---

## 2) Train locally or on PythonAnywhere

### A) On your computer (recommended first)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# edit train.py → set DATA_URL, TARGET, optional COLUMN_NAMES
python train.py
# This writes:
# - model_pipeline.joblib
# - meta.json (feature names, choices, metrics)
# - static/cm.png (confusion matrix image)
```

### B) On PythonAnywhere
- Open a **Bash console**.
- `git clone` your repo **or** upload ZIP then `unzip` it (Files tab has a 100MB limit; if needed, use GitHub or `wget` to pull large files).
- Create a virtualenv and install:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

---

## 3) Run the web app locally
```bash
flask --app app.py run
# or: python app.py
```

Open http://127.0.0.1:5000

---

## 4) Deploy to PythonAnywhere

1. **Files**: Put project in `~/mysite/` or a folder you like.
2. **Virtualenv**: Create & install `requirements.txt` inside it.
3. **Web** tab:
   - Add a new Flask web app (Manual config).
   - **Virtualenv path**: `/home/<your_user>/<your_project>/.venv`
   - **WSGI file**: point it to `wsgi.py` (edit path inside if needed).
4. **Reload** the web app.

> If you see import errors:
- Make sure your virtualenv is selected on the Web tab.
- `pip install -r requirements.txt` inside that virtualenv.
- Ensure `model_pipeline.joblib` and `meta.json` exist (run `python train.py`).

---

## 5) Make your dataset unique

You can further ensure uniqueness by:
- Selecting **different target** column if dataset has many.
- Filtering rows/columns differently (but keep enough samples).
- Using different **binning strategy** (change `N_BINS` in `train.py`).

Happy building!
