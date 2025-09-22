# === train.py (Car Evaluation → nilai & label Bahasa Indonesia, tanpa gambar CM) ===
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

# ================== CONFIG ==================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
COLUMN_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
TARGET = "class"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BINS = 5
APPLY_TRANSLATION = True
# ============================================

# ====== Terjemahan (khusus Car Evaluation) ======
TRANSLATE_FEATURES = {
    "buying":   "Harga Beli",
    "maint":    "Biaya Perawatan",
    "doors":    "Jumlah Pintu",
    "persons":  "Kapasitas Orang",
    "lug_boot": "Ukuran Bagasi",
    "safety":   "Tingkat Keamanan",
}

TRANSLATE_VALUES_BY_COL = {
    "buying":  {"vhigh": "Sangat Tinggi", "high": "Tinggi", "med": "Sedang", "low": "Rendah"},
    "maint":   {"vhigh": "Sangat Tinggi", "high": "Tinggi", "med": "Sedang", "low": "Rendah"},
    "doors":   {"2": "2", "3": "3", "4": "4", "5more": "5 atau lebih"},
    "persons": {"2": "2", "4": "4", "more": "Lebih Banyak"},
    "lug_boot":{"small": "Kecil", "med": "Sedang", "big": "Besar"},
    "safety":  {"vhigh": "Sangat Tinggi", "high": "Tinggi", "med": "Sedang"},
}

TRANSLATE_TARGET = {
    "unacc": "Tidak Layak",
    "acc":   "Layak",
    "good":  "Bagus",
    "vgood": "Sangat Bagus",
}
# ================================================

BASE = Path(__file__).parent

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, header=None if COLUMN_NAMES else "infer")
    if COLUMN_NAMES:
        df.columns = COLUMN_NAMES
    return df

# ---------- Normalisasi doors/persons dari numerik → kategori ----------
def to_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s in ("", "nan", "none"):
        return np.nan
    try:
        return float(''.join(ch for ch in s if (ch.isdigit() or ch == '.'))) if any(ch.isdigit() for ch in s) else np.nan
    except Exception:
        return np.nan

def bucket_doors(val):
    if np.isnan(val): return np.nan
    v = int(round(val))
    if v <= 2:  return "2"
    if v == 3:  return "3"
    if v == 4:  return "4"
    return "5more"

def bucket_persons(val):
    if np.isnan(val): return np.nan
    v = int(round(val))
    if v <= 2:  return "2"
    if v <= 4:  return "4"
    return "more"

def normalize_doors_persons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "doors" in df.columns:
        uniq = set(str(x).lower() for x in df["doors"].dropna().unique())
        canonical = {"2", "3", "4", "5more"}
        if not uniq.issubset(canonical):
            df["doors"] = df["doors"].map(to_num).map(bucket_doors)
    if "persons" in df.columns:
        uniq = set(str(x).lower() for x in df["persons"].dropna().unique())
        canonical = {"2", "4", "more"}
        if not uniq.issubset(canonical):
            df["persons"] = df["persons"].map(to_num).map(bucket_persons)
    return df
# ----------------------------------------------------------------------

def translate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not APPLY_TRANSLATION:
        return df
    df = df.copy()
    for col, mapper in TRANSLATE_VALUES_BY_COL.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: mapper.get(x, x))
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(str).map(lambda x: TRANSLATE_TARGET.get(x, x))
    return df

def build_pipeline(df: pd.DataFrame, target: str):
    feature_cols = [c for c in df.columns if c != target]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    transformers = []
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("kbins", KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("num", num_pipe, num_cols))

    pre = ColumnTransformer(transformers)
    pipe = Pipeline([("pre", pre), ("clf", MultinomialNB())])
    return pipe, feature_cols, num_cols, cat_cols

def main():
    # 1) Load & clean
    df_raw = load_dataset()
    df_raw = df_raw.replace(r"^\s+$", np.nan, regex=True).dropna(how="all")
    for c in df_raw.columns:
        if pd.api.types.is_object_dtype(df_raw[c]):
            df_raw[c] = df_raw[c].astype(str).str.strip()

    # 2) Paksa doors/persons menjadi kategori jika numerik
    df_mid = normalize_doors_persons(df_raw)

    # 3) Terjemahkan nilai fitur & target ke Bahasa Indonesia
    df = translate_dataframe(df_mid)

    # 4) Build & train
    if TARGET not in df.columns:
        raise ValueError(f"TARGET '{TARGET}' tidak ditemukan. Kolom: {list(df.columns)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    pipe, feature_cols, num_cols, cat_cols = build_pipeline(df, TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels_sorted = sorted(pd.Series(y).unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 5) Simpan model
    joblib.dump(pipe, BASE / "model_pipeline.joblib")

    # 6) Siapkan label fitur (Indonesia) & opsi dropdown
    feature_labels = [TRANSLATE_FEATURES.get(c, c) for c in feature_cols]
    choices = {}
    for c_orig, c_label in zip(feature_cols, feature_labels):
        vals = pd.Series(X[c_orig]).dropna().astype(str).unique().tolist()
        choices[c_label] = sorted(vals, key=lambda x: x)

    # 7) Simpan meta (termasuk Confusion Matrix dalam angka)
    meta = {
        "target_col": "Kelayakan Mobil",
        "feature_cols": feature_labels,
        "numeric_cols": [TRANSLATE_FEATURES.get(c, c) for c in num_cols],
        "categorical_cols": [TRANSLATE_FEATURES.get(c, c) for c in cat_cols],
        "choices": choices,
        "accuracy": float(acc),
        "labels": labels_sorted,
        "cm": cm.tolist(),            # <-- simpan angka CM
        "n_train": int(y_train.shape[0]),
        "n_test": int(y_test.shape[0]),
        "report": report,
    }
    (BASE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Selesai. Akurasi={acc:.4f}. Model & meta tersimpan.")

if __name__ == "__main__":
    main()
