from flask import Flask, render_template, request
import joblib, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
app = Flask(__name__)
# agar {% for f,v in zip(features, values) %} di result.html bisa jalan
app.jinja_env.globals.update(zip=zip)

# Mapping label UI (Indonesia) -> nama kolom saat training (Inggris)
UI_TO_MODEL = {
    "Harga Beli":       "buying",
    "Biaya Perawatan":  "maint",
    "Jumlah Pintu":     "doors",
    "Kapasitas Orang":  "persons",
    "Ukuran Bagasi":    "lug_boot",
    "Tingkat Keamanan": "safety",
}

def load_assets():
    pipe_path = BASE / "model_pipeline.joblib"
    meta_path = BASE / "meta.json"
    if not pipe_path.exists() or not meta_path.exists():
        return None, None
    pipe = joblib.load(pipe_path)
    meta = json.loads(meta_path.read_text())
    return pipe, meta

@app.route("/", methods=["GET", "POST"])
def index():
    pipe, meta = load_assets()
    if pipe is None:
        return render_template("not_ready.html")

    FEATURES_UI = meta["feature_cols"]     # label Indonesia untuk UI
    CHOICES     = meta["choices"]
    TARGET      = meta["target_col"]

    FEATURES_MODEL = [UI_TO_MODEL[f] for f in FEATURES_UI]

    if request.method == "POST":
        # input user yang kuncinya = label UI (Indonesia)
        raw_ui = {f: request.form.get(f, "") for f in FEATURES_UI}

        # kosong → NaN (biar SimpleImputer di pipeline aktif)
        for k, v in raw_ui.items():
            if v is None or v == "" or str(v).lower() == "(kosongkan)":
                raw_ui[k] = np.nan

        # map ke kolom model (Inggris), nilai tetap Indonesia (karena model dilatih dg nilai Indo)
        row_for_model = {UI_TO_MODEL[k]: raw_ui[k] for k in FEATURES_UI}
        X = pd.DataFrame([row_for_model], columns=FEATURES_MODEL)

        # Prediksi
        pred = pipe.predict(X)[0]

        # Probabilitas (jika tersedia)
        try:
            proba = pipe.predict_proba(X)[0]
            classes = list(pipe.classes_)
            probs = sorted(list(zip(classes, proba)), key=lambda x: x[1], reverse=True)
        except Exception:
            probs = None

        values_ui_order = [raw_ui[f] for f in FEATURES_UI]
        return render_template(
            "result.html",
            features=FEATURES_UI,
            values=values_ui_order,
            prediction=pred,
            probs=probs,
            meta=meta,
            target=TARGET
        )

    return render_template("index.html", features=FEATURES_UI, choices=CHOICES, meta=meta)

if __name__ == "__main__":
    app.run(debug=True)
