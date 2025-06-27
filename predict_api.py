from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend communication

# === Load Model & Column Info ===
model = joblib.load("final_model.pkl")
spectral_cols = joblib.load("spectral_cols.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        df = pd.read_excel(file)

        if not all(col in df.columns for col in spectral_cols):
            return jsonify({"error": "Missing spectral columns in uploaded file"}), 400

        X = df[spectral_cols].values.astype(float)
        preds = model.predict(X).flatten()

        df["Predicted Protein%"] = preds

        result_summary = {
            "num_samples": len(df),
            "predictions": preds[:5].tolist(),  # sample output
            "columns": df.columns.tolist()
        }

        return jsonify(result_summary)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
