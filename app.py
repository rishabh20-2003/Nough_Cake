from preprocessing import SNV, BaselineCorrection, SavitzkyGolay, MSC
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load all models into memory
model_dir = "saved_models"
models = {
    name.replace(".pkl", ""): load(os.path.join(model_dir, name))
    for name in os.listdir(model_dir) if name.endswith(".pkl")
}

@app.route("/")
def home():
    return "NIRS Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    model_name = request.form.get("model")

    if not file or model_name not in models:
        return jsonify({"error": "Missing file or invalid model name"}), 400

    try:
        df = pd.read_excel(file)
        df.columns = df.columns.astype(str)
        spectral_cols = df.columns[df.columns.str.fullmatch(r'\\d+')]
        if spectral_cols.empty:
            return jsonify({"error": "No valid spectral columns found"}), 400

        X = df[spectral_cols].astype(float).values

        # Predict
        model = models[model_name]
        predictions = model.predict(X)

        df['Predicted Protein'] = predictions
        results = df[['Predicted Protein']].to_dict(orient='records')

        return jsonify({
            "model_used": model_name,
            "predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)