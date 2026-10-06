"""
Small local API that wraps your MLflow model so a browser page can call it.

Why this exists instead of just using `mlflow models serve`:
`mlflow models serve` does not send CORS headers, so a browser page
(like our canvas UI) gets blocked from calling it directly. This script
loads the model once and exposes a plain endpoint with CORS enabled.

Run:
    pip install flask flask-cors mlflow
    python predict_api.py

Then open index.html in your browser.
"""

import numpy as np
import mlflow.pyfunc
from flask import Flask, request, jsonify
from flask_cors import CORS

MLFLOW_TRACKING_URI = "http://localhost:5000"   # your mlflow server
MODEL_URI = "models:/digit-classifier-cnn/1"    # name/version from the registry

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"Loading model from {MODEL_URI} ...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded. Ready for predictions.")

app = Flask(__name__)
CORS(app)  # allows the browser page to call this from a different origin


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pixels = data.get("image")  # expected: flat list of 784 floats, 0-1 range

    if pixels is None or len(pixels) != 784:
        return jsonify({"error": "expected 'image' as a flat list of 784 values"}), 400

    arr = np.array(pixels, dtype="float32").reshape(1, 28, 28, 1)
    preds = model.predict(arr)

    # preds shape is (1, 10) - probability per digit 0-9
    probs = np.array(preds).reshape(-1).tolist()
    predicted_digit = int(np.argmax(probs))

    return jsonify({
        "predicted_digit": predicted_digit,
        "probabilities": probs
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)