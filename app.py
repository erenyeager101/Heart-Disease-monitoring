# app.py
import os
import numpy as np
import joblib  # use joblib for robustness (works with pickle too)
from flask import Flask, render_template, request, jsonify, abort

# --- Flask App Config ---
app = Flask(__name__)

# --- Load Model Once ---
MODEL_PATH = "stacking_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# --- Feature Order (ensure consistent input order) ---
FEATURE_ORDER = [
    "age", "sex", "cpt", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]


# --- Utility: Validate and Parse Inputs ---
def parse_input(data_source):
    """
    Parse input features from form or JSON and convert to numpy array.
    Raises ValueError if any input is missing or invalid.
    """
    values = []
    for feature in FEATURE_ORDER:
        if feature not in data_source:
            raise ValueError(f"Missing feature: {feature}")
        val = data_source[feature]

        try:
            # float conversion handles both int/float input types
            val = float(val)
        except ValueError:
            raise ValueError(f"Invalid value for {feature}: {val}")

        values.append(val)

    return np.array([values])  # shape (1, n_features)


# --- ROUTES ---

@app.route("/")
def main_index():
    """Landing page"""
    return render_template("mainindex.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict endpoint (supports both HTML form & JSON)
    """
    try:
        # Get form data or JSON data
        if request.is_json:
            input_data = request.get_json()
        else:
            input_data = request.form

        data = parse_input(input_data)
        prediction = model.predict(data)

        result = int(prediction[0])  # ensure it's JSON-serializable

        # Handle web form
        if not request.is_json:
            return render_template("result.html", prediction=result)

        # Handle API JSON response
        return jsonify({"prediction": result})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Optional: Static Info Pages ---
@app.route("/<page>")
def render_page(page):
    """
    Dynamically render any template under 'templates/' without repeating routes.
    Example: /appointment -> templates/appointment.html
    """
    try:
        return render_template(f"{page}.html", prediction=None)
    except Exception:
        abort(404)


# --- Run Server ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
