import os
import logging
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the static folder to 'dist' (where Vite builds the frontend)
base_dir = os.path.abspath(os.path.dirname(__file__))
static_folder = os.path.join(base_dir, 'dist')
app = Flask(__name__, static_folder=static_folder)
CORS(app)

# Log the application structure
logger.info(f"Base Directory: {base_dir}")
logger.info(f"Static Folder: {static_folder}")
if os.path.exists(static_folder):
    logger.info(f"Contents of dist: {os.listdir(static_folder)}")
else:
    logger.warning("dist folder NOT found!")

# ======================
# Load trained models
# ======================
try:
    linear_model = joblib.load("linear_model.pkl")
    logistic_model = joblib.load("logistic_model.pkl")
    logistic_scaler = joblib.load("logistic_scaler.pkl")
    svr_model = joblib.load("svr_model.pkl")
    svr_scaler = joblib.load("svr_scaler.pkl")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")

# API Route
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        size = float(data.get("size", 0))
        bedrooms = int(data.get("bedrooms", 0))
        age = int(data.get("age", 0))
        location = int(data.get("location", 0))

        user_input = np.array([[size, bedrooms, age, location]])

        # Linear Regression
        lr_price = linear_model.predict(user_input)[0]

        # SVR
        svr_scaled = svr_scaler.transform(user_input)
        svr_price = svr_model.predict(svr_scaled)[0]

        # Logistic Regression
        log_scaled = logistic_scaler.transform(user_input)
        log_pred = logistic_model.predict(log_scaled)[0]
        category = "High Price House" if log_pred == 1 else "Low Price House"

        response = {
            "best_model": "Logistic Regression",
            "comparison": [
                {
                    "model": "Linear Regression",
                    "prediction": f"₹ {round(lr_price,2)} Lakhs",
                    "performance": "R² = 0.99"
                },
                {
                    "model": "SVR",
                    "prediction": f"₹ {round(svr_price,2)} Lakhs",
                    "performance": "R² = 0.92"
                },
                {
                    "model": "Logistic Regression",
                    "prediction": category,
                    "performance": "Accuracy = 100%"
                }
            ]
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug")
def debug():
    try:
        dist_files = os.listdir(static_folder) if os.path.exists(static_folder) else "dist folder not found"
        return jsonify({
            "cwd": os.getcwd(),
            "base_dir": base_dir,
            "static_folder": static_folder,
            "root_files": os.listdir('.'),
            "dist_exists": os.path.exists(static_folder),
            "dist_files": dist_files
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Serve React App
@app.route("/", methods=["GET"])
def index():
    logger.info("Serving index.html from dist")
    return send_from_directory(static_folder, "index.html")

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    else:
        # SPA behavior: return index.html for unknown routes
        return send_from_directory(static_folder, "index.html")

if __name__ == "__main__":
    # Use PORT environment variable for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
