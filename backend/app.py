from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {}


def init_models():
    print("Initializing models... please wait.")
    try:
        # Load Scikit-learn models (Pickle)
        for m_name in ["diabetes", "heart", "breast_cancer", "kidney", "liver"]:
            path = os.path.join(BASE_DIR, "models", f"{m_name}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    MODELS[m_name] = pickle.load(f)

        # Load Keras models (Pneumopnia and Malaria)
        malaria_path = os.path.join(BASE_DIR, "models", "malaria.h5")
        if os.path.exists(malaria_path):
            MODELS["malaria"] = load_model(malaria_path)

        pneumonia_path = os.path.join(
            BASE_DIR, "models/output/models/", "vgg16_pneumonia.h5"
        )
        if os.path.exists(pneumonia_path):
            MODELS["pneumonia"] = load_model(pneumonia_path)

        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")


# Call initialization (it initializes the MODELS dictionary)
init_models()


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        disease = data.get("disease")
        features_list = data.get("features")

        if not features_list:
            return jsonify({"message": "No features provided"}), 400

        # Mapping request keys to loaded model keys
        model_key_map = {
            "diabetes": "diabetes",
            "heart-disease": "heart",
            "breast-cancer": "breast_cancer",
            "kidney-disease": "kidney",
            "liver-disease": "liver",
        }
        # Hint: feature counts for each disease
        # diabetes=8
        # breast-cancer=26
        # heart disease=13
        # liver =9
        # kidney =24

        model = MODELS.get(model_key_map.get(disease))
        if not model:
            return jsonify({"message": f"Model for {disease} not loaded"}), 400

        features = np.array(features_list).reshape(1, -1)
        prediction = int(model.predict(features)[0])

        return jsonify(
            {
                "prediction": prediction,
                "message": (
                    "Positive indication found. Please consult a doctor."
                    if prediction == 1
                    else "No significant indicators found."
                ),
            }
        )
    except Exception as e:
        return jsonify({"message": f"Server Error: {str(e)}"}), 500


@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"message": "No image uploaded"}), 400

        disease = request.form.get("disease")
        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        if disease == "malaria":
            model = MODELS.get("malaria")
            if not model:
                return jsonify({"message": "Malaria model not found"}), 500

            # FIX: Resize to 33 and Normalize
            img = img.resize((33, 33))
            img_array = np.asarray(img).astype(np.float32) / 255.0
            img_array = img_array.reshape((1, 33, 33, 3))

            prediction = model.predict(img_array)
            pred_class = np.argmax(prediction[0])

            if pred_class == 0:
                msg, status = "Infection detected (Parasitized)", 1
            else:
                msg, status = "Clear (Uninfected)", 0

        elif disease == "pneumonia":
            model = MODELS.get("pneumonia")
            if not model:
                return jsonify({"message": "Pneumonia model not found"}), 500

            img = img.resize((150, 150))
            img_array = np.asarray(img).astype(np.float32) / 255.0
            img_array = img_array.reshape((1, 150, 150, 3))

            # Raw probability score
            prediction_score = model.predict(img_array)[0][0]
            print(f"DEBUG: Pneumonia Raw Score: {prediction_score}")

            if prediction_score > 0.5:
                status = 0
                msg = "Lungs appear clear"
            else:
                status = 1
                msg = "Pneumonia detected"

        else:
            return jsonify({"message": "Invalid disease type"}), 400

        return jsonify({"prediction": status, "message": msg})

    except Exception as e:
        return jsonify({"message": f"Image processing error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
