# from flask import Flask, render_template, request, flash, redirect
# import pickle
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model

# app = Flask(__name__)


# def predict(values, dic):
#     if len(values) == 8:
#         model = pickle.load(open("models/diabetes.pkl", "rb"))
#         values = np.asarray(values)
#         return model.predict(values.reshape(1, -1))[0]
#     elif len(values) == 26:
#         model = pickle.load(open("models/breast_cancer.pkl", "rb"))
#         values = np.asarray(values)
#         return model.predict(values.reshape(1, -1))[0]
#     elif len(values) == 13:
#         model = pickle.load(open("models/heart.pkl", "rb"))
#         values = np.asarray(values)
#         return model.predict(values.reshape(1, -1))[0]
#     elif len(values) == 18:
#         model = pickle.load(open("models/kidney.pkl", "rb"))
#         values = np.asarray(values)
#         return model.predict(values.reshape(1, -1))[0]
#     elif len(values) == 10:
#         model = pickle.load(open("models/liver.pkl", "rb"))
#         values = np.asarray(values)
#         return model.predict(values.reshape(1, -1))[0]


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/contactus")
# def contactUs():
#     return render_template("contact.html")


# @app.route("/aboutus")
# def aboutUs():
#     return render_template("about-us.html")


# @app.route("/diabetes", methods=["GET", "POST"])
# def diabetesPage():
#     return render_template("diabetes.html")


# @app.route("/cancer", methods=["GET", "POST"])
# def cancerPage():
#     return render_template("breast_cancer.html")


# @app.route("/heart", methods=["GET", "POST"])
# def heartPage():
#     return render_template("heart.html")


# @app.route("/kidney", methods=["GET", "POST"])
# def kidneyPage():
#     return render_template("kidney.html")


# @app.route("/liver", methods=["GET", "POST"])
# def liverPage():
#     return render_template("liver.html")


# @app.route("/malaria", methods=["GET", "POST"])
# def malariaPage():
#     return render_template("malaria.html")


# @app.route("/predict", methods=["POST", "GET"])
# def predictPage():
#     try:
#         if request.method == "POST":
#             to_predict_dict = request.form.to_dict()
#             to_predict_list = list(map(float, list(to_predict_dict.values())))
#             pred = predict(to_predict_list, to_predict_dict)
#     except:
#         message = "Please enter valid Data"
#         return render_template("index.html", message=message)

#     return render_template("predict.html", pred=pred)


# @app.route("/malariapredict", methods=["POST", "GET"])
# def malariapredictPage():
#     if request.method == "POST":
#         try:
#             if "image" in request.files:
#                 img = Image.open(request.files["image"])
#                 img = img.resize((36, 36))
#                 img = np.asarray(img)
#                 img = img.reshape((1, 36, 36, 3))
#                 img = img.astype(np.float64)
#                 model = load_model("models/malaria.h5")
#                 pred = np.argmax(model.predict(img)[0])
#         except:
#             message = "Please upload an Image"
#             return render_template("malaria.html", message=message)
#     return render_template("malaria1_predict.html", pred=pred)


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)


def predict(values, dic):
    # Convert list to numpy array once
    values = np.asarray(values).reshape(1, -1)

    # Logic based on feature counts
    if len(values[0]) == 8:
        model = pickle.load(open("models/diabetes.pkl", "rb"))
        return model.predict(values)[0]
    elif len(values[0]) == 26:
        model = pickle.load(open("models/breast_cancer.pkl", "rb"))
        return model.predict(values)[0]
    elif len(values[0]) == 13:
        model = pickle.load(open("models/heart.pkl", "rb"))
        return model.predict(values)[0]
    elif len(values[0]) == 24:
        model = pickle.load(open("models/kidney.pkl", "rb"))
        return model.predict(values)[0]
    elif len(values[0]) == 10:
        model = pickle.load(open("models/liver.pkl", "rb"))
        return model.predict(values)[0]
    else:
        # This is why your code was failing silently
        raise ValueError(
            f"Expected 8, 10, 13, 18, or 26 features, but got {len(values[0])}"
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/contactus")
def contactUs():
    return render_template("contact.html")


@app.route("/aboutus")
def aboutUs():
    return render_template("about-us.html")


@app.route("/diabetes")
def diabetesPage():
    return render_template("diabetes.html")


@app.route("/cancer")
def cancerPage():
    return render_template("breast_cancer.html")


@app.route("/heart")
def heartPage():
    return render_template("heart.html")


@app.route("/kidney")
def kidneyPage():
    return render_template("kidney.html")


@app.route("/liver")
def liverPage():
    return render_template("liver.html")


@app.route("/malaria")
def malariaPage():
    return render_template("malaria.html")


@app.route("/predict", methods=["POST"])
def predictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            # This line crashes if any field is empty
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
            return render_template("predict.html", pred=pred)
    except Exception as e:
        # LOOK AT YOUR TERMINAL/CONSOLE TO SEE THE REAL ERROR
        print(f"DEBUG ERROR: {e}")
        message = f"Error: {str(e)}. Ensure all fields are filled correctly."
        return render_template("index.html", message=message)


@app.route("/malariapredict", methods=["POST"])
def malariapredictPage():
    try:
        if "image" in request.files:
            img = Image.open(request.files["image"])
            img = img.resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 3)).astype(np.float64)
            model = load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template("malaria1_predict.html", pred=pred)
    except Exception as e:
        print(f"MALARIA ERROR: {e}")
        return render_template("malaria.html", message="Please upload a valid Image")


if __name__ == "__main__":
    app.run(debug=True)
