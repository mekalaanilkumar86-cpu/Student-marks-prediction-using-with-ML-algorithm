"""Prediction marks module using machine learning models."""
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Your trained model (e.g. LinearRegression) saved as model.pkl
model = joblib.load("Student_Marks_Prediction_Model.pkl")


@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Predict student marks based on input features."""
    if request.method == "POST":
        return render_template("index.html")

    hours = float(request.form["hours"])
    marks = float(np.ravel(model.predict(np.array([[hours]])))[0])
    text = f"If you study {hours:g} hours then you can get {marks:.2f}% marks"
    return render_template("index.html", prediction_text=text)


if __name__ == "__main__":
    app.run(debug=True)
