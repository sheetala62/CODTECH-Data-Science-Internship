from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Student Score Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    hours = request.json["hours"]
    prediction = model.predict([[hours]])
    return jsonify({"Predicted Score": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
