from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        email_text = request.form["email"]

        # Convert text to vector
        data = vectorizer.transform([email_text])

        # Predict
        result = model.predict(data)[0]

        if result == 1:
            prediction = "Spam Email"
        else:
            prediction = "Not Spam (Ham)"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
