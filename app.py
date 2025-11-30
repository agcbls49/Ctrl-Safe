import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
with open("models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the model
# with open("models/voting_nb_svm.pkl", "rb") as f:
#     model = pickle.load(f)

# Load the model
# with open("models/nb_model.pkl", "rb") as f:
#     model = pickle.load(f)

# Load the vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        text = request.form.get("input_text", "")
        if text:
            # Transform input text using the vectorizer
            X = vectorizer.transform([text])
            # Get prediction
            pred_label = model.predict(X)[0]
            pred_prob = model.predict_proba(X).max() * 100
            prediction = {
                "label": "Hate/Offensive" if pred_label == 1 else "Neutral",
                "probability": f"{pred_prob:.2f}"
            }
    return render_template("home.html", prediction=prediction)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
