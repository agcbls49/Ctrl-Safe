import re
import pickle
from flask import Flask, render_template, request
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')



# Load Tagalog stopwords from JSON file
with open("stopwords-tl.json", "r", encoding="utf-8") as f:
    tagalog_stopwords = set(json.load(f))

# Load English stopwords
english_stopwords = set(stopwords.words("english"))

# Combine them
all_stopwords = english_stopwords.union(tagalog_stopwords)


# Cleaning Function
def clean_bilingual_text(text, remove_stopwords=True, normalize_slang=True):
    text = str(text).lower().strip()

    import re

    # Remove retweet markers
    text = re.sub(r"\brt\b", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove mentions
    text = re.sub(r"@\w+", '', text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", '', text)

    # Remove emojis / punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers-only tokens
    text = re.sub(r"\b\d+\b", "", text)

    # Normalize elongated words (helloooo → hello)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Normalize repeated punctuation
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    # Social media fillers
    fillers = r"\b(haha+|lmao+|lol+|hehe+|huhu+|ayy+|omg+|wtf+|tbh|idk|smh|fr|btw|irl)\b"
    text = re.sub(fillers, "", text)

    # Slang normalization
    if normalize_slang:
        slang_dict = {
            "u": "you",
            "ur": "your",
            "pls": "please",
            "po": "",
            "naman": "",
            "kasi": "because",
            "nga": "",
            "talaga": "",
            "ganun": "like that",
            "pano": "how",
            "di": "not",
            "diko": "i dont",
            "diba": "right",
            "nung": "when",
            "amp": "",
            "tite": "",
        }
        for word, replacement in slang_dict.items():
            text = re.sub(rf"\b{word}\b", replacement, text)

    # Remove extra characters
    text = re.sub(r"[^a-zA-Z0-9áéíóúñ'\s]", '', text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Apply stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in all_stopwords]

    return " ".join(tokens)

# Cyberbullying heuristic function
def is_cyberbullying(text):
    text = text.lower()
    
    # Heuristic 1: direct targeting (mentions, "you", "ikaw", "ka")
    h1 = bool(re.search(r'@\w+|you|ikaw|ka', text))
    
    # Heuristic 2: insults + pronoun pattern
    h2 = bool(re.search(r'(you|ikaw|ka).*(idiot|stupid|gago|tanga|bobo|slut|pokpok)', text))
    
    # Heuristic 3: threats
    h3 = bool(re.search(r'kill you|patayin kita|saktan kita|hurt you', text))
    
    # Heuristic 4: repeated slurs
    h4 = bool(re.search(r'(gago+|tanga+|bobo+)', text))
    
    # Combine heuristics
    return h1 or h2 or h3 or h4


app = Flask(__name__)

# Load the model
with open("models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the model
# with open("models/voting_nb_svm.pkl", "rb") as f:
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

            # Clean input text
            cleaned_text = clean_bilingual_text(text)

            # Vectorize
            X = vectorizer.transform([text])

            # Model prediction
            pred_label = model.predict(X)[0]
            pred_prob = model.predict_proba(X).max() * 100

            result_label = "Neutral"

            # If the model detects hateful/offensive content:
            if pred_label == 1:
                # Check cyberbullying heuristics
                if is_cyberbullying(text):
                    result_label = "Likely Cyberbullying"
                else:
                    result_label = "Hate/Offensive"
            else:
                result_label = "Neutral"

            prediction = {
                "label": result_label,
                "probability": f"{pred_prob:.2f}"
            }

    return render_template("home.html", prediction=prediction)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
