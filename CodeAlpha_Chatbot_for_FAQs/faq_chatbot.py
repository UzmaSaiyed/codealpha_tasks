"""
TASK 2: FAQ Chatbot using NLP

This chatbot:
  1. Stores a list of FAQs (questions + answers)
  2. Cleans and processes user input using NLTK
  3. Finds the most similar FAQ using TF-IDF + Cosine Similarity
  4. Returns the best matching answer

Install requirements (run once):
    pip install nltk scikit-learn flask flask-cors

Then run:
    python faq_chatbot.py
    Open browser at: http://localhost:5000
"""

import re            # regular expressions: used to remove punctuation
import string        # gives us all punctuation characters easily
import nltk                                     # Natural Language Toolkit
from nltk.tokenize import word_tokenize         # splits sentence into individual words
from nltk.corpus import stopwords               # common words like "the", "is", "a"
from nltk.stem import WordNetLemmatizer         # reduces words to their root form
from sklearn.feature_extraction.text import TfidfVectorizer   # converts text to numbers
from sklearn.metrics.pairwise import cosine_similarity        # measures how similar two texts are
from flask import Flask, request, jsonify, send_from_directory  # builds a simple web API
from flask_cors import CORS                                      # allows browser to call our API

nltk.download("punkt",       quiet=True)   # needed for word_tokenize
nltk.download("punkt_tab",   quiet=True)   # updated tokenizer data
nltk.download("stopwords",   quiet=True)   # list of common words to ignore
nltk.download("wordnet",     quiet=True)   # dictionary used for lemmatization


#  SECTION 1 – FAQ Data  (topic: a general tech product / company)
#  Add or edit entries here to customise for any domain you like.

FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "You can return any product within 30 days of purchase for a full refund, "
                  "as long as the item is unused and in its original packaging."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page, enter your email address, "
                  "and we will send you a reset link within 2 minutes."
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Yes! Orders above ₹999 qualify for free standard shipping. "
                  "Express delivery is available for a small additional fee."
    },
    {
        "question": "How can I track my order?",
        "answer": "After placing an order you will receive a confirmation email with a tracking link. "
                  "You can also track your order under 'My Orders' in your account dashboard."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept UPI, credit/debit cards (Visa, Mastercard, RuPay), net banking, "
                  "wallets like Paytm and PhonePe, and cash on delivery."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach our support team 24/7 via live chat on the website, "
                  "email at support@example.com, or call us at 1800-123-4567 (toll-free)."
    },
    {
        "question": "Can I change or cancel my order?",
        "answer": "Orders can be changed or cancelled within 1 hour of placing them. "
                  "After that, please wait for delivery and use the return process."
    },
    {
        "question": "Is my personal data safe?",
        "answer": "Absolutely. We use 256-bit SSL encryption for all transactions and never "
                  "sell your personal data to third parties. See our Privacy Policy for details."
    },
    {
        "question": "Do you have a mobile app?",
        "answer": "Yes! Our app is available for free on both the Google Play Store (Android) "
                  "and the Apple App Store (iOS). Search for 'ShopEasy'."
    },
    {
        "question": "What is the warranty on products?",
        "answer": "Most electronics come with a 1-year manufacturer warranty. "
                  "Extended warranty plans of up to 3 years are available at checkout."
    },
    {
        "question": "How long does delivery take?",
        "answer": "Standard delivery takes 3-5 business days. Express delivery arrives within "
                  "1-2 business days. Same-day delivery is available in select cities."
    },
    {
        "question": "How do I apply a coupon or promo code?",
        "answer": "On the checkout page, you will see a 'Apply Coupon' field. "
                  "Enter your code there and click Apply. The discount will be shown instantly."
    },
    {
        "question": "Can I buy a product as a gift for someone?",
        "answer": "Yes! During checkout, tick the 'This is a gift' box. You can add a personal "
                  "message and we will ship it in special gift packaging directly to the recipient."
    },
    {
        "question": "What happens if I receive a damaged product?",
        "answer": "We are sorry to hear that! Please take a photo of the damage and contact our "
                  "support team within 48 hours. We will arrange a free replacement or full refund."
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Currently we ship to India, the UAE, Singapore, and the USA. "
                  "International orders typically arrive within 7-10 business days."
    },
]


#  SECTION 2 – Text Preprocessing  (NLP cleaning pipeline)

# Create a lemmatizer once (expensive to create each time)
# Lemmatizer converts words like "running" → "run", "orders" → "order"
lemmatizer = WordNetLemmatizer()

# Load the set of English stopwords: words like "the", "is", "at" that carry
# almost no meaning and would confuse our similarity calculation
STOPWORDS = set(stopwords.words("english"))


def preprocess(text):
    """
    Clean and normalise a sentence so it can be compared mathematically.

    Steps:
      1. Lowercase everything            → "What IS" becomes "what is"
      2. Remove punctuation              → "reset?" becomes "reset"
      3. Tokenise into words             → "reset password" → ["reset", "password"]
      4. Remove stopwords               → removes "how", "do", "i", etc.
      5. Lemmatize                       → "resetting" → "reset"

    Returns a single cleaned string of the important words.
    """

    # Step 1 – lowercase so "Password" and "password" are treated the same
    text = text.lower()

    # Step 2 – remove every punctuation character using a regex
    # re.sub replaces anything that is NOT a letter or digit with a space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Step 3 – split the sentence into a list of individual words (tokens)
    tokens = word_tokenize(text)

    # Steps 4 & 5 – remove stopwords AND lemmatize in a single loop
    cleaned_tokens = [
        lemmatizer.lemmatize(word)   # reduce to root form
        for word in tokens
        if word not in STOPWORDS     # keep only meaningful words
        and word.strip()             # skip empty strings
    ]

    # Join the cleaned words back into a single string
    return " ".join(cleaned_tokens)


#  SECTION 3 – Build the TF-IDF index  (done once at startup, not per query)

# Preprocess every FAQ question and store the cleaned versions
# This is done ONCE so we don't repeat the work for every user query
cleaned_questions = [preprocess(faq["question"]) for faq in FAQ_DATA]

# TF-IDF stands for Term Frequency – Inverse Document Frequency
# It converts each text into a vector of numbers:
#   - words that appear often in ONE question but rarely elsewhere get HIGH scores
#   - words that appear in EVERY question get LOW scores (not very useful)
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform: learn the vocabulary from all questions, then build vectors
# Result: a matrix where each row = one FAQ question, each column = one word
faq_matrix = tfidf_vectorizer.fit_transform(cleaned_questions)


#  SECTION 4 – Query function  (called for every user message)

# How similar the best match must be (0=no match, 1=perfect match)
# Below this threshold we say "I don't know"
SIMILARITY_THRESHOLD = 0.15


def find_best_answer(user_question):
    """
    Given a user's question, find the most similar FAQ and return its answer.

    How it works:
      1. Clean the user's question the same way we cleaned the FAQ questions
      2. Convert it to a TF-IDF vector
      3. Compute cosine similarity against all FAQ vectors
         (cosine similarity = how much do these two vectors point in the same direction?)
      4. Pick the FAQ with the highest similarity score
      5. If the score is too low, admit we don't know

    Returns a dict with: answer, matched_question, confidence (0-100)
    """

    # Clean the user's input exactly like we cleaned the FAQ questions
    cleaned_input = preprocess(user_question)

    # If after cleaning nothing meaningful is left (e.g. user typed "???")
    if not cleaned_input.strip():
        return {
            "answer": "Could you please rephrase your question? I didn't catch that.",
            "matched_question": None,
            "confidence": 0,
        }

    # Convert the user's cleaned question into a TF-IDF vector
    # transform() uses the same vocabulary learned during fit() above
    user_vector = tfidf_vectorizer.transform([cleaned_input])

    # Compute cosine similarity between the user vector and ALL FAQ vectors at once
    # cosine_similarity returns a 2D array; [0] gets the first (only) row
    similarities = cosine_similarity(user_vector, faq_matrix)[0]

    # Find the index of the FAQ with the HIGHEST similarity score
    best_index = similarities.argmax()

    # Get the actual similarity score for that best match
    best_score = float(similarities[best_index])

    # If the best match is still very weak, the question is probably out-of-scope
    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer": "I'm sorry, I don't have information about that. "
                      "Please contact our support team at support@example.com.",
            "matched_question": None,
            "confidence": 0,
        }

    # Return the matched answer along with metadata for the UI
    return {
        "answer": FAQ_DATA[best_index]["answer"],
        "matched_question": FAQ_DATA[best_index]["question"],
        "confidence": round(best_score * 100, 1),   # convert 0-1 to 0-100%
    }


#  SECTION 5 – Flask Web API  (serves the chat UI and handles /chat requests)

# Create the Flask web application
app = Flask(__name__, static_folder=".")

# Allow the browser (running on the same machine) to call our API
CORS(app)


@app.route("/")
def index():
    """Serve the chat UI HTML file when user opens http://localhost:5000"""
    return send_from_directory(".", "faq_chat_ui.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint: receives a user message, returns a chatbot response.
    Expects JSON body: { "message": "How do I reset my password?" }
    Returns  JSON:     { "answer": "...", "matched_question": "...", "confidence": 87.3 }
    """

    # Parse the incoming JSON body
    data = request.get_json()

    # Get the user's message; default to empty string if key is missing
    user_message = data.get("message", "").strip()

    # Return an error if the message is empty
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Find the best FAQ answer
    result = find_best_answer(user_message)

    # Send the result back as JSON
    return jsonify(result)


@app.route("/faqs", methods=["GET"])
def get_faqs():
    """
    Optional endpoint that returns all FAQ questions (no answers).
    The UI uses this to show suggested questions to the user.
    """
    questions = [faq["question"] for faq in FAQ_DATA]
    return jsonify({"faqs": questions})


#  SECTION 6 – Entry point

if __name__ == "__main__":
    print("=" * 55)
    print("  FAQ Chatbot is running!")
    print("  Open your browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)

    # debug=True auto-reloads when you save the file (great for development)
    # host="0.0.0.0" makes it accessible on your local network too
    app.run(debug=True, host="0.0.0.0", port=5000)
