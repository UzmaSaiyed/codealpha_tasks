"""
TASK 3: Language Translation Tool — Backend (Flask + deep-translator)
This file runs a small web server that:
  1. Serves the translation UI (translator_ui.html)
  2. Accepts translation requests from the browser
  3. Uses deep-translator (free, no API key needed) to translate text
  4. Returns the translated text back to the browser as JSON

Install requirements (run once in your terminal):
    pip install flask flask-cors deep-translator langdetect

Run the server:
    python translator_backend.py
Then open:  http://localhost:5000
"""

import os           # used to find files on disk

from flask import Flask, request, jsonify, send_from_directory
# Flask  - creates our web server
# request - lets us read data sent by the browser
# jsonify - converts Python dicts to JSON responses
# send_from_directory - serves HTML/CSS/JS files to the browser

from flask_cors import CORS
# CORS - allows the browser to call our API from the same machine

from deep_translator import GoogleTranslator
# deep_translator wraps Google Translate for free — no API key required.
# It sends requests to translate.google.com just like a browser would.

from langdetect import detect, LangDetectException
# detect()  → figures out what language a piece of text is written in
# LangDetectException → error we catch if detection fails


#  Supported Languages
#  This dict maps human-readable names → language codes used by Google Translate

LANGUAGES = {
    "Auto Detect":  "auto",   # let the library figure out the source language
    "English":      "en",
    "Hindi":        "hi",
    "Spanish":      "es",
    "French":       "fr",
    "German":       "de",
    "Chinese (Simplified)": "zh-CN",
    "Japanese":     "ja",
    "Korean":       "ko",
    "Arabic":       "ar",
    "Portuguese":   "pt",
    "Russian":      "ru",
    "Italian":      "it",
    "Dutch":        "nl",
    "Turkish":      "tr",
    "Bengali":      "bn",
    "Gujarati":     "gu",
    "Tamil":        "ta",
    "Telugu":       "te",
    "Punjabi":      "pa",
    "Urdu":         "ur",
    "Marathi":      "mr",
    "Swedish":      "sv",
    "Polish":       "pl",
    "Greek":        "el",
    "Vietnamese":   "vi",
    "Thai":         "th",
    "Indonesian":   "id",
    "Malay":        "ms",
    "Ukrainian":    "uk",
    "Hebrew":       "iw",
}


#  Flask App Setup

# Create the Flask application.
# static_folder="." means Flask can serve files from the current directory.
app = Flask(__name__, static_folder=".")

# Allow the browser (on the same machine) to call our API without CORS errors
CORS(app)


# Routes (URL endpoints)

@app.route("/")
def index():
    """
    When the user opens http://localhost:5000 in their browser,
    send them the HTML translation interface.
    """
    return send_from_directory(".", "translator_ui.html")


@app.route("/languages", methods=["GET"])
def get_languages():
    """
    Return the full list of supported languages as JSON.
    The browser calls this once at startup to populate the dropdowns.

    Returns: { "languages": [{"name": "English", "code": "en"}, ...] }
    """
    # Build a list of dicts from our LANGUAGES dictionary
    lang_list = [
        {"name": name, "code": code}
        for name, code in LANGUAGES.items()
    ]
    return jsonify({"languages": lang_list})


@app.route("/translate", methods=["POST"])
def translate():
    """
    Main translation endpoint.
    The browser sends JSON: { "text": "...", "source": "en", "target": "hi" }
    We translate and send back: { "translated": "...", "detected_language": "..." }
    """

    # Read the incoming JSON body 
    data        = request.get_json()          # parse JSON from browser
    text        = data.get("text", "").strip()  # the text to translate
    source_code = data.get("source", "auto")  # source language code
    target_code = data.get("target", "en")    # target language code

    # Validate: don't process empty text 
    if not text:
        # 400 = "Bad Request" — the browser sent invalid data
        return jsonify({"error": "Please enter some text to translate."}), 400

    #  Validate: can't translate to the same language 
    if source_code != "auto" and source_code == target_code:
        return jsonify({"error": "Source and target languages must be different."}), 400

    # Optional: auto-detect the source language 
    detected_language = None   # will hold the detected language name (if auto)

    if source_code == "auto":
        try:
            # langdetect analyses patterns in the text and returns a code like "en", "fr"
            detected_code = detect(text)
            # Look up the human-readable name for this code
            detected_language = next(
                (name for name, code in LANGUAGES.items() if code == detected_code),
                detected_code    # fall back to showing just the code if name not found
            )
        except LangDetectException:
            # langdetect can fail on very short or ambiguous text — that's fine
            detected_language = "Unknown"

    # Perform the translation 
    try:
        # Create a GoogleTranslator for this specific language pair
        # source="auto" → Google Translate detects the language automatically
        translator = GoogleTranslator(source=source_code, target=target_code)

        # .translate() sends the text to Google Translate and returns the result
        translated_text = translator.translate(text)

    except Exception as e:
        # Something went wrong (network error, unsupported language pair, etc.)
        # 500 = "Internal Server Error"
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

    # Return the result as JSON 
    return jsonify({
        "translated":         translated_text,    # the translated string
        "detected_language":  detected_language,  # e.g. "English" (only if auto-detect was used)
        "char_count":         len(text),          # how many characters were translated
    })


@app.route("/detect", methods=["POST"])
def detect_language():
    """
    Lightweight endpoint: just detect the language of text without translating.
    Used by the UI to show the detected language as the user types.

    Expects: { "text": "Bonjour le monde" }
    Returns: { "language": "French", "code": "fr" }
    """
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text) < 5:
        # Too short to detect reliably
        return jsonify({"language": None, "code": None})

    try:
        code = detect(text)  # get the language code, e.g. "fr"
        # Convert the code back to a human-readable name
        name = next(
            (n for n, c in LANGUAGES.items() if c == code),
            code   # fall back to code if we don't have a name for it
        )
        return jsonify({"language": name, "code": code})
    except LangDetectException:
        return jsonify({"language": None, "code": None})


#  Entry Point

if __name__ == "__main__":
    print("=" * 55)
    print("  Language Translation Tool is running!")
    print("  Open your browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)

    # debug=True → auto-reloads when you save this file (great for development)
    app.run(debug=True, host="0.0.0.0", port=5000)
