from nlp_utils import extract_keywords
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import webbrowser
from threading import Timer
import csv
from datetime import datetime
import cv2
from gradcam_utils import get_img_array, make_gradcam_heatmap
import tensorflow as tf
import collections
import numpy as np
import json
import os
import requests


from preprocess import preprocess_image  # yahi se aa raha hai

print("=== app.py starting ===")

app = Flask(__name__)
print("=== flask app created ===")

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model and remedies
model = tf.keras.models.load_model('models/ayurspace_model.h5')
print("=== model loaded ===")

with open('remedies.json', 'r', encoding='utf-8') as f:
    remedies_db = json.load(f)
print("=== remedies loaded ===")

# Order MUST match train_gen.class_indices
PLANT_CLASSES = ['aloe_vera', 'ashwagandha', 'neem', 'tulsi']

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LOG_FILE = "prediction_history.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "plant_key", "confidence", "lang", "client_ip"])

LAST_CONV_LAYER = "conv5_block3_out"  # ResNet50 last conv layer

# ---------- Simple Q&A knowledge base ----------
QA_KB = {
    "tulsi": {
        "en": [
            {
                "patterns": ["cough", "cold", "sore throat"],
                "answer": "For cough and sore throat, boil tulsi leaves in water and drink warm 2–3 times a day."
            },
            {
                "patterns": ["fever", "flu"],
                "answer": "For mild fever or flu, a tulsi decoction with ginger and honey is commonly used."
            }
        ]
    },
    "neem": {
        "en": [
            {
                "patterns": ["skin", "acne", "pimples"],
                "answer": "For skin problems, neem leaf paste can be applied on the cleaned affected area."
            }
        ]
    }
    # baaki plants ke liye bhi 1–2 intents baad me add kar sakte ho
}

# ---------- static uploads serve ----------
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------- main prediction helper ----------
def predict_plant(image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)[0]
    predicted_class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100.0)
    predicted_plant = PLANT_CLASSES[predicted_class_idx]
    return predicted_plant, confidence

# ---------- Grad-CAM API ----------
@app.route("/gradcam", methods=["POST"])
def gradcam():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # 1. Normal prediction
        img_array = get_img_array(filepath, size=(224, 224))
        preds = model.predict(img_array)[0]
        class_idx = int(np.argmax(preds))
        plant_key = PLANT_CLASSES[class_idx]
        confidence = float(np.max(preds) * 100.0)

        # 2. Heatmap banao
        heatmap = make_gradcam_heatmap(
            img_array, model, last_conv_layer_name=LAST_CONV_LAYER,
            pred_index=class_idx
        )

        # 3. Overlay original image par
        orig = cv2.imread(filepath)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.5, heatmap_color, 0.5, 0)

        gradcam_name = "gradcam_" + filename
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_name)
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return jsonify({
            "plant_key": plant_key,
            "confidence": confidence,
            "heatmap_image": gradcam_name
        })

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ---------- normal HTML page ----------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

@app.route('/')
def index():
    return render_template('index.html')

# ---------- main /predict API ----------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    lang = request.form.get('lang', 'en')
    if lang not in ('en', 'hi', 'mr'):
        lang = 'en'
    suffix = '_' + lang

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        plant_key, confidence = predict_plant(filepath)

        if plant_key not in remedies_db:
            return jsonify({'error': 'Plant not found in remedies DB'}), 500

        info = remedies_db[plant_key]

        # --- keyword extraction (English text base) ---
        uses_en = info.get('uses_en', [])
        remedy_en = info.get('remedy_en', "")
        full_text_en = " ".join(uses_en) + " " + remedy_en
        keywords = extract_keywords(full_text_en, top_k=5)  # simple freq-based keywords [web:151]

        response = {
            'plant_key': plant_key,
            'plant_name': info['name' + suffix],
            'uses': info['uses' + suffix],
            'remedy': info['remedy' + suffix],
            'precautions': info['precautions' + suffix],
            'confidence': float(confidence),
            'low_confidence': bool(confidence < 50.0),
            'keywords': keywords       # NEW field for UI tags
        }

        # history me save karo
        try:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    plant_key,
                    f"{confidence:.2f}",
                    lang,
                    request.remote_addr or ""
                ])
        except Exception as e:
            print("Log write error:", e)

        return jsonify(response)

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ---------- External AI fallback config (OpenAI) ----------
AI_API_URL = "https://api.openai.com/v1/chat/completions"
AI_MODEL   = "gpt-4o-mini"   # ya "gpt-4o" agar account allow karta ho
AI_API_KEY = os.environ.get("AI_API_KEY")

def ask_external_ai(question: str) -> str:
    """
    OpenAI chat-completions style AI call.
    """
    if not AI_API_KEY:
        return "AI service is not configured (missing API key)."

    try:
        headers = {
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": AI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for an Ayurvedic plant app. "
                        "Answer briefly and clearly. Do NOT prescribe medicines; "
                        "only give general wellness information based on herbal traditions."
                    )
                },
                {"role": "user", "content": question}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }

        resp = requests.post(AI_API_URL, headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            print("AI API error status:", resp.status_code, resp.text[:200])
            return "Sorry, AI service is temporarily unavailable."

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "Sorry, AI could not generate a useful answer."

        answer_text = choices[0]["message"]["content"]
        return answer_text.strip()

    except Exception as e:
        print("AI API exception:", e)
        return "Sorry, there was an error contacting the AI service."


# ---------- Q&A API ----------
@app.route("/qa", methods=["POST"])
def qa():
    data = request.get_json(force=True)
    query = data.get("question", "").lower()
    lang = data.get("lang", "en")

    if not query:
        return jsonify({"answer": "Please type a question."})

    # ----- Level 1: Knowledge-base (QA_KB) -----
    best_answer = None
    best_score = 0

    for plant_key, lang_block in QA_KB.items():
        intents = lang_block.get(lang, [])
        for intent in intents:
            score = sum(1 for kw in intent["patterns"] if kw in query)
            if score > best_score:
                best_score = score
                best_answer = {
                    "plant_key": plant_key,
                    "answer": intent["answer"]
                }

    if best_answer and best_score > 0:
        # KB se direct match mila
        return jsonify(best_answer)

    # ----- Level 2: External AI fallback (OpenAI) -----
    ai_answer = ask_external_ai(query)

    return jsonify({
        "answer": (
            ai_answer
            + "\n\n(Note: This is an AI-generated general explanation, "
              "not a medical prescription. Consult a qualified practitioner for treatment.)"
        )
    })


# ---------- stats API ----------
@app.route('/stats')
def stats():
    if not os.path.exists(LOG_FILE):
        return jsonify({"total": 0, "by_plant": {}, "by_lang": {}})

    plant_counter = collections.Counter()
    lang_counter = collections.Counter()
    total = 0

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        next(f, None)  # header skip
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            _, plant_key, confidence, lang, *_ = parts
            total += 1
            plant_counter[plant_key] += 1
            lang_counter[lang] += 1

    return jsonify({
        "total": total,
        "by_plant": plant_counter,
        "by_lang": lang_counter
    })

# ---------- run server ----------
if __name__ == '__main__':
    print("=== starting flask server ===")
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()
    app.run(debug=True, port=5001)
