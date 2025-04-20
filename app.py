from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from collections import Counter

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load NER model and tokenizer (CPU only)
model_path = "model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
model.eval()

# Extract text from uploaded PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract named entities using NER model
def extract_entities(text):
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    
    offset_mapping = encoded.pop("offset_mapping")  # remove it before passing to the model
    tokens = {k: v for k, v in encoded.items()}  # keep everything on CPU

    with torch.no_grad():
        output = model(**tokens)

    predictions = torch.argmax(output.logits, dim=-1).squeeze().tolist()
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())
    offset_mapping = offset_mapping.squeeze().tolist()

    labels = model.config.id2label

    entities = []
    current_entity = ""
    current_label = None
    for token, label_id, offset in zip(token_list, predictions, offset_mapping):
        if token.startswith("##"):
            current_entity += token[2:]
        else:
            if current_entity and current_label:
                entities.append((current_entity, current_label))
            current_label = labels[label_id] if label_id != 0 else None
            current_entity = token if current_label else ""

    if current_entity and current_label:
        entities.append((current_entity, current_label))

    return entities


# Compute score based on overlapping entity types
def score_match(cv_text, jd_text):
    cv_entities = extract_entities(cv_text)
    jd_entities = extract_entities(jd_text)

    # Extract only the actual entity text (lowercased for comparison)
    cv_set = set(ent[0].lower() for ent in cv_entities)
    jd_set = set(ent[0].lower() for ent in jd_entities)

    overlap = cv_set & jd_set
    total = len(jd_set)

    match_score = len(overlap) / total if total > 0 else 0.0
    return round(match_score * 100, 2), cv_set, jd_set


# Route
@app.route("/", methods=["GET", "POST"])
def index():
    match_score = None
    show_score = False
    resume_text = None

    if request.method == "POST":
        resume = request.files.get("resume")
        jd_text = request.form.get("jd_text", "").strip()

        if resume and jd_text:
            resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
            resume.save(resume_path)

            resume_text = extract_text_from_pdf(resume_path)
            match_score, _, _ = score_match(resume_text, jd_text)
            show_score = True

    return render_template("index.html",
                           match_score=match_score,
                           show_score=show_score)

if __name__ == "__main__":
    app.run(debug=True)
