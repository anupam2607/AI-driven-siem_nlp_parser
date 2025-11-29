# src/nlp_infer.py
import os
import json
import sys
from typing import Dict
import pandas as pd

MODEL_DIR = os.path.join(os.getcwd(), "siem_nlp_model")
FALLBACK_PATH = os.path.join(os.getcwd(), "models", "fallback.pkl")
DATA_CSV = os.path.join(os.getcwd(), "data.csv")

# Try HF load
try:
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch
    hf_available = os.path.isdir(MODEL_DIR)
except Exception as e:
    print("HF transformers not available:", e)
    hf_available = False

# Fallback libs
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

_tokenizer = None
_model = None
_label_map = None
_fallback_pipeline = None


def _load_hf():
    global _tokenizer, _model, _label_map
    print(f"Loading HF model from {MODEL_DIR} ...")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()
    mapping_file = os.path.join(MODEL_DIR, "label_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            mapping = json.load(f)
            _label_map = mapping.get("id2label") or mapping.get("id_to_label") or mapping
    else:
        cfg = _model.config
        if hasattr(cfg, "id2label") and cfg.id2label:
            _label_map = {str(k): v for k, v in cfg.id2label.items()}
        else:
            _label_map = None
    print("HF model loaded successfully.")


def _train_fallback():
    global _fallback_pipeline
    if _fallback_pipeline is not None:
        return
    if os.path.exists(FALLBACK_PATH):
        print(f"Loading fallback model from {FALLBACK_PATH} ...")
        _fallback_pipeline = joblib.load(FALLBACK_PATH)
        print("Fallback model loaded successfully.")
        return
    if not os.path.exists(DATA_CSV):
        print("No data.csv found. Creating sample fallback data ...")
        os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
        sample_data = pd.DataFrame({
            "text": [
                "Suspicious login attempt",
                "Malware detected on endpoint",
                "Phishing email reported",
                "User login successful",
            ],
            "label": [
                "login_attack",
                "malware",
                "phishing",
                "normal"
            ]
        })
        sample_data.to_csv(DATA_CSV, index=False)
        print(f"Sample data.csv created at {DATA_CSV}")

    print("Training fallback model ...")
    df = pd.read_csv(DATA_CSV)
    d = df.dropna(subset=["text","label"])
    X = d["text"].astype(str)
    y = d["label"].astype(str)
    pipe = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1,2)), MultinomialNB())
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(FALLBACK_PATH), exist_ok=True)
    joblib.dump(pipe, FALLBACK_PATH)
    _fallback_pipeline = pipe
    print("Fallback model trained and saved.")


# Load appropriate model
if hf_available and os.path.isdir(MODEL_DIR):
    try:
        _load_hf()
    except Exception as e:
        print("Failed loading HF model:", e)
        hf_available = False

if not hf_available:
    try:
        _train_fallback()
    except Exception as e:
        print("Fallback model not ready:", e)


def predict_attack_category(text: str) -> str:
    text = str(text)
    if hf_available and _model is not None:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = _model(**inputs)
            pred = int(outputs.logits.argmax(dim=-1).cpu().numpy()[0])
        if _label_map:
            lbl = _label_map.get(str(pred)) or _label_map.get(pred) or str(pred)
        else:
            lbl = str(pred)
        return lbl
    else:
        if _fallback_pipeline is None:
            _train_fallback()
        return _fallback_pipeline.predict([text])[0]


def nlp_output_from_text(text: str) -> Dict:
    label = predict_attack_category(text)
    return {"intent": "fetch_logs", "entities": {"attack_category": label}}


if __name__ == "__main__":
    print("=== NLP Inference Script Started ===")
    
    # CLI input
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        print(f"Running inference on input text: {input_text}")
        output = nlp_output_from_text(input_text)
        print("NLP Output:", output)
    else:
        # Run sample texts
        sample_texts = [
            "Suspicious login attempt detected from unknown IP",
            "Endpoint malware detected",
            "Normal user login event",
            "Phishing email reported by user"
        ]
        for t in sample_texts:
            output = nlp_output_from_text(t)
            print(f"\nInput: {t}\nOutput: {output}")
    
    print("\n=== NLP Inference Script Finished ===")
