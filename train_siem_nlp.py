# train_siem_nlp.py
import pandas as pd
import numpy as np
import re
import spacy
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

def clean_text(text, nlp, stops):
    """
    Clean and lemmatize text:
    - Lowercase
    - Remove URLs, numbers, special chars
    - Lemmatize using spacy
    - Remove stopwords and short tokens
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stops and len(token.text) > 2]
    return ' '.join(tokens)

def train_model(input_csv="siem_nlp_ready.csv", output_dir="siem_nlp_model", eval_strategy="epoch"):
    # NLTK stopwords
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english"))

    # Spacy model
    nlp = spacy.load("en_core_web_sm")

    # Load dataset
    df = pd.read_csv(input_csv)
    df.dropna(inplace=True)
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, nlp, stops))

    # Encode labels
    label2id = {l: i for i, l in enumerate(df["label"].unique())}
    id2label = {v: k for k, v in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # Train-test split
    X_train, X_test = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Prepare HuggingFace datasets
    train_ds = Dataset.from_pandas(
        X_train[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "label"})
    )
    test_ds = Dataset.from_pandas(
        X_test[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "label"})
    )

    # Tokenization
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label2id)
    )

    # Detect GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    model.to(device)

    # TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,       # ✅ correct argument
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),    # ✅ automatic FP16 if GPU available
        report_to="none",                  # disable wandb or other reporting
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training finished!")

    # Evaluate
    print("📊 Evaluating model on test set...")
    preds = trainer.predict(test_ds)
    y_true = test_ds["label"]
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(classification_report(y_true, y_pred, target_names=list(label2id.keys())))

    # Save model, tokenizer, and label mapping
    print(f"💾 Saving model, tokenizer, and label mapping to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)
    print("✅ Model, t+okenizer, and label mapping saved successfully!")

if __name__ == "__main__":
    train_model(eval_strategy="epoch")
