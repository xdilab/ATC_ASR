import os
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from transformers import AutoModelForCTC, Wav2Vec2Processor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "wav2vec2_finetuned_bart"

# ----------------------------
# METRICS
# ----------------------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("bleu")

# ----------------------------
# HELPERS
# ----------------------------
def generate_prediction(audio, model, processor):
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

def compute_cosine_similarity(pred, ref):
    try:
        vectorizer = TfidfVectorizer().fit([pred, ref])
        pred_vec = vectorizer.transform([pred])
        ref_vec = vectorizer.transform([ref])
        return cosine_similarity(pred_vec, ref_vec)[0][0]
    except ValueError:
        return 0.0

def evaluate_model(model, processor, dataset, dataset_name):
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    references = dataset["text"]
    predictions = []

    for i, example in enumerate(tqdm(dataset, desc=f"Predicting on {dataset_name}")):
        pred = generate_prediction(example["audio"], model, processor)
        ref = example["text"]

        if not pred.strip():
            print(f"Sample {i}: ⚠️ Empty prediction — REF: '{ref}'")
        else:
            print(f"Sample {i}: Predicted = {pred}")

        predictions.append(pred)

    results = []
    for ref, pred in zip(references, predictions):
        wer = wer_metric.compute(predictions=[pred], references=[ref])
        cer = cer_metric.compute(predictions=[pred], references=[ref])

        try:
            bleu = bleu_metric.compute(predictions=[pred], references=[[ref]])["bleu"]
        except Exception:
            bleu = 0.0

        cos_sim = compute_cosine_similarity(pred, ref)

        results.append({
            "Model": MODEL_DIR,
            "True Transcription": ref,
            "Predicted Transcription": pred,
            "WER": wer,
            "CER": cer,
            "BLEU": bleu,
            "Cosine Similarity": cos_sim
        })

    return pd.DataFrame(results)

# ----------------------------
# MAIN
# ----------------------------
def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model from {MODEL_DIR}...")
    model = AutoModelForCTC.from_pretrained(MODEL_DIR).to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model.eval()

    dataset_name = "uwb_atcc"
    dataset = load_dataset("Jzuluaga/uwb_atcc", split="test")

    print(f"\nEvaluating on {dataset_name}...")
    df = evaluate_model(model, processor, dataset, dataset_name)

    detailed_path = f"{MODEL_DIR}_{dataset_name}_detailed.csv"
    df.to_csv(detailed_path, index=False)
    print(f"Saved detailed results to {detailed_path}")

    metrics = ["WER", "CER", "BLEU", "Cosine Similarity"]
    summary = pd.DataFrame({
        "Metric": metrics,
        "Mean": df[metrics].mean().values,
        "Median": df[metrics].median().values
    })
    summary_path = f"{MODEL_DIR}_{dataset_name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
