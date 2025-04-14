import os
import pandas as pd
import torch
import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "bart_finetuned_model"
CSV_PATH = "llm_evaluation_summary.csv"
OUTPUT_CSV = "bart_eval_predictions.csv"

# ----------------------------
# LOAD METRICS
# ----------------------------
wer = evaluate.load("wer")
cer = evaluate.load("cer")
bleu = evaluate.load("bleu")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Predicted Transcription", "True Transcription"])
inputs = df["Predicted Transcription"].tolist()
references = df["True Transcription"].tolist()

# ----------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------
tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# ----------------------------
# GENERATE PREDICTIONS
# ----------------------------
predictions = []
print("Generating predictions...")
for text in tqdm(inputs):
    inputs_encoded = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs_encoded, max_length=256, num_beams=4)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(decoded)

# ----------------------------
# CALCULATE METRICS
# ----------------------------
print("Calculating metrics...")
wer_score = wer.compute(predictions=predictions, references=references)
cer_score = cer.compute(predictions=predictions, references=references)
bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]

pred_embeds = embedder.encode(predictions, convert_to_tensor=True)
ref_embeds = embedder.encode(references, convert_to_tensor=True)
cos_sim = torch.nn.functional.cosine_similarity(pred_embeds, ref_embeds).mean().item()

print("\n--- Evaluation Summary ---")
print(f"WER: {wer_score:.4f}")
print(f"CER: {cer_score:.4f}")
print(f"BLEU: {bleu_score:.4f}")
print(f"Cosine Similarity: {cos_sim:.4f}")

# ----------------------------
# SAVE TO CSV
# ----------------------------
df_out = df.copy()
df_out["BART Corrected"] = predictions
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
