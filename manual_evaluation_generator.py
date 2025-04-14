from datasets import Dataset, Audio
from transformers import AutoModelForCTC, Wav2Vec2Processor
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate
import os
import torch
from datetime import datetime
import re
import soundfile as sf

# === CONFIGURATION ===
base_dir = r"C:\Users\tim3l\OneDrive\Desktop\Local_Wav2Vec"
compiled_xlsx = os.path.join(base_dir, "compiled_transcriptions.xlsx")
snippets_dir = os.path.join(base_dir, "Snippets")
output_csv = os.path.join(base_dir, "local_evaluation_summary.csv")
models_dir = os.path.join(base_dir, "wav2vec_models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Text Cleaning ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip()

# === Time Formatter: Fully padded HH_MM_SS ===
def time_to_snippet_format(t):
    try:
        dt = datetime.strptime(t.strip(), "%H:%M:%S")
        return f"{dt.hour:02}_{dt.minute:02}_{dt.second:02}"  # e.g., 00_05_56
    except Exception as e:
        print(f"❌ Time parsing failed for '{t}': {e}")
        return "00_00_00"

def build_snippet_filename(row):
    base = row["Filename"].replace(".wav", "Z")
    start = time_to_snippet_format(row["Start"])
    end = time_to_snippet_format(row["End"])
    return f"{base}-{start}-{end}.wav"

# === Load Spreadsheet ===
df = pd.read_excel(compiled_xlsx)
df = df[df["Transcription"].notnull() & df["Transcription"].astype(str).str.strip().ne("")]
df["snippet_name"] = df.apply(build_snippet_filename, axis=1)
df["audio"] = df["snippet_name"].apply(lambda f: os.path.join(snippets_dir, f))
df["text"] = df["Transcription"].astype(str).fillna("").str.strip().replace(r"\n", " ", regex=True)
df["Start"] = df["Start"].astype(str)
df["End"] = df["End"].astype(str)
df["Filename"] = df["Filename"].astype(str)
df["Transcription"] = df["Transcription"].astype(str)

# === Check for Missing/Unreadable Files ===
missing = []
unreadable = []
valid_rows = []

print("\n🔍 Checking audio files...")

for i, row in df.iterrows():
    path = row["audio"]
    if not os.path.exists(path):
        missing.append(path)
        print(f"❌ MISSING: {os.path.basename(path)}")
        continue
    try:
        sf.read(path)
        valid_rows.append(row)
    except Exception:
        unreadable.append(path)
        print(f"⚠️ UNREADABLE: {os.path.basename(path)}")

print(f"\n🧼 TOTAL spreadsheet entries: {len(df)}")
print(f"❌ Missing audio files: {len(missing)}")
print(f"⚠️ Unreadable audio files: {len(unreadable)}")
print(f"✅ Usable samples: {len(valid_rows)}")

# Keep only valid entries
df_valid = pd.DataFrame(valid_rows)

# === Convert to Dataset ===
dataset = Dataset.from_pandas(df_valid)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# === Load Models ===
MODELS = []
for root, dirs, files in os.walk(models_dir):
    if any(file.endswith(".json") for file in files) and "language_model" not in root:
        MODELS.append(root)

print("\n🧠 Models found:")
for model_path in MODELS:
    print(f"  ✅ {model_path}")

models = {name: AutoModelForCTC.from_pretrained(name).to(device) for name in MODELS}
processors = {name: Wav2Vec2Processor.from_pretrained(name) for name in MODELS}

# === Metrics ===
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("bleu")

def generate_predictions(audio, model, processor):
    try:
        inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(pred_ids)[0]
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return "[UNRECOGNIZED]"

def calculate_cosine_similarity(preds, refs):
    vectorizer = TfidfVectorizer().fit(preds + refs)
    pred_vecs = vectorizer.transform(preds).toarray()
    ref_vecs = vectorizer.transform(refs).toarray()
    return cosine_similarity(pred_vecs, ref_vecs).diagonal().mean()

# === Run Evaluation ===
results = []

for entry in dataset:
    true_text_raw = str(entry.get("text", "")).strip().replace("\n", " ")
    true_text = clean_text(true_text_raw)
    if not true_text:
        continue

    for model_name, model in models.items():
        processor = processors[model_name]
        try:
            pred_text_raw = generate_predictions(entry["audio"], model, processor)
            pred_text = clean_text(pred_text_raw)
            if not pred_text:
                pred_text = "[UNRECOGNIZED]"

            wer = wer_metric.compute(predictions=[pred_text], references=[true_text])
            cer = cer_metric.compute(predictions=[pred_text], references=[true_text])
            bleu = bleu_metric.compute(predictions=[pred_text], references=[true_text])["bleu"]
            cosine_sim = calculate_cosine_similarity([pred_text], [true_text])

            results.append({
                "Dataset": "Manual",
                "Model": model_name,
                "True Transcription": true_text,
                "Predicted Transcription": pred_text,
                "WER": float(wer),
                "CER": float(cer),
                "BLEU": float(bleu),
                "Cosine Similarity": float(cosine_sim)
            })

            print(f"[{model_name}]")
            print("PRED:", pred_text)
            print("TRUE:", true_text)
            print("-" * 60)

        except Exception as e:
            print(f"❌ Skipped sample due to error with model {model_name}: {e}")

# === Save Results ===
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"\n✅ Evaluation complete. Results saved to:\n{output_csv}")
                                                                                                                                              