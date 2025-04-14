from datasets import load_dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ATCoSim corpus
atcosim = load_dataset("Jzuluaga/atcosim_corpus", split="test")

# Load UWB-ATCC corpus
uwb_atcc = load_dataset("Jzuluaga/uwb_atcc", split="test")

# Collecting all models in wav2vec_models
MODELS = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'wav2vec_models')
for root, dir, files in os.walk(MODELS_DIR):
    if any(file.endswith(".json") for file in files) and "language_model" not in root:
        MODELS.append(root)

print(MODELS)

# Load models and their processors (instead of tokenizers for audio data)
models = {name: AutoModelForCTC.from_pretrained(name).to(device) for name in MODELS}
processors = {name: Wav2Vec2Processor.from_pretrained(name) for name in MODELS}

def generate_predictions(audio, model, processor):
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    return transcription

# Initialize evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")  # For character error rate
bleu_metric = evaluate.load("bleu")

# Function to calculate cosine similarity
def calculate_cosine_similarity(predictions, references):
    vectorizer = TfidfVectorizer().fit(predictions + references)
    pred_vectors = vectorizer.transform(predictions).toarray()
    ref_vectors = vectorizer.transform(references).toarray()
    cosine_similarities = cosine_similarity(pred_vectors, ref_vectors).diagonal()
    return cosine_similarities.mean()

results = []

print(atcosim.column_names)
print(uwb_atcc.column_names)

def evaluate_dataset(dataset, dataset_name, models, processors):
    for data in dataset:
        true_text = data['text']
        for model_name, model in models.items():
            processor = processors[model_name]
            pred_text = generate_predictions(data['audio'], model, processor)
            
            # Calculate individual metrics
            wer = wer_metric.compute(predictions=[pred_text], references=[true_text])
            cer = cer_metric.compute(predictions=[pred_text], references=[true_text])
            # Safely calculate BLEU only if both texts are non-empty
            bleu = 0.0  # default BLEU to 0 in case of empty reference or prediction
            if pred_text and true_text:
                bleu = bleu_metric.compute(predictions=[pred_text], references=[true_text])['bleu']
            cosine_sim = calculate_cosine_similarity([pred_text], [true_text])

            # Append results for this sample
            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "True Transcription": true_text,
                "Predicted Transcription": pred_text,
                "WER": wer,
                "CER": cer,
                "BLEU": bleu,
                "Cosine Similarity": cosine_sim,
            })
            print(pred_text, "\n", true_text, "\n")

# Evaluate both datasets
evaluate_dataset(atcosim, "atcosim", models, processors)
evaluate_dataset(uwb_atcc, "uwb_atcc", models, processors)

# Convert results to DataFrame and save as CSV
df = pd.DataFrame(results)
df.to_csv("llm_evaluation_summary.csv", index=False)

print("Evaluation completed and saved to llm_evaluation_summary.csv.")
