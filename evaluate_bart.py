import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from jiwer import wer, cer
from sacrebleu import corpus_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import torch
from tqdm import tqdm

# Ensure NLTK dependencies are downloaded (if needed)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def count_outliers(df):
    """
    Count outliers for each metric based on defined valid ranges:
    - BLEU: [0, 100]
    - WER: [0, 1]
    - CER: [0, 1]
    - Cosine Similarity: [0, 1]
    
    Returns a dictionary with counts per metric and a total outlier count.
    """
    outlier_conditions = {
        "BLEU": ~df["BLEU"].between(0, 100),
        "WER": ~df["WER"].between(0, 1),
        "CER": ~df["CER"].between(0, 1),
        "Cosine Similarity": ~df["Cosine Similarity"].between(0, 1)
    }

    outlier_counts = {metric: cond.sum() for metric, cond in outlier_conditions.items()}
    any_outlier_condition = np.logical_or.reduce(list(outlier_conditions.values()))
    total_outliers = any_outlier_condition.sum()
    outlier_counts["Total Outliers"] = total_outliers
    return outlier_counts


def get_max_length_from_dir(dir_name, default=64):
    """
    Extract max_length from a directory naming convention if it exists.
    For example, 'my_bart_corrector_ml128' => max_length = 128.
    Otherwise, returns the default.
    """
    if '_ml' in dir_name:
        try:
            max_length_str = dir_name.split('_ml')[-1]
            return int(max_length_str)
        except ValueError:
            return default
    return default


if __name__ == '__main__':
    # Path to your CSV file
    file_path = 'llm_evaluation_summary.csv'
    
    print("Loading test data...")
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['True Transcription', 'Predicted Transcription']).reset_index(drop=True)

    references = data["True Transcription"].tolist()
    original_predictions = data["Predicted Transcription"].tolist()

    # Precompute TF-IDF vectors for Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectorizer.fit(references + original_predictions)
    reference_vectors = vectorizer.transform(references)

    # Look for directories starting with "bart_finetuned_model"
    model_dirs = [
        d for d in os.listdir('.') 
        if os.path.isdir(d) and d.startswith('bart_finetuned_model')
    ]
    if not model_dirs:
        print("No BART model directories found matching 'bart_finetuned_model*'. Exiting.")
        exit()

    for model_dir in model_dirs:
        print(f"\nEvaluating BART model in: {model_dir}")

        # Load tokenizer and model
        tokenizer = BartTokenizer.from_pretrained(model_dir)
        model = BartForConditionalGeneration.from_pretrained(model_dir)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Determine max_length from directory name (default=128)
        max_length = get_max_length_from_dir(model_dir, default=128)
        print(f"Using max_length={max_length} for generation...")

        bart_predictions = []
        print("Generating corrected predictions...")
        for pred_text in tqdm(original_predictions, desc=f"Generating from {model_dir}", leave=True):
            inputs = tokenizer(
                pred_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_length)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bart_predictions.append(corrected_text)

        print("Computing metrics for each sample...")
        results_per_sample = []
        for i, (ref, bart_pred) in tqdm(
            enumerate(zip(references, bart_predictions)), 
            total=len(bart_predictions), 
            desc=f"Metrics for {model_dir}",
            leave=True
        ):
            # Compute WER and CER
            sample_wer = wer(ref, bart_pred)
            sample_cer = cer(ref, bart_pred)
            
            # Compute BLEU using sacrebleu
            sample_bleu = corpus_bleu([bart_pred], [[ref]]).score
            
            # Compute Cosine Similarity using precomputed TF-IDF vectors
            ref_vector = reference_vectors[i]
            bart_vector = vectorizer.transform([bart_pred])
            sample_cosine = cosine_similarity(ref_vector, bart_vector)[0][0]

            results_per_sample.append({
                "Model": model_dir,
                "True Transcription": ref,
                "Original Prediction": original_predictions[i],
                "BART Corrected": bart_pred,
                "WER": sample_wer,
                "CER": sample_cer,
                "BLEU": sample_bleu,
                "Cosine Similarity": sample_cosine
            })

        detailed_results_df = pd.DataFrame(results_per_sample)

        # Save per-sample evaluation results
        output_file = f'{model_dir}_detailed_evaluation.csv'
        print(f"Saving per-sample evaluation results to {output_file}...")
        detailed_results_df.to_csv(output_file, index=False)

        # Compute mean & median metrics on all data (including outliers)
        metrics = ["WER", "CER", "BLEU", "Cosine Similarity"]
        mean_metrics = detailed_results_df[metrics].mean()
        median_metrics = detailed_results_df[metrics].median()

        summary_df = pd.DataFrame({
            "Metric": metrics,
            "Mean (All Data)": mean_metrics.values,
            "Median (All Data)": median_metrics.values
        })

        summary_file = f'{model_dir}_summary_evaluation.csv'
        print(f"Saving summary (including outliers) results to {summary_file}...")
        summary_df.to_csv(summary_file, index=False)

        # Count and save outliers
        outlier_counts = count_outliers(detailed_results_df)
        outlier_counts_df = pd.DataFrame(
            list(outlier_counts.items()),
            columns=["Metric", "Outlier Count"]
        )
        outlier_counts_file = f'{model_dir}_outlier_counts.csv'
        print(f"Saving outlier counts to {outlier_counts_file}...")
        outlier_counts_df.to_csv(outlier_counts_file, index=False)

    print("\nBART Evaluation complete and saved.")
