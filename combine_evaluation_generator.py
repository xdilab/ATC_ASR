import pandas as pd
import os

# === FILE PATHS ===
base_dir = r"C:\Users\tim3l\OneDrive\Desktop\Local_Wav2Vec"
llm_eval_path = os.path.join(base_dir, "llm_evaluation_summary.csv")
local_eval_path = os.path.join(base_dir, "local_evaluation_summary.csv")
output_path = os.path.join(base_dir, "complete_eval.csv")

# === LOAD CSVs ===
df_llm = pd.read_csv(llm_eval_path)
df_manual = pd.read_csv(local_eval_path)

# === Add Dataset name to manual transcription data ===
df_manual["Dataset"] = "Manual"

# === Normalize Columns (ensure consistency) ===
columns_needed = ["Dataset", "Model", "True Transcription", "Predicted Transcription", "WER", "CER", "BLEU", "Cosine Similarity"]
df_llm = df_llm[columns_needed]
df_manual = df_manual[columns_needed]

# === Shorten Model Names ===
def shorten_model(path_or_name):
    parts = str(path_or_name).replace("/", os.sep).split(os.sep)
    return os.path.join(*parts[-2:]) if len(parts) >= 2 else path_or_name

df_llm["Model"] = df_llm["Model"].apply(shorten_model)
df_manual["Model"] = df_manual["Model"].apply(shorten_model)

# === Combine and Save ===
combined_df = pd.concat([df_llm, df_manual], ignore_index=True)
combined_df.to_csv(output_path, index=False)

print(f"✅ Combined evaluation saved to: {output_path}")
