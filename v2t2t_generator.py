"""
v2t2t_generator.py

Description:
Unified V2T2T transcription pipeline for ATC ASR, combining Voice-to-Text and Text-to-Text correction.
Outputs CSV under Results/pipeline/ with Dataset, Sample, and Prediction columns.

Author: Everett-Alan Hood
"""

import os
import re
import pandas as pd
from datasets import load_dataset, Audio
from tqdm import tqdm
from cycle_pipeline.models.v2t2t.combinator import V2T2T

# Paths (adjusted for your structure)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_V2T = os.path.join(SCRIPT_DIR, "cycle_pipeline", "models", "v2t")
MODELS_T2T = os.path.join(SCRIPT_DIR, "cycle_pipeline", "models", "t2t")
PIPELINE_RESULTS = os.path.join(SCRIPT_DIR, "Results", "pipeline")
os.makedirs(PIPELINE_RESULTS, exist_ok=True)

SAMPLING_RATE = 16000

def list_models(directory):
    if not os.path.exists(directory):
        print(f"[ERROR] Directory {directory} does not exist. Please verify your folder structure.")
        exit(1)
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def prompt_selection(options, label):
    print(f"\nAvailable {label} Models:")
    for idx, option in enumerate(options, 1):
        print(f"{idx}: {option}")
    while True:
        try:
            selection = int(input(f"\nSelect {label} model by number: ")) - 1
            if 0 <= selection < len(options):
                return options[selection]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

def get_next_version(base_name):
    existing = [f for f in os.listdir(PIPELINE_RESULTS) if f.startswith(base_name)]
    versions = [int(re.search(r'v(\\d+)', f).group(1)) for f in existing if re.search(r'v(\\d+)', f)]
    return max(versions, default=0) + 1

def run_pipeline(v2t_name, t2t_name):
    print(f"\n[INFO] Running unified V2T2T pipeline: {v2t_name} + {t2t_name}")

    v2t_path = os.path.join(MODELS_V2T, v2t_name)
    t2t_path = os.path.join(MODELS_T2T, t2t_name)

    v2t2t = V2T2T(v2t_path, t2t_path)

    dataset_names = ["Jzuluaga/atcosim_corpus", "Jzuluaga/uwb_atcc"]
    results = []

    for name in dataset_names:
        ds_name = name.split("/")[-1]
        dataset = load_dataset(name, split="test").cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

        for i, sample in enumerate(tqdm(dataset, desc=f"Processing {ds_name}")):
            corrected = v2t2t.transcribe(sample["audio"]["array"])
            results.append({
                "Dataset": ds_name,
                "Sample": f"{ds_name}/sample_{i}.wav",
                "Prediction": corrected
            })

    df = pd.DataFrame(results)
    filename_base = f"{v2t_name}-AND-{t2t_name}"
    version = get_next_version(filename_base)
    final_name = f"{filename_base}-v{version}.csv"
    output_path = os.path.join(PIPELINE_RESULTS, final_name)

    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved pipeline output to {output_path}")

if __name__ == "__main__":
    v2t_models = list_models(MODELS_V2T)
    t2t_models = list_models(MODELS_T2T)

    if not v2t_models or not t2t_models:
        print("[ERROR] Required models not found.")
        exit(1)

    selected_v2t = prompt_selection(v2t_models, "V2T")
    selected_t2t = prompt_selection(t2t_models, "T2T")

    run_pipeline(selected_v2t, selected_t2t)
