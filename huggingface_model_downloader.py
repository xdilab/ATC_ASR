from huggingface_hub import snapshot_download
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer
import os

# Define the model name and target directory
model_name = "youngsangroh/whisper-large-finetuned-atcosim_corpus"
target_dir = r"C:\Users\tim3l\OneDrive\Desktop"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download the model and tokenizer
model = Wav2Vec2Model.from_pretrained(model_name)
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the target directory
model.save_pretrained(target_dir)
tokenizer.save_pretrained(target_dir)

snapshot_download(repo_id=model_name, local_dir=target_dir)

print(f"Model and tokenizer downloaded to {target_dir}")