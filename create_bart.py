import os
import torch
import pandas as pd
import numpy as np
import random
import evaluate
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from sentence_transformers import SentenceTransformer

# Enable CUDA memory growth and disable tokenizer parallelism
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat32 on CUDA

# Load a global sentence embedding model for cosine similarity metric
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file_path: str):
    """
    Load CSV data in chunks, drop rows with missing transcriptions,
    and return a cleaned DataFrame.
    """
    chunks = pd.read_csv(file_path, chunksize=10000)
    df = pd.concat(
        [chunk.dropna(subset=["Predicted Transcription", "True Transcription"]) for chunk in chunks],
        ignore_index=True
    )
    print(f"Loaded {len(df)} samples from {file_path}")
    return df

def preprocess_function(examples, tokenizer, max_length=128):
    """
    Tokenize input and target texts with an increased max_length.
    The labels are tokenized using the text_target argument to avoid the deprecated context.
    """
    model_inputs = tokenizer(
        examples["Predicted Transcription"],
        truncation=True,
        max_length=max_length
    )
    labels = tokenizer(
        text_target=examples["True Transcription"],
        truncation=True,
        max_length=max_length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    """
    Compute evaluation metrics: WER, CER, BLEU, and Cosine Similarity.
    BLEU: Uses decoded strings directly.
    Cosine Similarity: Uses sentence embeddings to compute similarity.
    """
    preds, labels = eval_preds

    preds = np.array(preds)
    labels = np.array(labels)
    
    # Replace -100 with pad token id for decoding
    labels[labels == -100] = tokenizer.pad_token_id
    preds[preds == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_metric = evaluate.load("wer")
    wer_value = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Compute CER
    cer_metric = evaluate.load("cer")
    cer_value = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Compute BLEU using decoded strings directly
    bleu_metric = evaluate.load("bleu")
    bleu_value = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    
    # Compute Cosine Similarity using sentence embeddings
    embeddings_preds = embedding_model.encode(decoded_preds, convert_to_tensor=True)
    embeddings_labels = embedding_model.encode(decoded_labels, convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings_preds, embeddings_labels)
    cosine_similarity_value = cos_sim.mean().item()

    return {
        "wer": wer_value,
        "cer": cer_value,
        "bleu": bleu_value["bleu"],
        "cosine_similarity": cosine_similarity_value
    }

def main():
    # Read dataset and split
    csv_path = "llm_evaluation_summary.csv"
    print("Loading dataset...")
    df = load_data(csv_path)
    dataset = Dataset.from_pandas(df).shuffle(seed=seed)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # Load model and tokenizer
    model_name = "facebook/bart-large"
    print("Loading model and tokenizer...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Update model generation configuration for improved output quality
    model.config.num_beams = 4  # Use beam search for better generations
    model.config.max_length = 128

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Enable gradient checkpointing if using a single GPU and disable caching for memory efficiency
    if torch.cuda.device_count() <= 1:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Use DataCollatorForSeq2Seq for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt"
    )

    # Optimized training arguments with adjusted hyperparameters and generation settings
    training_args = Seq2SeqTrainingArguments(
        output_dir="bart_model_checkpoints",
        num_train_epochs=7,                 # Increased epochs for better fine-tuning
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        fp16=torch.cuda.is_available(),     # Enable mixed precision for performance
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,                 # Lower learning rate for finer updates
        warmup_steps=500,
        label_smoothing_factor=0.1,         # Added label smoothing for better generalization
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        logging_steps=250,
        predict_with_generate=True,
        report_to="tensorboard",
        dataloader_num_workers=4,
        eval_accumulation_steps=4,
        generation_num_beams=4,             # Improve generation quality with beam search
        generation_max_length=128
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,  # Updated to use processing_class as required
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and tokenizer
    save_dir = "bart_finetuned_model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
