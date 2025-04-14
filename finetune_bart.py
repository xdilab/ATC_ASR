import os
import pandas as pd
import numpy as np
from transformers import (BartTokenizer, BartForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
import evaluate
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "bart_finetuned_model"
CHECKPOINT_DIR = "bart_model_checkpoints"
CSV_PATH = "/home/elhood/Desktop/fine_tuning/wav2vec2_finetuned_bart_atcosim_detailed.csv"

# ----------------------------
# METRICS
# ----------------------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("bleu")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# LOAD & PREP DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["True Transcription", "Predicted Transcription"])

def make_dataset(df):
    return Dataset.from_pandas(df[["Predicted Transcription", "True Transcription"]].rename(columns={
        "Predicted Transcription": "input_text",
        "True Transcription": "target_text"
    }))

dataset = make_dataset(df)

# ----------------------------
# TOKENIZER & MODEL
# ----------------------------
tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.config.num_beams = 4
model.config.max_length = 256
if torch.cuda.device_count() <= 1:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

# ----------------------------
# TOKENIZATION
# ----------------------------
def tokenize(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=256, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"], max_length=256, padding="max_length", truncation=True)["input_ids"]
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
    ]
    return model_inputs

dataset = dataset.map(tokenize, batched=True)

# ----------------------------
# TRAINING SETUP
# ----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", return_tensors="pt")

training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=250,
    predict_with_generate=True,
    report_to="tensorboard",
    dataloader_num_workers=4,
    generation_num_beams=4,
    generation_max_length=256
)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer_value = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    cer_value = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_value = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    emb_pred = embedding_model.encode(decoded_preds, convert_to_tensor=True)
    emb_label = embedding_model.encode(decoded_labels, convert_to_tensor=True)
    cosine_sim = torch.nn.functional.cosine_similarity(emb_pred, emb_label).mean().item()

    return {
        "wer": wer_value,
        "cer": cer_value,
        "bleu": bleu_value["bleu"],
        "cosine_similarity": cosine_sim
    }

trainer = Seq2SeqTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("Final fine-tuning complete and model saved.")
