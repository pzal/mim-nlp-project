import logging
import sys
import os
import traceback
import accelerate
from datetime import datetime
from dotenv import load_dotenv

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


load_dotenv(".public_env")
load_dotenv(".env")

BATCH_SIZE = 16
EPOCHS = 1


model = SentenceTransformer("distilbert/distilroberta-base")

dataset = load_from_disk("data/medi/medi.arrow")

train_dataset = dataset['train'].select(range(1000))
val_dataset = dataset['val'].select(range(1000))

loss = losses.TripletLoss(model=model)
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="output_dir",
    # Optional training parameters:
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    report_to="neptune"
)


trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    args=args,
)

trainer.train()
