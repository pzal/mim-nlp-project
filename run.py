import os
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
import math
from datasets import load_from_disk, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import torch

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

ENV = dotenv_values(".public_env")
ENV.update(dotenv_values(".env"))

os.environ["NEPTUNE_API_TOKEN"] = ENV["NEPTUNE_API_TOKEN"]
os.environ["NEPTUNE_PROJECT"] = ENV["NEPTUNE_PROJECT"]

TARGET_BATCH_SIZE = 64
BATCH_SIZE_PER_GPU = 8
N_GPUS = torch.cuda.device_count()
EPOCHS = 1
push_to_hub_organization = "mim-nlp-project"
model_id = "ff-768"
output_dir = f"output/{model_id}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

model = SentenceTransformer("distilbert/distilroberta-base", device="cuda")

dataset = load_dataset("mim-nlp-project/medi-joined")

train_dataset = (
    dataset["train"]
    # .select(range(10000))
    .select_columns(["anchor", "positive", "negative"])
)
val_dataset = (
    dataset["val"]
    # .select(range(1000))
    .select_columns(["anchor", "positive", "negative"])
)

loss = losses.MultipleNegativesRankingLoss(model=model)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    dataloader_num_workers=8,
    seed=42,
    per_device_train_batch_size=BATCH_SIZE_PER_GPU,
    per_device_eval_batch_size=BATCH_SIZE_PER_GPU,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=20,
    report_to="neptune",
    gradient_accumulation_steps=math.ceil(
        TARGET_BATCH_SIZE / BATCH_SIZE_PER_GPU / N_GPUS
    ),
    push_to_hub_model_id=model_id,
    push_to_hub_organization=push_to_hub_organization,
    push_to_hub=True,
    hub_private_repo=True,
)


trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    args=args,
)

trainer.train()
