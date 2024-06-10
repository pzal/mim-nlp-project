import os
import accelerate
from dotenv import load_dotenv

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers.integrations import NeptuneCallback

from training_models.modules import create_sentence_transformer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

load_dotenv(".public_env")
load_dotenv(".env")

BATCH_SIZE = 8
EPOCHS = 1

def main():

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    model = create_sentence_transformer("distilbert/distilroberta-base", output_dim=64)

    dataset = load_from_disk("data/medi/medi.arrow")

    train_dataset = dataset['train'].select(range(10000)).select_columns(["anchor", "positive", "negative"])
    val_dataset = dataset['val'].select(range(1000)).select_columns(["anchor", "positive", "negative"])

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
    run = NeptuneCallback.get_run(trainer)
    run["slurm_output"].upload_files("slurm_logs")

    trainer.train()
    run["slurm_output"].upload_files("slurm_logs")
    
if __name__ == "__main__":
    main()
