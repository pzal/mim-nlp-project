import os
import accelerate
from dotenv import load_dotenv

from datasets import load_from_disk, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from transformers.integrations import NeptuneCallback
import neptune
import json

from matryoshka_experiment.training_models.modules import create_sentence_transformer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

load_dotenv(".public_env")
load_dotenv(".env")

BATCH_SIZE = 8
EPOCHS = 1

ARRAY_INDEX_TO_EMB_SIZE = {
    0: 64,
    1: 128,
    2: 256,
    3: 768}


def main():

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    job_array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    emb_size = ARRAY_INDEX_TO_EMB_SIZE[job_array_index]
    MODEL_DIR = f"output_dir/baseline_{emb_size}"
    print("Training baseline model with embedding size: ", emb_size)

    model = create_sentence_transformer("distilbert/distilroberta-base", output_dim=emb_size)

    dataset = load_dataset("mim-nlp-project/medi-joined")
    print(f"RANK: {os.environ['RANK']}")
    print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")

    train_dataset = dataset["train"].select_columns(["anchor", "positive", "negative"])
    val_dataset = dataset["val"].select_columns(["anchor", "positive", "negative"])
    test_dataset = dataset["test"].select_columns(["anchor", "positive", "negative"])

    if int(os.environ["RANK"]) == 0:
        run = neptune.init_run()
        run["embedding_size"] = emb_size
        neptune_callback = NeptuneCallback(run=run)
        id_of_run = run["sys/id"].fetch()

        
    loss = losses.TripletLoss(model=model)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=MODEL_DIR,
        # Optional training parameters:
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500, # original maybe 2500
        save_strategy="steps",
        save_steps=5000,  # in final scale run maybe about 5000
        logging_steps=100,
        # report_to="neptune",
        gradient_accumulation_steps=3,
        save_total_limit=7,                    # Limit the total amount of checkpoints
        load_best_model_at_end=True,           # Load the best model when finished training
        metric_for_best_model="eval_loss",     # Use validation loss to determine the best model
        greater_is_better=False,               # Lower validation loss is better
        push_to_hub=True,                      # Push the model to Hugging Face Hub
        hub_model_id=f"testing-baseline-{emb_size}",        # Model ID for the Hugging Face Hub
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=args,
        callbacks=[neptune_callback] if int(os.environ["RANK"]) == 0 else None,
    )

    trainer.train()
    
    evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="medi-test-subset",
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
    results = evaluator(model)
    print(f"test results: {results}")
    with open(f'{MODEL_DIR}/test_metrics.json', 'w') as f:
        json.dump(results, f)





if __name__ == "__main__":
    main()
