import os
import torch
from pathlib import Path
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers.integrations import NeptuneCallback
from matryoshka_experiment.models import (
    get_untrained_ff_model,
    toggle_freeze_linear_in_ff_model,
    push_sentence_transformers_model_to_hf,
)
from matryoshka_experiment.utils import get_revision, extract_available_checkpoints
from matryoshka_experiment.data import get_datasets


def get_training_args(output_dir, epochs=1, learning_rate=None):
    TARGET_BATCH_SIZE = 72
    BATCH_SIZE = 8

    accumulation_steps = TARGET_BATCH_SIZE / torch.cuda.device_count() / 8
    assert accumulation_steps % 1 == 0

    report_to = "none"
    if os.environ.get("RANK", 0) == 0:
        report_to = "neptune"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=5000,
        logging_steps=100,
        report_to=report_to,
        gradient_accumulation_steps=accumulation_steps,
        save_total_limit=8,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use validation loss to determine the best model
        greater_is_better=False,
        learning_rate=learning_rate,
        push_to_hub=False,
    )
    return training_args


def get_trainer(model, training_args):
    train_dataset, val_dataset, test_dataset = get_datasets()
    loss = losses.TripletLoss(model=model)
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=training_args,
    )
    return trainer


def train_baseline(embedding_size, version):
    assert os.environ["NEPTUNE_API_TOKEN"]
    assert os.environ["NEPTUNE_PROJECT"]

    repo_id = f"mim-nlp-project/ff-{embedding_size}"
    model = get_untrained_ff_model(embedding_size)

    # Let's save the untrained model
    push_sentence_transformers_model_to_hf(
        model, repo_id, branch=get_revision(0, version=version, type="pretraining")
    )

    # First let's train the linear layer only
    toggle_freeze_linear_in_ff_model(model, freeze=True)
    epochs = 0.1
    learning_rate = 0.1
    output_dir = f"{os.environ['OUTPUT_DIR']}/baseline/pretraining/{embedding_size}"
    training_args = get_training_args(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    trainer = get_trainer(model, training_args)
    run = NeptuneCallback.get_run(trainer)
    run["hyperparams"] = {
        "custom_model_type": "baseline",
        "custom_training_type": "pretraining",
        "embedding_size": embedding_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    trainer.train()

    # Save the checkpoints
    for checkpoint_step in extract_available_checkpoints(output_dir):
        path = str(Path(output_dir) / f"checkpoint-{checkpoint_step}")
        model = SentenceTransformer(path)
        branch = get_revision(checkpoint_step, version=version, type="pretraining")
        push_sentence_transformers_model_to_hf(model, repo_id, branch)

    # Now let's train the full model
    toggle_freeze_linear_in_ff_model(model, freeze=False)
    epochs = 1
    learning_rate = None
    output_dir = f"output_dir/baseline/finetuning/{embedding_size}"
    training_args = get_training_args(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    trainer = get_trainer(model, training_args)
    run = NeptuneCallback.get_run(trainer)
    run["hyperparams"] = {
        "custom_model_type": "baseline",
        "custom_training_type": "pretraining",
        "embedding_size": embedding_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    trainer.train()

    # Save the checkpoints
    for checkpoint_step in extract_available_checkpoints(output_dir):
        path = str(Path(output_dir) / f"checkpoint-{checkpoint_step}")
        model = SentenceTransformer(path)
        branch = get_revision(checkpoint_step, type="finetuning")
        push_sentence_transformers_model_to_hf(model, repo_id, branch)

    # TODO
    # evaluator = TripletEvaluator(
    #     anchors=test_dataset["anchor"],
    #     positives=test_dataset["positive"],
    #     negatives=test_dataset["negative"],
    #     name="medi-test-subset",
    #     batch_size=BATCH_SIZE,
    #     show_progress_bar=True,
    # )
    # results = evaluator(model)
    # print(f"test results: {results}")
    # with open(f'{MODEL_DIR}/test_metrics.json', 'w') as f:
    #     json.dump(results, f)