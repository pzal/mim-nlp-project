import os
from typing import List
from pathlib import Path
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainer,
)
from transformers.integrations import NeptuneCallback
from matryoshka_experiment.models import (
    get_untrained_mrl_model,
)
from matryoshka_experiment.utils import (
    get_revision,
    push_sentence_transformers_model_to_hf,
)
from matryoshka_experiment.data import get_datasets
from matryoshka_experiment.training.common import (
    get_training_args,
    extract_available_checkpoints,
)


def get_trainer(model, training_args):
    train_dataset, val_dataset, test_dataset = get_datasets()
    base_loss = losses.TripletLoss(model=model)
    loss = losses.MatryoshkaLoss(
        model=model, loss=base_loss, matryoshka_dims=[768, 384, 192, 96, 48]
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=training_args,
    )
    return trainer


def train_matryoshka(
    version,
    batch_size_per_gpu,
    tags: List[str] = None,
    initial_model_revision=None,
):
    assert os.environ["NEPTUNE_API_TOKEN"]
    assert os.environ["NEPTUNE_PROJECT"]
    assert os.environ["OUTPUT_DIR"]

    repo_id = "mim-nlp-project/mrl"

    if initial_model_revision:
        model = SentenceTransformer(repo_id, revision=initial_model_revision)
    else:
        model = get_untrained_mrl_model()

    # Let's save the untrained model
    push_sentence_transformers_model_to_hf(
        model, repo_id, branch=get_revision(0, version=version, type="finetuning")
    )

    epochs = 1.0
    learning_rate = None
    output_dir = f"{os.environ['OUTPUT_DIR']}/matryoshka/finetuning"
    training_args = get_training_args(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size_per_gpu=batch_size_per_gpu,
    )
    trainer = get_trainer(model, training_args)
    run = NeptuneCallback.get_run(trainer)
    run["hyperparams"] = {
        "model_type": "matryoshka",
        "training_type": "finetuning",
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    if tags:
        run["sys/tags"].add(tags)
    trainer.train()

    # Save the checkpoints
    for checkpoint_step in extract_available_checkpoints(output_dir):
        path = str(Path(output_dir) / f"checkpoint-{checkpoint_step}")
        _model = SentenceTransformer(path)
        branch = get_revision(checkpoint_step, version=version, type="finetuning")
        push_sentence_transformers_model_to_hf(_model, repo_id, branch)

    # Push the final model of finetuning
    push_sentence_transformers_model_to_hf(
        model, repo_id, branch=f"{version}-finetuning-final"
    )
