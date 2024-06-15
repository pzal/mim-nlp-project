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
    get_untrained_ff_model,
    toggle_freeze_other_layers_in_ff_model,
)
from matryoshka_experiment.utils import get_revision, push_sentence_transformers_model_to_hf
from matryoshka_experiment.data import get_datasets
from matryoshka_experiment.training.common import (
    get_training_args,
    extract_available_checkpoints,
)


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


def train_baseline(
    embedding_size,
    version,
    batch_size_per_gpu,
    tags: List[str] = None,
    load_pretrained=False,
):
    assert os.environ["NEPTUNE_API_TOKEN"]
    assert os.environ["NEPTUNE_PROJECT"]
    assert os.environ["OUTPUT_DIR"]

    repo_id = f"mim-nlp-project/ff-{embedding_size}"

    if embedding_size == 768:
        # There's no untrained Linear in this case.
        pass
    elif load_pretrained:
        model = SentenceTransformer(repo_id, revision=f"{version}-pretraining-final")
    else:
        model = get_untrained_ff_model(embedding_size)

        # Let's save the untrained model
        push_sentence_transformers_model_to_hf(
            model, repo_id, branch=get_revision(0, version=version, type="pretraining")
        )

        # First let's train the linear layer only
        toggle_freeze_other_layers_in_ff_model(model, freeze=True)
        epochs = 0.2
        learning_rate = 0.02
        output_dir = f"{os.environ['OUTPUT_DIR']}/baseline/pretraining/{embedding_size}"
        training_args = get_training_args(
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size_per_gpu=72,  # Here we have very few params
        )
        trainer = get_trainer(model, training_args)
        run = NeptuneCallback.get_run(trainer)
        run["hyperparams"] = {
            "model_type": "baseline",
            "training_type": "pretraining",
            "embedding_size": embedding_size,
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
            branch = get_revision(checkpoint_step, version=version, type="pretraining")
            push_sentence_transformers_model_to_hf(_model, repo_id, branch)

        # Push the final model of pretraining
        push_sentence_transformers_model_to_hf(
            model, repo_id, branch=f"{version}-pretraining-final"
        )

    # Now let's train the full model
    if embedding_size != 768:
        toggle_freeze_other_layers_in_ff_model(model, freeze=False)
    else:
        model = get_untrained_ff_model(embedding_size)
    epochs = 1.0
    learning_rate = None
    output_dir = f"{os.environ['OUTPUT_DIR']}/baseline/finetuning/{embedding_size}"
    training_args = get_training_args(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size_per_gpu=batch_size_per_gpu,
    )
    trainer = get_trainer(model, training_args)
    run = NeptuneCallback.get_run(trainer)
    run["hyperparams"] = {
        "model_type": "baseline",
        "training_type": "finetuning",
        "embedding_size": embedding_size,
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
