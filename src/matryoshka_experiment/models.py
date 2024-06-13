from matryoshka_experiment.training_models.modules import create_sentence_transformer
from sentence_transformers import SentenceTransformer

from matryoshka_experiment.utils import (
    push_sentence_transformers_model_to_hf,
    get_revision,
)


#
# FF model
#


def get_untrained_ff_model(embedding_size):
    model = create_sentence_transformer(
        "distilbert/distilroberta-base", output_dim=embedding_size
    )
    return model


def get_trained_ff_model(embedding_size, version, checkpoint_step=18000):
    repo_id = f"mim-nlp-project/ff-{embedding_size}"
    revision = get_revision(checkpoint_step, version)
    model = SentenceTransformer(repo_id, revision=revision)
    return model


def push_trained_ff_model(model, embedding_size, checkpoint_step, version):
    repo_id = f"mim-nlp-project/ff-{embedding_size}"
    revision = get_revision(checkpoint_step, version)
    push_sentence_transformers_model_to_hf(model, repo_id, revision)


def toggle_freeze_linear_in_ff_model(model, freeze=True):
    for name, param in model.named_parameters():
        if "2.linear_layers" not in name:
            param.requires_grad = not freeze


#
# Matryoshka model
#

# TODO
