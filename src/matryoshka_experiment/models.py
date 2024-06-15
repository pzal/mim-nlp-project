from matryoshka_experiment.training_models.modules import create_sentence_transformer


#
# FF model
#


def get_untrained_ff_model(embedding_size):
    model = create_sentence_transformer(
        "distilbert/distilroberta-base", output_dim=embedding_size
    )
    return model


def toggle_freeze_other_layers_in_ff_model(model, freeze=True):
    for name, param in model.named_parameters():
        if "2.linear_layers" not in name:
            param.requires_grad = not freeze


#
# Matryoshka model
#


def get_untrained_mrl_model():
    model = create_sentence_transformer("distilbert/distilroberta-base", output_dim=768)
    return model
