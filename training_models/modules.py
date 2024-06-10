from typing import Dict, List

import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from torch import Tensor


def create_sentence_transformer(model_name_or_path: str, output_dim: int, **kwargs):
    """
    Creates a SentenceTransformer model with a linear layer to adapt the output dimension. When asked dimension
    is the same as the base model, the linear layer is not added.
    :param model_name_or_path: same as in SentenceTransformer constructor
    :param output_dim: the dimension of the output embeddings
    :param kwargs: kwargs passed to SentenceTransformer constructor
    :return: SentenceTransformer model
    """
    base = Transformer(model_name_or_path)
    pooling = Pooling(base.get_word_embedding_dimension(), "mean")
    modules = [base, pooling]
    if output_dim != base.get_word_embedding_dimension():
        modules.append(LinearEmbeddingAdapter(base.get_word_embedding_dimension(), output_dim))
    sentence_transformer = SentenceTransformer(modules=modules, **kwargs)
    return sentence_transformer


class LinearEmbeddingAdapter(nn.Module):
    """
    A simple linear layer to adapt the output dimension of the embeddings. It is adapted to output given by the
    SentenceTransformer model, i.e a dictionary with tensor values. Linear mapping is applied only to the values
    with the names given in the input_names_to_convert list, by default to "sentence_embedding" and "token_embeddings".

    The hope is that this adapter will work gracefully with SentenceTransformerTrainer...
    """

    def __init__(self, original_dim: int, output_dim: int, input_names_to_convert: List[str] = None):
        super(LinearEmbeddingAdapter, self).__init__()
        if input_names_to_convert is None:
            input_names_to_convert = ["sentence_embedding", "token_embeddings"]

        self.linear_layers = {name: nn.Linear(original_dim, output_dim) for name in input_names_to_convert}

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = {}
        for name, values in input.items():
            if name in self.linear_layers:
                output[name] = self.linear_layers[name](values)
            else:
                output[name] = values
        return output
