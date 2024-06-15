import json
import os
from typing import List, Dict, Any

import torch
from torch import nn as nn, Tensor


class LinearEmbeddingAdapter(nn.Module):
    """
    A simple linear layer to adapt the output dimension of the embeddings. It is adapted to output given by the
    SentenceTransformer model, i.e a dictionary with tensor values. Linear mapping is applied only to the values
    with the names given in the input_names_to_convert list, by default to "sentence_embedding" and "token_embeddings".

    The hope is that this adapter will work gracefully with SentenceTransformerTrainer...
    Possible point of failure: sentence_transformer library seems to relay on base.get_word_embedding_dimension() to get
    dimensionality of the embeddings. I was this once in initialization of the SentenceTransformer, hope this is not the
    case in the training code.
    """

    def __init__(self, original_dim: int, output_dim: int, input_names_to_convert: List[str] = None):
        super(LinearEmbeddingAdapter, self).__init__()
        if input_names_to_convert is None:
            input_names_to_convert = ["sentence_embedding", "token_embeddings"]

        self.original_dim = original_dim
        self.output_dim = output_dim
        self.input_names_to_convert = input_names_to_convert

        self.linear_layers = nn.ModuleDict(
            {name: nn.Linear(original_dim, output_dim) for name in input_names_to_convert}
        )

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = {}
        for name, values in input.items():
            if name in self.linear_layers:
                output[name] = self.linear_layers[name](values)
            else:
                output[name] = values
        return output

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "original_dim": self.original_dim,
            "output_dim": self.output_dim,
            "input_names_to_convert": self.input_names_to_convert,
        }

    def save(self, output_path: str) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
        # Save the state dictionary of the linear layers
        torch.save(self.linear_layers.state_dict(), os.path.join(output_path, "linear_layers.pt"))

    @staticmethod
    def load(input_path: str) -> 'LinearEmbeddingAdapter':
        with open(os.path.join(input_path, "config.json"), "r") as fIn:
            config = json.load(fIn)
        adapter = LinearEmbeddingAdapter(**config)
        # Load the state dictionary of the linear layers
        adapter.linear_layers.load_state_dict(torch.load(os.path.join(input_path, "linear_layers.pt"), map_location="cpu"))
        return adapter
