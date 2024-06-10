from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from matryoshka_experiment.training_models.LinearEmbeddingAdapter import LinearEmbeddingAdapter


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


