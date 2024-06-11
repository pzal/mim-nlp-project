from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch

from matryoshka_experiment.training_models import create_sentence_transformer

load_dotenv(".public_env")

model = create_sentence_transformer("distilbert/distilroberta-base", output_dim=64)
text_data = [
        "search_query: What is TSNE?",
        "search_document: t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.",
        "search_document: Amelia Mary Earhart was an American aviation pioneer and writer.",
    ]
embeddings = model.encode(text_data)


print(embeddings.shape)

model.save("temp_model")

model = SentenceTransformer.load("temp_model")

embeddings2 = model.encode(text_data)

torch.testing.assert_allclose(embeddings, embeddings2)
