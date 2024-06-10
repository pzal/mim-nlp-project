from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from matryoshka_experiment.training_models.modules import create_sentence_transformer

load_dotenv(".public_env")

model = create_sentence_transformer("distilbert/distilroberta-base", output_dim=64)

embeddings = model.encode(
    [
        "search_query: What is TSNE?",
        "search_document: t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.",
        "search_document: Amelia Mary Earhart was an American aviation pioneer and writer.",
    ]
)

print(embeddings.shape)
