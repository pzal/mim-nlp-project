from itertools import product

import mteb
from sentence_transformers import SentenceTransformer
from matryoshka_experiment.evaluation.model_mixins import PrependPromptMixin


DIMENSIONS = [96, 192, 384, 768]
MODELS_TYPES = ["ff", "mrl"]
OUR_TASKS = [
    (
        "AmazonCounterfactualClassification",
        "Represent the Amazon review for counterfactual classification;",
    )
]

def get_model(model_type: str, dim: int) -> SentenceTransformer:
    if model_type == "ff":
        return SentenceTransformer(f"mim-nlp-project/ff-{dim}", revision="v2-finetuning-checkpoint-015000")
    elif model_type == "mrl":
        return SentenceTransformer(f"mim-nlp-project/mrl", revision="v2-finetuning-checkpoint-015000",
                                   truncate_dim=dim)
    else:
        raise ValueError(f"Invalid model type: {model_type}, it should be either 'ff' or 'mrl'")

def export_results_to_dataframe():
    pass



def main():
    for model_type, dimensionality in product(MODELS_TYPES, DIMENSIONS):

        model = get_model(model_type, dimensionality)

        for task_name, prompt in OUR_TASKS:
            model = PrependPromptMixin(prompt, model, debug=False)
            print(f"Model: {model_type}, Dimensionality: {dimensionality}, Task: {task_name}")
            tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
            evaluation = mteb.MTEB(tasks, task_langs=["eng"])
            results = evaluation.run(
                model,
                output_folder=f"evaluation_results/{model_type}_{dimensionality}/{task_name}",
            )
            # print(results)


if __name__ == "__main__":
    main()
