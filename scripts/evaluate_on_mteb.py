import mteb
from sentence_transformers import SentenceTransformer
from matryoshka_experiment.evaluation.model_mixins import PrependPromptMixin

OUR_TASKS = [
    (
        "AmazonCounterfactualClassification",
        "Represent the Amazon review for counterfactual classification; Input: ",
    )
]


def main():
    repo_id = "mim-nlp-project/ff-768"
    version = "v1"
    checkpoint_step = 18000

    model_name_or_path = repo_id
    revision = f"{version}-checkpoint-{str(checkpoint_step).zfill(6)}"
    model = SentenceTransformer(model_name_or_path, revision=revision)
    for task_name, prompt in OUR_TASKS:
        model = PrependPromptMixin(prompt, model, debug=False)
        tasks = mteb.get_tasks(tasks=[task_name])
        evaluation = mteb.MTEB(tasks, task_langs=["en"])
        results = evaluation.run(
            model,
            output_folder=f"evaluation_results/{model_name_or_path}-{revision}/{task_name}",
        )
        print(results)

    return results


if __name__ == "__main__":
    main()
