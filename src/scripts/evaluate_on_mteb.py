import mteb
from sentence_transformers import SentenceTransformer
from matryoshka_experiment.evaluation.model_mixins import PrependPromptMixin

OUR_TASKS = [("AmazonCounterfactualClassification", "This is prompt for AmazonCounterfactualClassification")]

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2") # or whatever model we want to use
    for task_name, prompt in OUR_TASKS:
        model = PrependPromptMixin(prompt, model, debug=False)
        tasks = mteb.get_tasks(tasks=[task_name])
        evaluation = mteb.MTEB(tasks, task_langs=["en"])
        results = evaluation.run(model, output_folder=f"evaluation_results/{task_name}")
        print(results)


if __name__ == "__main__":
    main()