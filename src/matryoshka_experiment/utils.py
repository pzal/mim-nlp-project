from typing import Literal
import re
import os
from huggingface_hub import HfApi
from sentence_transformers import (
    SentenceTransformer,
)


def get_revision(
    checkpoint_step,
    version,
    type: Literal["finetuning", "pretraining"] = "finetuning",
):
    return f"{version}-{type}-checkpoint-{str(checkpoint_step).zfill(6)}"


def push_sentence_transformers_model_to_hf(model: SentenceTransformer, repo_id, branch):
    assert isinstance(model, SentenceTransformer)

    # sentence_transformers doesn't support pushing to a specific branch,
    # so we're pushing to main, and creating a target branch below.
    model.push_to_hub(repo_id=repo_id, private=True, exist_ok=True)

    # copy current main branch to the target branch
    api = HfApi()
    try:
        api.delete_branch(repo_id=repo_id, branch=branch)
    except Exception:
        print("Branch does not exist yet")
    api.create_branch(repo_id=repo_id, branch=branch)


def extract_available_checkpoints(directory_path):
    pattern = re.compile(r"checkpoint-(\d+)")
    checkpoint_numbers = []

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                number = int(match.group(1))
                checkpoint_numbers.append(number)

    return checkpoint_numbers
