from sentence_transformers import (
    SentenceTransformer,
)
import numpy as np
from pathlib import Path
from huggingface_hub import HfApi
import os
import re

repo_id = "mim-nlp-project/ff-768"
# Specify the path to sentence_transformers outputs for your run.
outputs_folder_path = Path("./output/ff-768/2024-06-09_18-10-49")


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


# for checkpoint in np.arange(0, 18001, step=500):
for checkpoint in extract_available_checkpoints(outputs_folder_path):
    path = str(outputs_folder_path / f"checkpoint-{checkpoint}")
    model = SentenceTransformer(path)
    # sentence_transformers doesn't support pushing to a specific branch,
    # so we're pushing to main, and creating a target branch below.
    model.push_to_hub(repo_id=repo_id, private=True, exist_ok=True)

    # copy current main branch to the target branch
    api = HfApi()
    api.create_branch(
        repo_id=repo_id, branch=f"v1-checkpoint-{str(checkpoint).zfill(6)}"
    )
