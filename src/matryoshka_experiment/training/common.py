import re
import os
import torch
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
)


def get_training_args(output_dir, epochs=1, learning_rate=None, batch_size_per_gpu=8):
    TARGET_BATCH_SIZE = 72

    accumulation_steps = (
        TARGET_BATCH_SIZE / torch.cuda.device_count() / batch_size_per_gpu
    )
    assert accumulation_steps % 1 == 0

    report_to = "none"
    if os.environ.get("RANK", 0) == 0:
        report_to = "neptune"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size_per_gpu,
        per_device_eval_batch_size=batch_size_per_gpu,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        dataloader_num_workers=os.cpu_count(),
        report_to=report_to,
        gradient_accumulation_steps=accumulation_steps,
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use validation loss to determine the best model
        greater_is_better=False,
        learning_rate=learning_rate or 5e-5,  # default from Transformers
        push_to_hub=False,
    )
    return training_args


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
