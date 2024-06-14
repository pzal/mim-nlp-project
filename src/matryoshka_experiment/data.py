from datasets import load_dataset


def get_datasets():
    # Ensure that we're pulling the same dataset every time
    dataset = load_dataset(
        "mim-nlp-project/medi-joined",
        revision="c227a20a8a3ecc3e7eec431a162946b74317e6cb",
    )

    train_dataset = dataset["train"].select_columns(["anchor", "positive", "negative"])
    val_dataset = dataset["val"].select_columns(["anchor", "positive", "negative"])
    test_dataset = dataset["test"].select_columns(["anchor", "positive", "negative"])

    return train_dataset, val_dataset, test_dataset
