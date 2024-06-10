import json
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

MEDI_JSON = "data/medi/medi-data.json"
HF_DATASET_REPO = "mim-nlp-project/medi-joined"
REVISION = None


with open(MEDI_JSON, "r") as f:
    data = json.load(f)

processed_data = []
for item in tqdm(data):
    query = item["query"]
    pos = item["pos"]
    neg = item["neg"]

    assert len(query) == len(pos) == len(neg)

    if len(query) == 1:
        processed_data.append(
            {
                "anchor": query,
                "positive": pos,
                "negative": neg,
                "task_name": item["task_name"],
            }
        )
    else:
        assert len(query) == 2
        processed_data.append(
            {
                "anchor": f"{query[0]} Input: {query[1]}",
                "positive": f"{pos[0]} Input: {pos[1]}",
                "negative": f"{neg[0]} Input: {neg[1]}",
                "task_name": item["task_name"],
            }
        )

df = pd.DataFrame(processed_data)
dataset = Dataset.from_pandas(df)

seed = 42
train_indices, val_indices = train_test_split(
    range(len(dataset)), test_size=20000, stratify=df["task_name"], random_state=seed
)
train_indices, test_indices = train_test_split(
    train_indices,
    test_size=200000,
    stratify=df["task_name"][train_indices],
    random_state=seed,
)

train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)

dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})

dataset_dict.push_to_hub(HF_DATASET_REPO, revision=REVISION, private=True)
