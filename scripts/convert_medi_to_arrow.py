"""
Converts raw medi dataset into Arrow format. It creates train/val split 80/20
with stratification based on the task_name.

Parameters:
    MEDI_JSON (str): Path to the MEDI JSON file.
    ARROW_FILENAME (str): Path to the Arrow file.
    OVERWRITE (bool): If True, overwrite the existing Arrow file.

You can load the Arrow dataset using the following code:

```python
from datasets import load_from_disk
dataset = load_from_disk("data/medi/medi.arrow")

train_dataset = dataset['train']
val_dataset = dataset['val']
```
"""

import os
import shutil
import json
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

MEDI_JSON = "data/medi/medi-data.json"
ARROW_FILENAME = "data/medi/medi.arrow"
OVERWRITE = False

if os.path.exists(ARROW_FILENAME):
    if OVERWRITE:
        shutil.rmtree(ARROW_FILENAME)
    else:
        raise FileExistsError(f"The directory '{ARROW_FILENAME}' already exists.")


with open(MEDI_JSON, "r") as f:
    data = json.load(f)

processed_data = []
for item in tqdm(data):
    for i in range(len(item["query"])):
        processed_data.append(
            {"anchor": item["query"][1], "positive": item["pos"][1], "negative": item["neg"][1], "task_name": item["task_name"]}
        )

df = pd.DataFrame(processed_data)
dataset = Dataset.from_pandas(df)

train_indices, val_indices = train_test_split(
    range(len(dataset)), test_size=0.2, stratify=df["task_name"], random_state=42
)

train_dataset = dataset.select(train_indices)
val_dataset = dataset.select(val_indices)

dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})

dataset_dict.save_to_disk(ARROW_FILENAME)
print(f"Saved to {ARROW_FILENAME}")
