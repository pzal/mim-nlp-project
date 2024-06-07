import json
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

MEDI_JSON = "data/medi/medi-data.json"
ARROW_FILENAME = "data/medi/medi.arrow"

# Load your JSON data
with open(MEDI_JSON, 'r') as f:
    data = json.load(f)


# Prepare the data in the format expected by Hugging Face
processed_data = []
for item in tqdm(data):
    for i in range(len(item['query'])):
        processed_data.append({
            'query': item['query'][1],
            'pos': item['pos'][1],
            'neg': item['neg'][1],
            'task_name': item['task_name']
        })

# Convert to Pandas DataFrame
df = pd.DataFrame(processed_data)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Save to Arrow format
dataset.save_to_disk(ARROW_FILENAME)
print(f"Saved to {ARROW_FILENAME}")
