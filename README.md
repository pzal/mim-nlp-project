# mim-nlp-project

## Installation

- ```bash
  pip install -e .
  ```
- Create `.env` file and provide your `NEPTUNE_API_TOKEN` and `HF_TOKEN` token (with write access). Optionally override values from `.public_env`.
- You need HuggingFace initialized (logged in).


## Usage

```bash
# baseline training
python3 scripts/train.py --model baseline --embedding-size 64 --version v2 --batch-size-per-gpu 8

# MTEB eval
python3 scripts/evaluate_on_mteb.py
````
