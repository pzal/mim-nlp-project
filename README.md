# mim-nlp-project

## Installation

- ```bash
  pip install -e .
  ```
- Create `.env` file and provide your `NEPTUNE_API_TOKEN`. Optionally override values from `.public_env`.


## Usage

```bash
# baseline training
python3 scripts/train.py --model baseline --embedding-size 64 --version v2

# MTEB eval
python3 scripts/evaluate_on_mteb.py
````
