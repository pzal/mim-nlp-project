# mim-nlp-project

## Installation

- ```bash
  pip install -e .
  ```
- Create `.env` file and provide your `NEPTUNE_API_TOKEN` and `HF_TOKEN` token (with write access). Optionally override values from `.public_env`.
- You need HuggingFace initialized (logged in).


## Usage

### Baseline training
```bash
python3 scripts/train.py --model baseline --embedding-size 64 --version v2 --batch-size-per-gpu 8 --tag <your_tag_for_neptune>
```
Mandatory arguments:
- `--model` _baseline | matryoshka_
- `--embedding-size` _int_
- `--version` _string_

Optional arguments:
- `--batch-size-per-gpu` : 8 by default
- `--tag "some tag" --tag "some other tag"` [] by default
- `--load-pretrained` False by default

### MTEB eval
```bash
python3 scripts/evaluate_on_mteb.py
````
