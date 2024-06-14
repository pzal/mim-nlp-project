import os
from dotenv import dotenv_values
import argparse
import torch

from matryoshka_experiment.training.baseline import train_baseline
from matryoshka_experiment.training.matryoshka import train_matryoshka

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["baseline", "matryoshka"])
parser.add_argument("--embedding-size", type=int, default=None)
parser.add_argument("--version", type=str)
parser.add_argument("--batch-size-per-gpu", type=int, default=8)
parser.add_argument("--tag", type=str, action="append")
parser.add_argument(
    "--load-pretrained", action=argparse.BooleanOptionalAction, default=False
)
command_args = parser.parse_args()

ENV = dotenv_values(".public_env")
ENV.update(dotenv_values(".env"))

os.environ["NEPTUNE_API_TOKEN"] = ENV["NEPTUNE_API_TOKEN"]
os.environ["NEPTUNE_PROJECT"] = ENV["NEPTUNE_PROJECT"]
os.environ["OUTPUT_DIR"] = ENV["OUTPUT_DIR"]
os.environ["HF_TOKEN"] = ENV["HF_TOKEN"]

print(f"CUDA available: {torch.cuda.is_available()}")
n_gpus = torch.cuda.device_count()
print(f"Number of CUDA devices: {n_gpus}")

assert n_gpus >= 1

if n_gpus > 1:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if command_args.model == "baseline":
    assert command_args.embedding_size is not None
    train_baseline(
        embedding_size=command_args.embedding_size,
        version=command_args.version,
        batch_size_per_gpu=command_args.batch_size_per_gpu,
        tags=command_args.tag,
        load_pretrained=command_args.load_pretrained,
    )
elif command_args.model == "matryoshka":
    train_matryoshka(
        version=command_args.version,
        batch_size_per_gpu=command_args.batch_size_per_gpu,
        tags=command_args.tag,
    )
