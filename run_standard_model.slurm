#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --partition=common
#SBATCH --qos=3gpu1d
#SBATCH --mail-type=BEGIN,END,FAIL # Mail events
# (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pw431929@students.mimuw.edu.pl # Where to send mail
#SBATCH --time 00:10:00
#SBATCH --cpus-per-task=4 --mem-per-cpu=8GB --gpus=3

#SBATCH --output=slurm_logs/test_script.log # Standard output and error log
#SBATCH --error=slurm_logs/test_script.errors.txt

pwd; hostname; date
nvidia-smi -L

source /home/pw43/mim-nlp-project/.venv/bin/activate
cd /home/pw43/mim-nlp-project

torchrun --nproc_per_node=3 training_models/train_baseline.py

date