#!/bin/bash
#SBATCH --job-name=baseline_training
#SBATCH --partition=common
#SBATCH --qos=3gpu1d
#SBATCH --mail-type=BEGIN,END,FAIL # Mail events
# (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pw431929@students.mimuw.edu.pl # Where to send mail
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=4 --mem-per-cpu=8GB --gpus=3

#SBATCH --output=slurm_logs/baseline_training_%a.log # Standard output and error log
#SBATCH --error=slurm_logs/baseline_training_%a.errors.txt
#SBATCH --array=0-3%1

pwd; hostname; date
nvidia-smi -L

source /home/pw43/mim-nlp-project/.venv/bin/activate
cd /home/pw43/mim-nlp-project

torchrun --nproc_per_node=3 src/scripts/train_baseline.py  

date
