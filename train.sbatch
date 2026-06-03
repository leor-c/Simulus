#!/bin/bash
#SBATCH --job-name=simulus
#SBATCH --output=outputs/slurm/%j.out
#SBATCH --error=outputs/slurm/%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source ~/.bashrc
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sim2
cd projects/simulus-private

python src/main.py benchmark=atari common.display_mode=disabled "$@"