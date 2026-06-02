#!/bin/bash
#SBATCH --job-name=simulus
#SBATCH --output=outputs/slurm/%j.out
#SBATCH --error=outputs/slurm/%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=part-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sim

cd $SLURM_SUBMIT_DIR

python src/main.py benchmark=atari common.display_mode=disabled