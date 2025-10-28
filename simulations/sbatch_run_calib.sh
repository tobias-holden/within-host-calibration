#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name="ricky"
#SBATCH --error=log/ricky.%j.err
#SBATCH --output=log/ricky.%j.out


module purge all

# Navigate to project directory

python run_calib.py