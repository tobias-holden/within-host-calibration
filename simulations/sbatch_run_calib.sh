#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139testnode
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name="calibration_severe"
#SBATCH --error=log/calibration_severe.%j.err
#SBATCH --output=log/calibration_severe.%j.out


module purge all

# Navigate to project directory

/projects/b1139/environments/emodpy-torch/bin/python run_calib.py