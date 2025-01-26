#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139testnode
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name="calibration_withinHost_DUAL"
#SBATCH --error=log/calibration_withinHost_DUAL.%j.err
#SBATCH --output=log/calibration_withinHost_DUAL.%j.out


module purge all

# Navigate to project directory

python run_calib.py