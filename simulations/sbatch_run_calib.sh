#!/bin/bash
#SBATCH -A p32622
#SBATCH -p "long"
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name="calibration_withinHost_CK"
#SBATCH --error=log/calibration_withinHost_CK.%j.err
#SBATCH --output=log/calibration_withinHost_CK.%j.out


module purge all

# Navigate to project directory

python run_calib.py
