#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name="calib_IIVT-CYTOKINE_KILLING"
#SBATCH --error=log/calib_IIVT-CYTOKINE_KILLING.%j.err
#SBATCH --output=log/calib_IIVT-CYTOKINE_KILLING.%j.out


module purge all
source activate /projects/b1139/environments/emod_torch_tobias

cd /projects/b1139/calibration_scenarios/MII_variable_IIVT\-4/simulations
python run_calib.py