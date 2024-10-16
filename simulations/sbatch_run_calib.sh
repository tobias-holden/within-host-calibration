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

# Purge modules
module purge all

# Activate virtual environment
source activate /projects/b1139/environments/emod_torch_tobias

# Navigate to project directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd simulations

# Run calibration
python run_calib.py