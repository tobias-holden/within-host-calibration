#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139testnode
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name="calib_IIVT-PT_VS_AGE"
#SBATCH --error=log/calib_IIVT-PT_VS_AGE.%j.err
#SBATCH --output=log/calib_IIVT-PT_VS_AGE.%j.out

# Purge modules
module purge all

# Activate virtual environment
source activate /projects/b1139/environments/emod_torch_tobias

<<<<<<< HEAD
cd /projects/b1139/calibration_scenarios/MII_variable_IIVT\-1/simulations
=======
cd /projects/b1139/calibration_scenarios/MII_variable_IIVT\-0/simulations
# Navigate to project directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd simulations

# Run calibration
>>>>>>> a6c9316 (move suite/experiment/simulation files from calibration_testing to experiments folder.)
python run_calib.py