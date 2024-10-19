#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139testnode
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name="postcalib"
#SBATCH --error=log/postcalib.%j.err
#SBATCH --output=log/postcalib.%j.out


module purge all
source activate /projects/b1139/environments/emod_torch_tobias

python post_calibration_analysis.py --experiment '241013_20_max_infections' --prediction_plot --exclude_count 1000 --timer_plot --length_scales_plot --length_scales_by_objective