# EMOD Within-Host Calibration 2024

Each main-level folder in this repository houses nearly identical calibration machinery (with a fair amount of redundancy, to be removed). There are minor differences for each of the different InnateImmuneVariationType models being tested.

0. NONE
1. Pyrogenic Threshold vs. Age
2. Pyrogenic Threshold vs. Age Increasing
3. Pyrogenic Threshold  vs. Age with Inter-Individual Variation
4. Cytokine Killing

Within each there are subfolders with the following structure:

## batch_generators

* generic
* array
* expected_improvement
* thompson_sampling
* turbo_expected_improvement
* **turbo_thompson_sampling**
* turbo_thompson_sampling_local
* turbo_upper_confidence_bound
* upper_confidence_bound

## create_plots

## emulators

GP.py defining...

* **Exact GP**
* Exact GP Turbo Local
* Exact GP Fixed Noise
* Exact Multitask GP
* Approximate GPs...


## process_reference_data

## reference_datasets

## simulation_inputs

Files describing options for...

* case_management
* daily_eirs or monthly_eirs
* demographics_files
* itns
* nonmalarial_fevers
* report_density_bins
* smc
* summary_report_age_bins
* survey_days

Helper files *create_sweep_coordinator_csv.py* and *setup_survey_days_input.py*

Plus the *simulation_coordinator.csv* tuning site-specific options

## simulation_outputs

## simulations

Some helper files:

* clean_all.py
* generate_site_rules.py
* get_eradication.py
* get_version.py
* helpers.py (modified per IIVT)
* load_inputs.py
* manifest.py
* my_func.py (# dimensions varies between IIVT)
* params.py
* run_analyzers.py
* run_calib.py
* run_sims.py
* sbatch_run_calib.sh
* set_config.py
* test_paramater_key.csv (hyperparameters vary based on IIVT)
* translate_parameters.py (hyperparameters vary based on IIVT)
* utils_slurm.py
* wait_for_experiment.py

### analyzers

* **AnnualSummaryReportAnalyzer**
* EventRecorderSummaryAnalyzer
* **InfectiousnessByParDensAgeAnalyzer**
* InsetChartAnalyzer
* **MonthlySummaryReportAnalyzer**
* **NamawalaAnalyzer**
* **NoBloodAnalyzer**
* ParDensAgeAnalyzer
* PatientReportAnalyzer
* PatientReportAnalyzer_laterDays

### compare_to_data

* **age_annual_prevalence_comparison.py**
* age_gametocyte_prevalence_comparison.py
* **age_incidence_comparison.py**
* **age_prevalence_comparison.py** (monthly)
* **infectiousness_comparison.py**
* **no_blood_comparison.py**
* **parasite_density_comparison.py**
* **run_full_comparison.py**

### download

* Eradication.exe
* schema.json
* bin_230614_PT (custom build)
    * **Eradication.exe**
    * **schema.json**

### log

### output
