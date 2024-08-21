# EMOD Within-Host Calibration 2024

A shared virtual environment is available on QUEST (pytorch-1.11-emodpy-py39). Due to some of the installation needs and how this repo was installed, the following is the preferred method to call into the environment when configured in your .bashrc:

```
load_calib (){
    source activate /projects/b1139/environments/emod_torch_tobias
}
```
Likewise, we are using a shared custom build (included in /simulations/download/bin_230614_PT) which is based on Malaria-Ongoing current to June 14, 2023. This build includes Annie's work on expanding pyrogenic threshold and including age dependence. Scripts are available in the NU-malaria-PT branch of NU's DtkTrunk here.

As of August 2024, the calibration framework is largely functional with EMOD and its handlers when using the SLURM_LOCAL platform. We are currently running with eight sites (Ndiop, Dielmo, Laye, Dapelago, Matsari, Rafin Marke, Sugungum, and Namawala) across five objectives (incidence, prevalence, asexual parasite density, gametocyte density, and infectiousness). For scoring, site-objectives log-likelihoods are divided by the corresponding log-likelihood for the team default parameter set with InnateImmuneVariationType=NONE. The MAXIMUM site-objective score is the 'score' seen by botorch, and being minimized through the workflow.

Each main-level folder in this repository houses nearly identical calibration machinery (with a fair amount of redundancy, to be removed). There are minor differences for each of the different InnateImmuneVariationType models being tested.

0. NONE
1. Pyrogenic Threshold vs. Age
2. Pyrogenic Threshold vs. Age Increasing
3. Pyrogenic Threshold  vs. Age with Inter-Individual Variation
4. Cytokine Killing

There are 17 parameters under calibration, plus 3 heterogeneity hyperparameters when running Innate Immune Variation Types #3 and #4. These parameters InnateImmuneDistributionFlag, InnateImmuneDistribution1, and InnateImmuneDistribution2 describe the variability of either Pyrogenic Threshold or Fever IRBC Kill Rate, respectively.

| id | parameter_label                            | parameter_name                                   | min      | max      | transform | team_default |
|----|--------------------------------------------|--------------------------------------------------|----------|----------|-----------|--------------|
|  1 | Antigen Switch Rate                        | `Antigen_Switch_Rate`                            | 1E-11    | 0.001    | log       | 7.65E-10     |
|  2 | Gametocyte Sex Ratio                       | `Base_Gametocyte_Fraction_Male`                  | 0.05     | 0.95     | none      | 0.2          |
|  3 | Gametocyte Mosquito Survival Rate          | `Base_Gametocyte_Mosquito_Survival_Rate`         | 0.0001   | 1        | log       | 0.00088      |
|  4 | Gametocyte Production Rate                 | `Base_Gametocyte_Production_Rate`                | 0.01     | 1        | log       | 0.0615       |
|  5 | Falciparum MSP Variants                    | `Falciparum_MSP_Variants`                        | 1        | 1000     | log       | 32           |
|  6 | Falciparum Nonspecific Types               | `Falciparum_Nonspecific_Types`                   | 1        | 1000     | none      | 76           |
|  7 | Falciparum PfEMP1 Variants                 | `Falciparum_PfEMP1_Variants`                     | 1        | 20000    | log       | 1070         |
|  8 | Max Fever Kill Rate of iRBCs               | `Fever_IRBC_Kill_Rate`                           | 0.1      | 1000     | log       | 1.4          |
|  9 | Gametocyte Stage Survival Rate             | `Gametocyte_Stage_Survival_Rate`                 | 0.01     | 1        | none      | 0.5886       |
| 10 | MSP Merozoite Kill Fraction                | `MSP1_Merozoite_Kill_Fraction`                   | 0.01     | 1        | none      | 0.511735322  |
| 11 | Nonspecific Antibody Growth Rate Factor    | `Nonspecific_Antibody_Growth_Rate_Factor`        | 0.01     | 1000     | log       | 0.5          |
| 12 | Nonspecific Antigenicity Factor            | `Nonspecific_Antigenicity_Factor`             | 0.000000001 | 1        | none      | 0.4151       |
| 13 | Pyrogenic Threshold                        | `Pyrogenic_Threshold`                            | 500      | 500000   | log       | 15000        |
| 14 | Max Individual Infections                  | `Max_Individual_Infections`                      | 3        | 10       | none      |              |
| 15 | Erythropoeisis Anemia Effect               | `Erythropoiesis_Anemia_Effect`                   | 0.5      | 5        | none      | 3.5          |
| 16 | RBC Destruction Multiplier                 | `RBC_Destruction_Multiplier`                     | 0.5      | 5        | none      | 3.29         |
| 17 | Cytokine Gametocyte Inactivation           | `Cytokine_Gametocyte_Inactivation`               | 0.001    | 1        | log       | 0.02         |
| 18 | InnateImmuneDistributionFlag               | `InnateImmuneDistributionFlag`                   | 0        | 4.9      | none      | 0            |
| 19 | InnateImmuneDistribution1                  | `InnateImmuneDistribution1`                      | 0        | 1        | none      | 0            |
| 20 | InnateImmuneDistribution2                  | `InnateImmuneDistribution2`                      | 0        | 1        | none      | 1            |




Within each IIVT 'branch' are subfolders with the following structure:

## bo.py

Includes functions that control the entire calibration workflow, such as reading checkpoints, creating initial parameter sets, and what should be performed during each step/iteration of the workflow. Typically these can be left as is but note that this is where you may need to add changes for modifying the inner machinery of calibration.

## batch_generators

Includes basic batch generators using the various acquisition functions. Expected_improvement.py and turbo_thompson_sampling.py should be functional for QUEST. Others may require some small improvements related to botorch functionalities.

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

A mixed bag of helper scripts for coordinating relationships, reference sites, likelihood calculators, etc. Also includes some archival scripts from model validation workflow for plotting and generating the comparisons. This folder likely needs some cleanup and may not currently be used in the workflow.

## emulators

GP.py defining the basic setup for the GP Emulation. This script contains additional possible emulators; however, we have only used ExactGP and explored ExactMultiTaskGP so far.

* **Exact GP**
* Exact GP Turbo Local
* Exact GP Fixed Noise
* Exact Multitask GP
* Approximate GPs...

## process_reference_data

Includes some basic reformatting R scripts that are no longer currently in use for this workflow.

## reference_datasets

Home for .csv and .xlsx files containing the reference data points. 

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

Plus the ***simulation_coordinator.csv*** for tuning site-specific options

## simulation_outputs

All analyzer outputs will be saved here by site, and cleared after each 'batch' of simulations. Outputs are currently copied to the folder simulations/output/<experiment>/LF_<batch_number>/SO/

## simulations

Some helper files:

* clean_all.py - Removes old analyzer and waiting batch scripts from repository
* generate_site_rules.py - snakemake remnant, not in use
* get_eradication.py - Downloads new eradication if manifest doesn't dictate to use a local copy
* get_version.py - Gives version of eradication being used if needed
* helpers.py (modified per IIVT) - Contains general helper functions for simulation setup. If trying to change the core workings of sims, config files, etc, this is the place to do it.
* load_inputs.py - Loads the coordinator and sites requested
* manifest.py - Primarily defines paths for use throughout the workflow but also includes information on the SIF, job directory, and whether or not we want to use the local eradication file. Note that if you want to redownload you should also adjust the download directory in this file as it would currently overwrite the custom build. The build should also have a copy on DtkTrunk and in the /projects/b1139/bins folder if needed for recovery outside of github.
* my_func.py (# dimensions varies between IIVT) - Defines basic workflow for each iteration's simulations. Essentially calls the eradication, does parameter translation, submits the sims, and waits for the "finished.txt" file to indicate all analyzers are complete for each site and then runs the likelihood calculation. This function takes X values (normalized parameter sets) and returns Y (goodness of fit)
* params.py - snakemake remnant, not in use
* run_analyzers.py - Dictates which analyzers to run for which sites based on simulation coordinator. Makes finished.txt when all analyzers for a site are done running.
* run_calib.py (# dimensions varies between IIVT) - This is how we run the calibration machinery. It defines the calibration problem, where myfunc is called, and does some basic processing to remove any parameter sets that were scored as a NaN. From there is checks to see if there is a new best fit and will save data and epi plots if there is. Outside the problem, it defines the emulator (model) to use, acquisition function (batch generator), and runs the BO workflow. Once the workflow completes, the basic plot.py plotters will run.
* run_sims.py - Simulation submision script that manages bringing all of the helpers together to create simulations. During simulation submission we also submit a scheduled waiting script and analyzer script (that gets called by the waiting script). This allows us to easily link our experiments to the analyzers and the waiting script allows us to utilizing less space and memory as it requires very little to submit the actual analyzer from there. These are based on ID dependencies from the experiment as definied in submit_sim. Previous versions of this workflow did not include this scheduled submission so can be referred to if needed to run the analyzers more separately (as seen in myfunc).
* sbatch_run_calib.sh - A simple sbatch script to run the calibration workflow (specifically run_calib.py). This script is especially useful for running large calibration runs where it will require days to run. Be sure to check all paths and batching requests (time, memory, allocation, etc)
* set_config.py - Sets general config parameters across the simulations. Somewhat (potentially totally) redundant/useless given helpers.py
* test_paramater_key.csv (hyperparameters vary based on IIVT) - list of parameters under calibration, their transformation to unit space, data type, min/max, and team default values.
* translate_parameters.py (hyperparameters vary based on IIVT) - script for converting locations in unit parameter space to emod-compatible values, and vice-versa, according to 'test_parameter_key.csv'
* utils_slurm.py - slurm-specific helper functions for chain job submission on QUEST
* wait_for_experiment.py - helper to submit small job monitoring for simulation end before requesting resources for expensive analyzers.

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
