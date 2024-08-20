# basel-hackathon-2023
Repo for NU-IDM Basel hackathon to get EMOD hooked up with STPH calibration workflow that uses Gaussian Process Emulators. Scripts and sites are based off of IDM's malaria-model-validation repo as well as Aurelien Cavelan's bo repo for the GPE & botorch elements. Note that due to this use of old scripts there may be unnecessary files or builtin structures indicated for clean up.

A shared virtual environment is available on QUEST (pytorch-1.11-emodpy-py39). Due to some of the installation needs and how this repo was installed, the following is the preferred method to call into the environment when configured in your `.bashrc`:

```bash
load_calib (){
    source /projects/b1139/environments/pytorch-1.11-emodpy-py39/bin/activate
    module unload python
    export PYTHONPATH="${PYTHONPATH}:<path to repo>/basel-hackathon-2023/"
}
```

*Note that the python path should be updated to reflect where you have cloned the repository*.

Likewise, we are using a shared custom build (included in /simulations/download/bin_230614_PT) which is based off of Malaria-Ongoing current to June 14, 2023. This build includes Annie's work on expanding pyrogenic threshold and including age dependence. Scripts are available in the `NU-malaria-PT` branch of NU's DtkTrunk [here](https://github.com/numalariamodeling/DtkTrunk/tree/NU-malaria-PT/Eradication).

As of July 13, 2023, the calibration framework is largely functional with EMOD and its handlers when using the SLURM_LOCAL platform. We have hooked up seven sites (Ndiop, Dielmo, Laye, Dapelago, Matsari, Rafin Marke, and Sugungum) across five objectives (incidence, prevalence, asexual parasite density, gametocyte density, and infectiousness). Prior to runing in production, all sites, objectives, and scoring should be checked for accuracy and appropriateness in addition to choices on numbers of seeds to run and acquisition function(s). Parameters for calibration and their ranges should also be revisited prior to production (such as fixing max individual infections, likely to 9 based on Dan B's work trying to balance needs of general projects and vector genetics work).
        - For scoring: decisions need to be made on using one, combined score for all objectives and using individual objectives in parallel. Both options will require some thought towards weighting if there are some objectives that we deem of greater importance or confidence. Currently an "equitable weighting" is applied to the density objectives (LL scores / 10) as they tend to dominate the other objectives due to a greater number of reference comparisons. There are likely better ways to do this weighting long-term; however, this has been an effective bandaid.  
        - We should also consider adding some sort of bias factor for the garki/non-garki sites as STPH does due to differences between the sites in data collection. 

There are a number of important changes that have been made to configure this framework and avoid known issues in EMOD. Most notably, there is the "dead people check" analyzer which reads stdout.txt for individuals running out of blood which is not captured as an actual error. There is an additional check being completed during scoring of each intervention to ensure that all sites have outputs for all parameter sets - if not the combination will be marked as NaN so they will be dropped from the included parameter sets. This issue primarily occurs at the later stages of calibration iteration and is seen primarily in the Ndiop and Dielmo sites likely due to differences in site configuration that will require further exploration. So far we have chosen to use NaNs rather than arbitrary down weighting which negatively impacts the emulator's ability to predict how parameter sets will be scored. This could be modified in the future if desired as any parameter sets scored as NaNs can be re-suggested by the emulator for testing which does increase computational inefficiencies.
        - These issues often only arise when examining the overall calibration output file so it is *vital* to check, especially with changes and towards the end of the adaptive sampling iterations. The framework includes a lot of structure to allow it to keep running even if problems arise (like partial_analyze) which can exacerbate some of these problems. Many fixes we have made so far have been first identified here and likely would have been missed without it so please take the time to examine closely. As we run more and more evalutations it may be prudent to work to cut down any unnecessary outputs in this file and keep what is higher yield for identifying how well calibration is working (such as LL score reporting both overall and by site and objective).

**Overview of Scripts**

Generic Files:
- /batch_generators
    - Includes basic batch generators using the various acquisition functions. Expected_improvement.py and turbo_thompson_sampling.py should be functional for QUEST. Others may require some small improvements related to botorch functionalities.
- /create_pots
    - Mixed bag of helper scripts for coordinating relationships, reference sites, likelihood calculators, etc. Also includes some archival scripts from model validation workflow for plotting and generating the comparsisons. This folder likely needs some cleanup for our workflow.
- /emulators
    - Includes the basic setup for the GPE (GP.py). This script contains additional possible emulators; however, we have only used ExactGP so far. Others will need additional exploration if desired to use, such as ExactMultiTaskGP for emulating the emulator.
- /process_reference_data
    - Includes some basic reformatting R scripts that are not currently in use for this workflow. Likely can be removed.
- /reference_datasets
    - Includes csv and xlsx files containing the reference data points. All should be accurate and are currently in use but may be prudent to verify prior to production.
- /simulation_inputs
    - Generally contains input files to inform interventions, EIR, reporters, etc.
    - The simulation_coordinator.csv controls what actually will run. It contains details to match all of the inputs to sites, which reports and analyzers to run, etc. This file is incredibly important to the current setup to make sure you a referencing all of the right things and will include any sites or objectives of interest. It contains basic details on the sites and columns are largely self explanatory.
- /simulation_outputs (local)
    - All analyzer outputs will be saved here by site
- bo.py
    - Includes functions that control the entire calibration workflow, such as reading checkpoints, creating initial parameter sets, and what should be performed during each step/iteration of the workflow. Typically these can be left as is but note that this is where you may need to add changes for modifying the inner machinery of calibration.
- param_guess.csv
    - A sample of what our translated parameter inputs can look like during the calibration process. Note that it is relatively out of date and can be removed during cleaning if desired
- plot.py
    - Contains all of the end-of-calibration plotting functions such as convergence, x-flat (parameter value by evaluation number, colored by score), and prediction error
- pyproject.toml, requirements.txt, and setup.py
    - Mostly carried over from the model validation workflow and useful for "installation" of the repository. However, these are likely the cause of particular issues where the shared virtual environment may try to look on someone else's home directory. This installation has been removed from the shared environment, requiring the .bashrc statements above. It may still be useful to install your own version if working in your own environment. This setup may generally require some additional thought.

Simulation Files:
These files below are all contained within /simulations and are some of the most important in the repository. They will be the most used scripts as they control most of the knobs generated throughout the generic files.

- /COMPS_ID
    - This directory is a carryover and can likely be removed with some clean up. We don't use the same reference strucuture for finding experiments as model validation so these files are largely unnecessary for our work
- /Old_Eradication
    - Stores an older copy of eradication (and its schema and the index call) in order to see what previous versions may have used. We are using a custom build currently and not redownloading so it is largely not in use currently. 
- /analyzers
    - Includes all analyzer scripts, including NoBloodAnalyzer.py for the dead people check.
- /checkpoints
    - Includes results folders of calibration runs. For each checkpointed run there will will be "LF_X" subfolders that show the progress of findiing the best parameter sets - higher "LF_X"s are those that occur later during calibration. Within these folders there are epi plots for site and objective pairs as well as a csv containing the best parameter information and a emod.ymax.txt file containing the goodness of fit score. The overall best score will also be available in the main folder and the emod.n.txt will tell you which improvement step you are on (one greater than the highest "LF_X". These folders are particularly useful for understanding how your best parameter sets are performing relative to the reference data as well as where in the space they are.
- /compare_to_data
    - All goodness of fit analysis and plotting scripts can be found here. There are scripts for each objective under calibration that come together in run_full_comparison.py. Changes may need to be made to the calculations being done (and some objectives may already show multiple functions that could go into the calculation) and the full comparison may also need adjustments should it be selected to use separate scores for each objective rather than one aggregated (summed) score. The "equitable weighting" of density objectives is also included in the full comparison script as a simple division by 10 to support more equal weighting relative to the other objectives. Full weighting if desired may need to either be added to this script or in the run_calib script depending on particulars of the machinery that are not yet detailed here.
    - Note: each objective includes a check that all parameter sets are included. This is designed to avoid any scoring issues related to some sites and parameter set combinations failing while others succeed leading the emulator to focus on ostensibly good parameters, that actually result in quite a few problems, due to total scores that actually may be missing multiple site/objective combinations. This check now implements a NaN for any parameter set that has no data in simulation_outputs.
- /download
    - Contains eradication versions in use. The main level eradication is a previous download similar to the one in /Old_Eradication and can largely be ignored (fine to remove during cleanup), otherwise this is where new versions will download if not using local eradication. There is also a folder /bin_230614_PT that contains the pyrogenic threshold custom build based off of Annie's work. As described above, all DtkTrunk scripts are available on the team's fork of the repository. This is the copy of eradication currently being used for calibration. Other custom build folders may be added here if needed for future iterations. 
- /output
    - This directory is where all actual checkpointing for the calibration is saved as well as plots on the performance of the workflow (those written in plot.py). It may need some restructuring and/or renaming for clarity between it and the /checkpoints directory. Within a specific run folder you will find all of the necessary checkpoint scripts, such as the TurboThompson.json (or alternative acquisition function) and many pytorch (.pt) files to help botorch/the calibration machinery keep track of where it is at. This includes items like tested X (parameter sete) values (and respective Y/goodness of fits), iterations, etc.
    - Note: if picking up from a previous calibration run, all you need to do is make sure BO knows where to look for these files and update the maximum evaluations which is done easily in run_calib. For debugging, you may want to update the number of samples to test in the adaptive runs - this can ONLY be done by editing the acquisition function's .json file here's batch_size by hand. This is only recommended for debugging purposes where you may need to run in smaller batches.
- clean_all.py
    - Removes old analyzer and waiting batch scripts from repository 
- generate_site_rules.py
    - Snakemake model validation carryover, okay to remove if cleaning 
- get_eradication.py
    - Downloads new eradication if manifest doesn't dictate to use a local copy  
- get_version.py
    - Gives versioning on eradication being used if needed 
- helpers.py
    - Contains general helper functions for simulation setup. If trying to change the core workings of sims, config files, etc, this is the place to do it. 
- load_inputs.py
    - Loads the coordinator and sites requested 
- manifest.py
    - Primarily defines paths for use throughout the workflow but also includes information on the SIF, job directory, and whether or not we want to use the local eradication file. Note that if you want to redownload you should also adjust the download directory in this file as it would currently overwrite the custom build. The build should also have a copy on DtkTrunk and in the /projects/b1139/bins folder if needed for recovery outside of github.  
- my_func.py
    - Defines basic workflow for each iteration's simulations. Essentially calls the eradication, does parameter translation, submits the sims, and waits for the "finished.txt" file to indicate all analyzers are complete for each site and then runs the likelihood calculation. This function takes X values (normalized parameter sets) and returns Y (goodness of fit)
- params.py
    - Also mostly for snakemake workflow to generate the experiment name and number of sims to run. Connection needs to be checked since this has changed quite a bit. 
- run_analyzers.py
    - Dictates which analyzers to run for which sites based on simulation coordinator. Makes finished.txt when all analyzers for a site are done running. 
- run_calib.py
    - This is how we run the calibration machinery. It defines the calibration problem, where myfunc is called, and does some basic processing to remove any parameter sets that were scored as a NaN. From there is checks to see if there is a new best fit and will save data and epi plots if there is. Outside the problem, it defines the emulator (model) to use, acquisition function (batch generator), and runs the BO workflow. Once the workflow completes, the basic plot.py plotters will run.  
- run_sims.py
    - Simulation submision script that manages bringing all of the helpers together to create simulations. During simulation submission we also submit a scheduled waiting script and analyzer script (that gets called by the waiting script). This allows us to easily link our experiments to the analyzers and the waiting script allows us to utilizing less space and memory as it requires very little to submit the actual analyzer from there. These are based on ID dependencies from the experiment as definied in submit_sim. Previous versions of this workflow did not include this scheduled submission so can be referred to if needed to run the analyzers more separately (as seen in myfunc).
- sbatch_run_calib.sh
    - A simple sbatch script to run the calibration workflow (specifically run_calib.py). This script is especially useful for running large calibration runs where it will require days to run. Be sure to check all paths and batching requests (time, memory, allocation, etc)
- set_config.py
    - Sets general config parameters across the simulations. May or may not be in use relative to what is in helpers.py so will likely need checking. 
- snakefiles/snakemake
    - These are a carryover from model validation and can be removed during cleanup if we no longer want to run using this setup (likely true due to calibration machinery)
   
**Major Items Checklist for Updating Full Workflow - run_calib.py**
- `bo.initRandom`: does random sampling for the initial parameter sets
    - You supply how many you want and by how many batches (n_batches). For small initial samples 1 batch is fine, but for larger ones you should increase the number of batches such that each batch has something like 32 or 64 parameter sets each (ex: 640 parameter sets in 10 n_batches)
    - You can also define XPriors which is just particular parameter sets you want to include. We currently include the previous calibration values (in a list in the same order as in our key) to ensure that this spot in the parameter space is tested as we expect it to be a good fit.
- `TurboThompsonSampling` or `ExpectedImprovement`: defines the acquisition function/batch generator. Others are available in the repository but not currently in use.
    - Be sure to set up the acquisition function but also define it as the `batch_generator` in the next line.
    - `batch_size` here requests how many parameter sets to propose in the adaptive sampling iterations, not to be confused with `n_batches` above. If you want to run in sets of 64 you should request `batch_size=64` and the BO workflow will do this until the requested number of maximum evalutations is reached.
- `BO`: this function creates the calibration workflow. You supply the problem (largely calling myfunc and doing basic processing as well as monitoring best fit), model (emulator), batch_generator (acquisition function), checkpointdir (where the checkpoint pickup files will be saved), and max_evaluations.
    - Maximum evaluations are the total number of parameter sets that you are requesting the BO workflow to test, including the initialized set. 7000-8000 is likely a good range for production but will depend on how many dimensions (parameters) are under calibration. This can be changed during picking up from a checkpoint while the batch size cannot (without manual editing).  
