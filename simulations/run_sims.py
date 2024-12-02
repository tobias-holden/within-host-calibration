#!/usr/bin/env python3
import argparse
import os, sys, shutil
sys.path.append('/projects/b1139/environments/emod_torch_tobias/lib/python3.8/site-packages/')
import pandas as pd
import numpy as np
from functools import \
    partial 

# idmtools
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

# emodpy
from emodpy.emod_task import EMODTask
from emodpy_malaria.reporters.builtin import add_report_intervention_pop_avg

from helpers import set_param_fn, update_sim_random_seed, set_simulation_scenario_for_characteristic_site, \
    set_simulation_scenario_for_matched_site, get_comps_id_filename

from utils_slurm import submit_scheduled_analyzer

import params as params
import manifest as manifest


def submit_sim(site=None, nSims=1, characteristic=False, priority=manifest.priority, my_manifest=manifest,
               not_use_singularity=False, X=None):
    """
    This function is designed to be a parameterized version of the sequence of things we do 
    every time we run an emod experiment. 
    """
    # Create a platform
    # Show how to dynamically set priority and node_group
    platform_test=Platform("SLURM_LOCAL", job_directory=manifest.job_directory, partition='b1139testnode', time='12:00:00', 
                            account='b1139', modules=['singularity'], max_running_jobs=10, mem=2500)
    platform2 = Platform("SLURM_LOCAL", job_directory=manifest.job_directory, partition='short', time='4:00:00', 
                            account='p32622', modules=['singularity'], max_running_jobs=1000, mem=2500,
                            sbatch_custom=f"--job-name=run_{site}")
    platform1 = Platform("SLURM_LOCAL", job_directory=manifest.job_directory, partition='normal', time='12:00:00', 
                            account='p32622', modules=['singularity'], max_running_jobs=1000, mem=2500)
                            
    #platform = Platform(my_manifest.platform_name, priority=priority, node_group=my_manifest.node_group)
    #print("Prompting for COMPS creds if necessary...")

    experiment = create_exp(characteristic, nSims, site, my_manifest, not_use_singularity,platform1, X)

    # The last step is to call run() on the ExperimentManager to run the simulations.
    experiment.run(wait_until_done=False, platform=platform2)

    # Additional step to schedule analyzer to run after simulation finished running
    submit_scheduled_analyzer(experiment, platform1, site, analyzer_script='run_analyzers.py', mem=25000)

    # Save experiment id to file
    comps_id_file = get_comps_id_filename(site=site)
    with open(comps_id_file, "w") as fd:
        fd.write(str(experiment.uid))
    print()
    print(str(experiment.uid))
    return str(experiment.uid)

def add_calib_params(task, param, value, ptype):
    if ptype == "integer": #and (param != "Max_Individual_Infections" or (param == "Max_Individual_Infections" and value > 1)):
        task.set_parameter(param, int(value))
    #elif ptype == "integer" and param == "Max_Individual_Infections" and value <= 1:
     #   task.set_parameter('Enable_Superinfection', int(0))
    elif ptype in ["float", "double"]:
        task.set_parameter(param, float(value))
    elif ptype in ['string']:
        task.set_parameter(param, str(value))
        
    return {param: value}
    
def add_calib_param_func(simulation, calib_params, sets):
    X = calib_params[calib_params['param_set'] == sets]
    X = X.reset_index(drop=True)
    for j in range(len(X)):
        param=X['parameter'][j]
        value=X['emod_value'][j]
        ptype=X['type'][j]

        if ptype == "integer":
            simulation.task.set_parameter(param, int(value))
        elif ptype in ["float", "double"]:
            simulation.task.set_parameter(param, float(value))
        elif ptype in ['string']:
            simulation.task.set_parameter(param, str(value))
        
    return {'param_set':sets}

def create_exp(characteristic, nSims, site, my_manifest, not_use_singularity, platform, X):
    task = _create_task(my_manifest)

    if not not_use_singularity:
        task.set_sif(manifest.SIF_PATH, platform)
        #task.set_sif(my_manifest.sif_id.as_posix())
    task.config.parameters["Maternal_Antibodies_Type"] = "CONSTANT_INITIAL_IMMUNITY"
    task.config.parameters.Maternal_Antibodies_Type = "CONSTANT_INITIAL_IMMUNITY"
    task.config.parameters.Maternal_Antibody_Protection = 0.1239666434
    task.config.parameters.Maternal_Antibody_Decay_Rate = 0.01
    builder, exp_name = _create_builder(task,characteristic, nSims, site, X)
    # create experiment from builder

    experiment = Experiment.from_builder(builder, task, name=exp_name)
    return experiment


def _create_builder(task,characteristic, nSims, site, X):
    # Create simulation sweep with builder
    builder = SimulationBuilder()
    exp_name = "validation_" + site
    # Sweep run number
    builder.add_sweep_definition(update_sim_random_seed, range(nSims))
    # Sweep sites and seeds - based on values in simulation_coordinator csv
    # builder.add_sweep_definition(set_simulation_scenario, [site])
    if characteristic:
        builder.add_sweep_definition(set_simulation_scenario_for_characteristic_site, [site])
    else:
        builder.add_sweep_definition(set_simulation_scenario_for_matched_site, [site])
    
    #set if using csv/not running with my func
    #X = pd.read_csv(os.path.join(manifest.CURRENT_DIR,"10_initial_samples.csv"))

    builder.add_sweep_definition(partial(add_calib_param_func, calib_params=X), np.unique(X['param_set']))
    return builder, exp_name


def _create_task(my_manifest):
    # create EMODTask
    print("Creating EMODTask (from files)...")
    task = EMODTask.from_default2(config_path="my_config.json",
                                  eradication_path=str(my_manifest.eradication_path),
                                  ep4_custom_cb=None,
                                  campaign_builder=None,
                                  schema_path=str(my_manifest.schema_file),
                                  param_custom_cb=set_param_fn,
                                  demog_builder=None)
    # config.parameters.Clinical_Fever_Threshold_High = 0.1
    # add html intervention-visualizer asset to COMPS
    add_inter_visualizer = False
    if add_inter_visualizer:
        task.common_assets.add_asset(my_manifest.intervention_visualizer_path)
        add_report_intervention_pop_avg(task, my_manifest)
    return task


if __name__ == "__main__":
    # TBD: user should be allowed to specify (override default) erad_path and input_path from command line 
    # plan = EradicationBambooBuilds.MALARIA_LINUX
    # print("Retrieving Eradication and schema.json from Bamboo...")
    # get_model_files( plan, manifest )
    # print("...done.")

    parser = argparse.ArgumentParser(description='Process site name')
    parser.add_argument('--site', '-s', type=str, help='site name',
                        default="test_site")  # params.sites[0]) # todo: not sure if we want to make this required argument
    parser.add_argument('--nSims', '-n', type=int, help='number of simulations', default=params.nSims)
    parser.add_argument('--characteristic', '-c', action='store_true', help='site-characteristic sweeps')
    parser.add_argument('--not_use_singularity', '-i', action='store_true',
                        help='not using singularity image to run in Comps')
    parser.add_argument('--priority', '-p', type=str,
                        choices=['Lowest', 'BelowNormal', 'Normal', 'AboveNormal', 'Highest'],
                        help='Comps priority', default=manifest.priority)
    parser.add_argument('--calib_params', '-X', type=str, help='calib parameter set')
    args = parser.parse_args()

    submit_sim(site=args.site, nSims=args.nSims, characteristic=args.characteristic, priority=args.priority,
               not_use_singularity=args.not_use_singularity, X=args.calib_params)
