import os, sys, shutil
sys.path.append('/projects/b1139/environments/emod_torch_tobias/lib/python3.8/site-packages/')
import argparse
import params as params
import manifest as manifest
from helpers import get_comps_id_filename, load_coordinator_df, get_suite_id

from idmtools.core.platform_factory import Platform
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType
from analyzers.EventRecorderSummaryAnalyzer import EventRecorderSummaryAnalyzer
from analyzers.NamawalaAnalyzer import AnnualSummaryReportAnalyzer as NamawalaAnalyzer
from analyzers.AnnualSummaryReportAnalyzer import AnnualSummaryReportAnalyzer
from analyzers.ParDensAgeAnalyzer import ParDensAgeAnalyzer
from analyzers.InfectiousnessByParDensAgeAnalyzer import InfectiousnessByParDensAgeAnalyzer
from analyzers.PatientReportAnalyzer import PatientAnalyzer
from analyzers.MonthlySummaryReportAnalyzer import MonthlySummaryReportAnalyzer
from analyzers.NoBloodAnalyzer import NoBloodAnalyzer
from wait_for_experiment import check_experiment


def run_analyzers(site: str, expid: str = None, characteristic: bool = False) -> (bool, str):
    """
    Wait for experiment to be done and run relevant analyzers for site on Comps with SSMT
    Args:
        site ():
        characteristic ():

    Returns: If experiment is succeeded, returns analyzer work item status and id,
             if not, return experiment status and id.

    """
    #platform = Platform(manifest.platform_name)
    platform = Platform('SLURM_LOCAL',job_directory=manifest.job_directory, mem=14000)
    comps_id_file = get_comps_id_filename(site=site)
    if expid:
        exp_id = expid
    else:
        with open(comps_id_file, 'r') as id_file:
            exp_id = id_file.readline()
    # Wait for experiment to be done
    a_ok=True
    if a_ok:#check_experiment(site, platform): #checks if succeeded, removed for now to allow for partial analysis
        coord_df = load_coordinator_df(characteristic=characteristic, set_index=True)

        # for expt_name, id in exp_name_id.items():
        # site = expt_name.replace('validation_', '')
        report_start_day = int(coord_df.at[site, 'report_start_day'])
        simulation_duration=int(coord_df.at[site, 'simulation_duration'])
        # determine the analyzers to run for each site
        analyzers = []
        analyzer_args = []
        jdir=manifest.job_directory
        wdir=manifest.simulation_output_filepath #'/home/aew2948/test_analyzer'#
        if not os.path.exists(wdir):
            os.mkdir(wdir)
        # if coord_df.at[site, 'include_EventReport']:
        #     analyzer_args.append({'expt_name':site,
        #                          'sweep_variables':['Run_Number', 'Site', 'param_set'],
        #                          'time_cutoff': 0,
        #                          'output_filename':"event_counts"})
        #     analyzers.append(EventRecorderSummaryAnalyzer(expt_name=site,
        #                                                   sweep_variables = ['Run_Number', 'Site', 'param_set'],
        #                                                   working_dir = wdir,
        #                                                   time_cutoff=0,
        #                                                   output_filename="event_counts"))
            
        if coord_df.at[site, 'include_MonthlyMalariaSummaryReport']:
            if coord_df.at[site, 'age_parasite_density']:
                #analyzers.append(ParDensAgeAnalyzer)
                analyzer_args.append({'expt_name': site,
                                      'sweep_variables': ['Run_Number', 'Site', 'param_set'],
                                      'start_year': int(report_start_day / 365),
                                      'end_year': int(simulation_duration / 365)})
                analyzers.append(ParDensAgeAnalyzer(expt_name= site,
                                      sweep_variables= ['Run_Number', 'Site', 'param_set'],
                                      start_year= int(report_start_day / 365),
                                      end_year= int(simulation_duration / 365),
                                      working_dir=wdir))
            if coord_df.at[site, 'infectiousness_to_mosquitos']:
                #analyzers.append(InfectiousnessByParDensAgeAnalyzer)
                analyzer_args.append({'expt_name': site,
                                      'sweep_variables': ['Run_Number', 'Site', 'param_set'],
                                      'start_year': int(report_start_day / 365),
                                      'end_year': int(simulation_duration / 365)})
                analyzers.append(InfectiousnessByParDensAgeAnalyzer(expt_name= site,
                                      sweep_variables= ['Run_Number', 'Site', 'param_set'],
                                      start_year= int(report_start_day / 365),
                                      end_year= int(simulation_duration / 365),
                                      working_dir=wdir))
            if coord_df.at[site, 'age_prevalence']:
                #analyzers.append(MonthlySummaryReportAnalyzer)
                analyzer_args.append({'expt_name': site,
                                      'sweep_variables': ['Run_Number', 'Site', 'param_set'],
                                      'start_year': int(report_start_day / 365),
                                      'end_year': int(simulation_duration / 365)})
                analyzers.append(MonthlySummaryReportAnalyzer(expt_name= site,
                                      sweep_variables= ['Run_Number', 'Site', 'param_set'],
                                      start_year= int(report_start_day / 365),
                                      end_year= int(simulation_duration / 365),
                                      working_dir=wdir))
        if coord_df.at[site, 'include_AnnualMalariaSummaryReport']:
            #analyzers.append(AnnualSummaryReportAnalyzer)
            analyzer_args.append({'expt_name': site,
                                  'sweep_variables': ['Run_Number', 'Site', 'param_set']})
            if site == "namawala_2001":
                analyzers.append(NamawalaAnalyzer(expt_name= site,
                                          sweep_variables= ['Run_Number', 'Site', 'param_set'],
                                          working_dir=wdir))
            else:
                analyzers.append(AnnualSummaryReportAnalyzer(expt_name= site,
                                          sweep_variables= ['Run_Number', 'Site', 'param_set'],
                                          working_dir=wdir))
        if coord_df.at[site, 'include_MalariaPatientReport']:  # infection duration
            #analyzers.append(PatientAnalyzer)
            analyzer_args.append({'expt_name': site,
                                  'start_report_day': report_start_day,
                                  'sweep_variables': ['Run_Number', 'x_Temp_LH_values', 'Site', 'param_set']})
            analyzers.append(PatientAnalyzer(expt_name= site,
                                      start_report_day= report_start_day,
                                      sweep_variables= ['Run_Number', 'x_Temp_LH_values', 'Site', 'param_set'],
                                      working_dir=wdir))
        if coord_df.at[site, 'dead_people_check']:  # check for running out of blood due to PT
            #analyzers.append(PatientAnalyzer)
            analyzer_args.append({'expt_name': site,
                                  'sweep_variables': ['Run_Number', 'Site', 'param_set']})
            analyzers.append(NoBloodAnalyzer(expt_name= site,
                                      sweep_variables=['Run_Number', 'Site', 'param_set'],
                                      working_dir=wdir))

        #analysis = PlatformAnalysis(platform=platform, experiment_ids=[exp_id],
         #                           analyzers=analyzers,
          #                          analyzers_args=analyzer_args,
           #                         analysis_name=site)

        suite_id = get_suite_id()
        #analysis.tags = {'Suite': suite_id}
        #analysis.analyze(check_status=True)

        #wi = analysis.get_work_item()
        analyzers_id_file = get_comps_id_filename(site=site, level=2)
        
        with Platform('SLURM_LOCAL',job_directory=manifest.job_directory, partition='short',
                            account='p32622', mem=20000) as platform:
            # Create AnalyzerManager with required parameters
            manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],
                                     analyzers=analyzers,  partial_analyze_ok=True)#, analyze_failed_items=True)#
            # Run analyze
            manager.analyze()

        #if wi.succeeded:
         #   print(f"Analyzer work item {wi.uid} succeeded.\n")
          #  with open(analyzers_id_file, 'w') as id_file:
           #     id_file.write(wi.uid.hex)
        #else:
         #   print(f"Analyzer work item {wi.uid} failed.")
        with open(analyzers_id_file, 'w') as id_file:
            id_file.write(site)
        with open(os.path.join(manifest.simulation_output_filepath,site,'finished.txt')  , 'w') as f: 
            f.write("I'm done running :]") 
        #return wi.succeeded, wi.uid
        return
    else:
        return False, exp_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process site name')
    parser.add_argument('--site', '-s', type=str, help='site name',
                        default=params.sites[0])  # not sure if we want to make this required argument
    parser.add_argument(
        "-i",
        "--expid",
        type=str,
        default=None
    )
    args = parser.parse_args()
    run_analyzers(args.site,args.expid)
