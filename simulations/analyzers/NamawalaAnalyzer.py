import os
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from idmtools.entities.ianalyzer import IAnalyzer as BaseAnalyzer

import matplotlib as mpl
from idmtools.entities.iworkflow_item import IWorkflowItem
from idmtools.entities.simulation import Simulation

from logging import getLogger

from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType
from idmtools.core.platform_factory import Platform


mpl.use('Agg')


class AnnualSummaryReportAnalyzer(BaseAnalyzer):
    def __init__(self, expt_name, sweep_variables=None, working_dir="."):
        super().__init__(filenames=["output\\MalariaSummaryReport_Annual_Report.json"])
        self.expt_name = expt_name
        self.sweep_variables = sweep_variables or ["Run_Number", "Site"]
        self.working_dir = working_dir

    def initialize(self):
        """
        Initialize our Analyzer. At the moment, this just creates our output folder
        Returns:
        """
        if not os.path.exists(os.path.join(self.working_dir, self.expt_name)):
            os.mkdir(os.path.join(self.working_dir, self.expt_name))

    def map(self, data: Dict[str, Any], item: Union[IWorkflowItem, Simulation]) -> Any:
        """
        Extracts the Statistical Population, Data channel from InsetChart.
        Called for Each WorkItem/Simulation.
        Args:
            data: Data mapping str to content of file
            item: Item to Extract Data from(Usually a Simulation)
        Returns:
        """
        datatemp = data[self.filenames[0]]

        age_bins = datatemp['Metadata']['Age Bins']
        prevalence = datatemp['DataByTimeAndAgeBins']['PfPR by Age Bin']
        
        prevalence = np.array(np.array([i for i in prevalence]))
        prevalence[prevalence == 0] = np.nan
        if np.isnan(prevalence).all():
           prevalence = [0]
            
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            prevalence = np.nanmean(prevalence, axis=0)

        incidence = datatemp['DataByTimeAndAgeBins']['Annual Clinical Incidence by Age Bin']
        incidence = np.array(np.array([i for i in incidence]))
        #incidence[incidence == 0] = np.nan
        #if np.isnan(incidence).all():
        #    incidence = [0]
        #print(incidence)            
        incidence = np.nanmean(incidence, axis=0)

        pop = datatemp['DataByTimeAndAgeBins']['Average Population by Age Bin']
        pop = np.array(np.array([i for i in pop]))
        #print(pop[pop==0])
        for i in range(len(pop)):
         #   print(pop[i])
            if np.all(pop == 0):
                pop[i] = np.nan
        #pop[pop == 0] = np.nan
        #if np.all(pop == 0):
         #   pop
        if np.isnan(pop).all():
            pop = [0]
        #print(pop)            
        pop = np.nanmean(pop, axis=0)

        df = pd.DataFrame(list(zip(age_bins, prevalence, incidence, pop)),
                          columns=['Age', 'Prevalence', 'Incidence', 'Population'])

        return df

    def reduce(self, all_data: Dict[Union[IWorkflowItem, Simulation], Any]) -> Any:
        """
        Create the Final Population JSON and Plot
        Args:
            all_data: Populate data from all the Simulations
        Returns:
            None
        """
        df_final = pd.DataFrame()
        for s, v in all_data.items():
            dftemp = v.copy()
            for t in self.sweep_variables:
                dftemp[t] = [s.tags[t]]*len(v)
            df_final = pd.concat([df_final, dftemp])
        df_final.to_csv(os.path.join(self.working_dir, self.expt_name, "inc_prev_data_full.csv"))

        groupby_tags = self.sweep_variables
        groupby_tags.remove('Run_Number')
        df_summarized = df_final.groupby(['Age']+groupby_tags)[['Prevalence', 'Incidence', 'Population']].mean().reset_index()
        df_summarized_std = df_final.groupby(['Age']+groupby_tags)[['Prevalence', 'Incidence', 'Population']].std()
        for c in ['Prevalence', 'Incidence', 'Population']:
            df_summarized[c + '_std'] = list(df_summarized_std[c])

        df_summarized.to_csv(os.path.join(self.working_dir, self.expt_name, "inc_prev_data_final.csv"), index=False)


if __name__ == '__main__':
    # Set the platform where you want to run your analysis
    # In this case we are running in BELEGOST since the Work Item we are analyzing was run on COMPS
    logger = getLogger()
    with Platform('SLURM_LOCAL',job_directory='/projects/b1139/calibration_testing/experiments') as platform:

        # Initialize the analyser class with the path of the output csv file
        analyzers = [AnnualSummaryReportAnalyzer(expt_name='test')]

        # Set the experiment id you want to analyze
        experiment_id = '2a2f8217-7eeb-4f58-a05a-b1399a7ecc8f'

        # Specify the id Type, in this case an Experiment on COMPS
        manager = AnalyzeManager(partial_analyze_ok=True, ids=[(experiment_id, ItemType.EXPERIMENT)],
                                 analyzers=analyzers)
        manager.analyze()
