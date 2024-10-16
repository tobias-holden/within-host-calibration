import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from idmtools.entities.ianalyzer import IAnalyzer 

import matplotlib as mpl
from idmtools.entities.iworkflow_item import IWorkflowItem
from idmtools.entities.simulation import Simulation

from logging import getLogger

from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType
from idmtools.core.platform_factory import Platform


mpl.use('Agg')

class EventRecorderSummaryAnalyzer(IAnalyzer):
    """
    Pull out the ReportEventRecorder and stack them together.
    """

    def __init__(self,expt_name,sweep_variables=None, working_dir="./", time_cutoff=0, output_filename="event_counts"):
        super(EventRecorderSummaryAnalyzer, self).__init__(
            working_dir=working_dir, filenames=["output/ReportEventRecorder.csv"]
        )
        self.expt_name=expt_name
        self.sweep_variables = sweep_variables
        self.time_cutoff = time_cutoff
        self.output_filename = output_filename

    def map(self, data, simulation: Simulation):
        df = data[self.filenames[0]]
        df = df[df["Time"] >= self.time_cutoff].copy()
        df['Age_Year'] = [age // 365 for age in df['Age']]
        df2 = df.groupby(['Time',"Age_Year",'Event_Name'])['Individual_ID'].agg('count').reset_index()
        df2.rename(columns={"Individual_ID":"Event_Count"})
        # add tags
        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                df2[sweep_var] = simulation.tags[sweep_var]

        return df2

    def reduce(self, all_data):
        selected = [data for sim, data in all_data.items()]
        if len(selected) == 0:
            print("\nWarning: No data have been returned... Exiting...")
            return

        adf = pd.concat(selected).reset_index(drop=True)
        adf.to_csv(
            os.path.join(self.working_dir, self.expt_name, "".join((self.output_filename,".csv"))),
            index=False)
