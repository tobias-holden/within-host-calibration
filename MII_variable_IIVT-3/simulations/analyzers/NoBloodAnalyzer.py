import os
import warnings
import pandas as pd
import re
from logging import getLogger
from idmtools.core.platform_factory import Platform
from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType
from idmtools.entities.ianalyzer import IAnalyzer as BaseAnalyzer

class NoBloodAnalyzer(BaseAnalyzer):
    '''
    defines running out of RBC analyzer class
    '''

    def __init__(self, expt_name, sweep_variables=None, working_dir="."): 

        super(NoBloodAnalyzer, self).__init__(working_dir=working_dir,
                                                 filenames=["stdout.txt"]) 
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.expt_name = expt_name


    def map(self, data, simulation):
        
        pattern = re.compile('Individual RBC Count = 0, calling Die()', re.IGNORECASE)
        linenum = 0
        rbc = 0
        start_pop = 0
        
        pop_pattern = re.compile('Time: 1.0 Rank: 0')
        
        myfile = data[self.filenames[0]]
        for line in myfile.split('\n'):
            linenum += 1
            if pop_pattern.search(line) != None:
                s=line[65:]
                s=s[:-8]
                start_pop = int(re.search(r'\d+', s)[0])
            if pattern.search(line) != None:
                rbc += 1
                    
        dict = {'No_Blood' : rbc,
        'start_pop' : start_pop
        }
        
        simdata = pd.DataFrame(dict,index=[1])
        
        # add tags
        for sweep_var in self.sweep_variables:
            simdata[sweep_var] = simulation.tags[sweep_var]

        return simdata

    def reduce(self, all_data):

        # concatenate all simulation data into one dataframe
        selected = [data for sim, data in all_data.items()]  # grab data in tuple form
        if len(selected) == 0:  # error out if no data selected
            print("\nNo data have been returned... Exiting...")
            return
        df = pd.concat(selected, sort=False).reset_index(drop=True)  # concat into dataframe

        grouping_list = self.sweep_variables
        df = df.sort_values(by=grouping_list)

        # write to csv
        fn = os.path.join(self.working_dir, self.expt_name)
        if not os.path.exists(fn):
            os.makedirs(fn)
        print('\nSaving data to: %s' % fn)
        df.to_csv(os.path.join(fn, 'no_blood.csv'), index=False)
       
        
if __name__ == '__main__':

    # Set the experiment id you want to analyze
    experiment_id = 'afd71e61-bf4e-4156-bf72-63bee56217f3'

    # Set the platform where you want to run your analysis
    # In this case we are running in BELEGOST since the Work Item we are analyzing was run on COMPS
    logger = getLogger()
    with Platform('SLURM_LOCAL') as platform:

        # Initialize the analyser class with the path of the output csv file
        analyzers = [NoBloodAnalyzer(expt_name='Dielmo_1990',
                                      sweep_variables=['Run_Number', 'Site', 'param_set'])]

        # Specify the id Type, in this case an Experiment on COMPS
        manager = AnalyzeManager(partial_analyze_ok=True, ids=[(experiment_id, ItemType.EXPERIMENT)],
                                 analyzers=analyzers)
        manager.analyze()        
