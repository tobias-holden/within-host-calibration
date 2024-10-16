import os
import numpy as np
import pandas as pd

from simulations import manifest

from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

all_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'dead_people_check'] == 1 :
        all_sites.append(site)

 
#print(all_sites)
#print(f"No Blood Sites: {all_sites}")
#all_sites = ["dielmo_1990", "ndiop_1993", "laye_2007", "dapelogo_2007"]

def compute_dead_likelihood(combined_df):
    combined_df.dropna(inplace=True)
    combined_df['ll'] = np.nan
    for j in range(len(combined_df['No_Blood'])):
        if combined_df['No_Blood'][j] > 0:
            combined_df['ll'][j] = -10000
        elif combined_df['No_Blood'][j] == 0:
            combined_df['ll'][j] = 0
    return combined_df[["param_set","ll"]]



# The following function determines whether any parameters sets were missing for a site,
# if there are missing parameter set, this prepares compute_LL_by_site to shoot out a warning message
# This function additionally adds the missing parameter set to the dataframe with NaN for the ll.
def identify_missing_parameter_sets(combined_df, numOf_param_sets):

    param_list = list(range(1,numOf_param_sets+1))
    missing_param_sets = []
    for x in param_list:
        if x not in combined_df['param_set'].values:
            combined_df.loc[len(combined_df.index)] = [x,np.NaN]
            missing_param_sets.append(x)
    return combined_df, missing_param_sets
    
    
def compute_dead_LL_by_site(site, numOf_param_sets):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "no_blood.csv"))
    #sim_df = pd.read_csv(os.path.join('/projects/b1139/basel-hackathon-2023/simulations/output/3sites_240223/LF_0/SO',site,"no_blood.csv"))
    ll_by_param_set = compute_dead_likelihood(sim_df)
    
    ll_by_param_set, missing_param_sets = identify_missing_parameter_sets(ll_by_param_set, numOf_param_sets)
    
    ll_by_param_set["site"] = site
    ll_by_param_set["metric"] = "no_blood"

    if len(missing_param_sets) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets} for no blood')
        
    return ll_by_param_set
    
def compute_dead_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in all_sites:
        df_this_site = compute_dead_LL_by_site(s, numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)
