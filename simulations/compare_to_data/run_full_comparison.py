import sys
sys.path.append('../')
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import os
import manifest as manifest
from simulations.helpers import load_coordinator_df

from simulations.compare_to_data.age_incidence_comparison import compute_inc_LL_for_all_sites, \
    plot_incidence_comparison_all_sites
from simulations.compare_to_data.age_severe_incidence_comparison import compute_severe_incidence_LL_for_all_sites, \
    plot_severe_incidence_comparison_all_sites 
from simulations.compare_to_data.age_prevalence_comparison import compute_prev_LL_for_all_sites, \
    plot_prevalence_comparison_all_sites
from simulations.compare_to_data.infectiousness_comparison import compute_infectious_LL_for_all_sites, \
    plot_infectiousness_comparison_all_sites
from simulations.compare_to_data.parasite_density_comparison import compute_parasite_density_LL_for_all_sites, \
    plot_density_comparison_all_sites
from simulations.compare_to_data.age_annual_prevalence_comparison import compute_annual_prev_LL_for_all_sites, \
    plot_annual_prevalence_comparison_all_sites
from simulations.compare_to_data.no_blood_comparison import compute_dead_LL_for_all_sites

def compute_LL_across_all_sites_and_metrics(numOf_param_sets = 64):
    coord_df = load_coordinator_df(characteristic=False, set_index=True)
    coord_df = coord_df[coord_df['include_site']]
    combined = []
    if len(coord_df[coord_df['infectiousness_to_mosquitos'] == True]) > 0: 
        infectious_LL = compute_infectious_LL_for_all_sites(numOf_param_sets)
        combined.append(infectious_ll)
    if len(coord_df[coord_df['age_parasite_density'] == True]) > 0: 
        density_LL = compute_parasite_density_LL_for_all_sites(numOf_param_sets)
        combined.append(density_LL)
    if len(coord_df[coord_df['age_prevalence'] == True]) > 0: 
        prevalence_LL = compute_prev_LL_for_all_sites(numOf_param_sets)
        combined.append(prevalence_LL)
        annual_prevalence_LL = compute_annual_prev_LL_for_all_sites(numOf_param_sets)
        combined.append(annual_prevalence_LL)
    if len(coord_df[coord_df['age_incidence'] == True]) > 0: 
        incidence_LL = compute_inc_LL_for_all_sites(numOf_param_sets)
        combined.append(incidence_LL)
    if len(coord_df[coord_df['age_severe_incidence'] == True]) > 0: 
        severe_incidence_LL = compute_severe_incidence_LL_for_all_sites(numOf_param_sets)
        combined.append(severe_incidence_LL)
    if len(coord_df[coord_df['dead_people_check'] == True]) > 0: 
        dead_LL = compute_dead_LL_for_all_sites(numOf_param_sets)
        combined.append(dead_LL)

    combined_df = pd.concat(combined)

    weighting_rules = pd.read_csv(os.path.join(manifest.base_reference_filepath,'weights.csv'))
    b=pd.merge(combined, weighting_rules,  how='left', left_on=['site','metric'], right_on = ['site','metric'])
    b['my_weight'].fillna(1.0, inplace=True)
    b['baseline'].fillna(-1.0, inplace=True)

    return b

def plot_all_comparisons(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):

    if len(coord_df[coord_df['infectiousness_to_mosquitos'] == True]) > 0:
        plot_infectiousness_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir) 
    if len(coord_df[coord_df['age_parasite_density'] == True]) > 0:
        plot_density_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir) 
    if len(coord_df[coord_df['age_prevalence'] == True]) > 0:
        plot_prevalence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
        plot_annual_prevalence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir) 
    if len(coord_df[coord_df['age_incidence'] == True]) > 0: 
        plot_incidence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    if len(coord_df[coord_df['age_severe_incidence'] == True]) > 0:
        plot_severe_incidence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir) 
    

if __name__ == "__main__":
    
    print(compute_LL_across_all_sites_and_metrics(3))