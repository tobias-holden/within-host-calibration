import os
import sys
sys.path.append('../')
import warnings

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
from scipy.stats import poisson, binom, nbinom
import math
from scipy.special import gammaln
from simulations import manifest
from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df
from create_plots.helpers_reformat_sim_ref_dfs import get_mean_from_upper_age, \
    match_sim_ref_ages

coord_csv = load_coordinator_df(characteristic=False, set_index=True)#pd.read_csv(manifest.simulation_coordinator_path)

severe_incidence_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'age_severe_incidence'] == 1 :
        severe_incidence_sites.append(site)

print(f"Severe Incidence Sites: {severe_incidence_sites}")
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)

def prepare_severe_incidence_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with an incidence-by-age validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Process simulation data
    sim_df_sorted = sim_df.sort_values(by='Age')  # Sort by 'age'
    sim_df_sorted['cumulative_severe_incidence'] = (sim_df_sorted.groupby(['param_set',"Site"])['Severe Incidence'].cumsum())
    sim_df=sim_df_sorted

    # Take mean over Run Number
    sim_df = sim_df.groupby(["param_set", "Age", "Site"]).agg({"cumulative_severe_incidence": "mean", "Population": "mean"}).reset_index()
    # Load reference data
    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['age_severe_incidence_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df = ref_df[ref_df['Site'] == site]
    
    # Prepare merge
    ref_df.rename(columns={"cumulative_severe_incidence": "reference"},
                  inplace=True)
    ref_df = ref_df[["reference", "Age", "Site","n"]]
    
    sim_df.rename(columns={"cumulative_severe_incidence": "simulation"},
                  inplace=True)              

    # Merge
    combined_df = pd.merge(ref_df, sim_df, how='inner')
    combined_df['metric'] = 'severe_incidence'


    return combined_df


# The following function determines whether any parameters sets were missing for a site,
# if there are missing parameter set, this prepares compute_LL_by_site to shoot out a warning message
# This function additionally adds the missing parameter set to the dataframe with NaN for the ll.
def identify_missing_parameter_sets(combined_df, numOf_param_sets):
    param_list = list(range(1,numOf_param_sets+1))
    missing_param_sets = []
    for x in param_list:
        if x not in combined_df['param_set'].values:
            combined_df.loc[len(combined_df.index)] = [x,np.nan]
            missing_param_sets.append(x)
    return combined_df, missing_param_sets

def compute_severe_incidence_likelihood(combined_df):

    # fixme Maybe switch to an actual likelihood calculation, rather than RSS.
    # Do this by treating this as a Poisson process,
    # Note: assuming each person has an annual number of cases described by a Poisson rate, then the total
    # number of cases in the population is also described by a Poisson process
    combined_df['ll'] = combined_df['reference']-combined_df['simulation']
    combined_df['ll'] = combined_df['ll']**2
    return combined_df["ll"].sum()
  
  
def compute_severe_incidence_likelihood_poisson(combined_df):
   
    # Note: assuming each person has an annual number of cases described by a Poisson rate, then the total
    # number of cases in the population is also described by a Poisson process

    poisson_ll = np.vectorize(poisson.logpmf) # log probability mass function of poisson distribution

    combined_df["cases_predicted"] = combined_df["n"]*combined_df["simulation"]
    combined_df.loc[combined_df["cases_predicted"] == 0, "cases_predicted"] = 0.01
    combined_df["cases_observed"] = combined_df["n"]*combined_df["reference"]

    # Small fudge: Cases are discrete, and Poisson needs discrete values:
    combined_df["cases_observed"] = np.round(combined_df["cases_observed"])

    #combined_df.dropna(inplace=True)
    combined_df["ll"] = poisson_ll(combined_df["cases_observed"],
                                   combined_df["cases_predicted"])
                                   

    return combined_df["ll"].sum()


def compute_severe_incidence_LL_by_site(site, numOf_param_sets):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "inc_prev_data_final.csv"))
    combined_df = prepare_severe_incidence_comparison_single_site(sim_df, site)

    ll_by_param_set = combined_df.groupby("param_set") \
        .apply(compute_severe_incidence_likelihood_poisson) \
        .reset_index() \
        .rename(columns={0: "ll"})
        
    ll_by_param_set, missing_param_sets = identify_missing_parameter_sets(ll_by_param_set, numOf_param_sets)
    
    ll_by_param_set["site"] = site
    ll_by_param_set["metric"] = "severe_incidence"

    if len(missing_param_sets) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets} for severe incidence')
    
    return ll_by_param_set

    
def compute_severe_incidence_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in severe_incidence_sites:
        df_this_site = compute_severe_incidence_LL_by_site(s, numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)



def plot_severe_incidence_comparison_single_site(site, param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "inc_prev_data_final.csv"))
    combined_df = prepare_severe_incidence_comparison_single_site(sim_df, site)
    
    if param_sets_to_plot is None:
        param_sets_to_plot = list(set(combined_df["param_set"]))
    plt.figure(f"{site}_severe_incidence")
    for param_set, sdf in combined_df.groupby("param_set"):
        if param_set in param_sets_to_plot:
            sdf = sdf.sort_values('Age')
            sdf['simulation'] = sdf['simulation']
            plt.plot(sdf["Age"].to_numpy(), sdf["simulation"].to_numpy(), label=f"PS {param_set}",zorder=0)
    plt.scatter(combined_df["Age"].to_numpy(), combined_df["reference"].to_numpy(), label="reference",color="k",zorder=1)
    plt.xlabel("Age")
    plt.ylabel("Cumulative Severe Incidence")
    plt.title(site)
    plt.legend(loc='lower right')
    plt.tight_layout()
    #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"incidence_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"severe_incidence_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"severe_incidence_{site}.pdf"))
    plt.close()

def plot_severe_incidence_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in severe_incidence_sites:
        plot_severe_incidence_comparison_single_site(s, param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)


if __name__=="__main__":
    print("Running...")
    #site="siaya_1996"
    ps=[1,2]
    plot_severe_incidence_comparison_all_sites(param_sets_to_plot=ps,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots"))
    print(compute_severe_incidence_LL_for_all_sites(len(ps)))