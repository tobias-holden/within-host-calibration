import os
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
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

annual_prevalence_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'age_prevalence'] == 1 and coord_csv.at[site, 'include_AnnualMalariaSummaryReport'] == True:
        annual_prevalence_sites.append(site)

print(f"Annual Prevalence Sites: {annual_prevalence_sites}")
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)

def prepare_annual_prevalence_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with an prevalence-by-age validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Process simulation data
    #fixme sample_number column will be renamed based on upstream analyzer output

    # Take mean over Run Number
    # sim_df = sim_df.groupby(["sample_number", "site", "Age", "p_detect_case"])\
    sim_df = sim_df.groupby(["param_set", "Age", "Site"]).agg({"Prevalence": "mean", "Population": "mean"}).reset_index()

    # upper_ages = sorted(sim_df['Age'].unique())
    # sim_df['mean_age'] = sim_df['Age'].apply(get_mean_from_upper_age, upper_ages=upper_ages)
    sim_df['p_detect_case'] = coord_csv[coord_csv['site'] == site]['p_detect_case'].iloc[0]

    # scale down prevalence in simulation according to probability of detecting a case in the reference setting
    sim_df['Prevalence'] = sim_df['Prevalence'] * sim_df['p_detect_case']
    
    # Load reference data
    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['age_prevalence_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df = ref_df[ref_df['Site'].str.lower() == site.lower()]
    #ref_df['Prevalence'] = ref_df['INC'] / 1000
    #ref_df['mean_age'] = (ref_df['INC_LAR'] + ref_df['INC_UAR']) / 2
    ref_df['ref.Trials'] = ref_df['total_sampled']
    ref_df['ref.Observations'] = ref_df['num_pos']

    # Prepare merge
    ref_df.rename(columns={"prevalence": "reference",
                           "year": "ref_year"},
                  inplace=True)
    ref_df = ref_df[["reference", "agebin", "Site", "ref_year", "ref.Trials","ref.Observations"]]
    
    sim_df.rename(columns={"Prevalence": "simulation",
                           "Age" : "agebin"},
                  inplace=True)
                  
    # def _match_sim_ref_ages_simple(df):
    #     # simple wrapper since we do not care about bench_df
    #     #fixme Code cleanup
    #     return match_sim_ref_ages(ref_df, df)[0]
    # 
    # sim_df = sim_df.groupby("param_set")\
    #     .apply(_match_sim_ref_ages_simple)\
    #     .reset_index(drop=True)#\
    #     #.drop(columns="index")                  

    
    combined_df = pd.merge(ref_df, sim_df, how='inner')
    combined_df['metric'] = 'prevalence'
    
    combined_df['a'] = combined_df['ref.Observations']
    combined_df['n'] = combined_df['ref.Trials']
    combined_df['r'] = combined_df['a'] / combined_df['n']
    
    combined_df['se'] = np.sqrt((1-combined_df['r'])/combined_df['a'])
    combined_df['LL'] = np.exp(np.log(combined_df['r']-1.96*combined_df['se']))
    combined_df['UL'] = np.exp(np.log(combined_df['r']+1.96*combined_df['se']))
    combined_df['LL'] = combined_df['LL'].fillna(0)
    
    #print("Combined DF")
    print(combined_df)
    
    #print(combined_df)

    #Need to make sure sim prevalence cannot be exactly 0, otherwise resulting likelihoods can be infinite/NaN
    def _correct_extremes(x):
        if x < 0.001:
            return 0.001
        else:
            return x

    #combined_df['simulation'] = combined_df['simulation'].apply(_correct_extremes)
    #print(combined_df)
    combined_df['sim.Trials'] = combined_df['Population']
    combined_df['sim.Observations'] = combined_df['sim.Trials']*combined_df['simulation']

    return combined_df


def compute_prevalence_likelihood1(combined_df):
    #combined_df.dropna(inplace=True)

    #fixme Maybe switch to an actual likelihood calculation, rather than Euclidean distance.
    # Do this by treating this as a Poisson process,
    # Note: assuming each person has an annual number of cases described by a Poisson rate, then the total
    # number of cases in the population is also described by a Poisson process

    poisson_ll = np.vectorize(poisson.logpmf) # probability mass function of poisson distribution

    # n = combined_df["ref_pop_size"]
    # poisson_rate_per_person = combined_df["simulation"]
    combined_df["poisson_rate_population"] = combined_df["ref_pop_size"]*combined_df["simulation"]
    combined_df["cases_observed"] = combined_df["ref_pop_size"]*combined_df["reference"]

    # Small fudge: Cases are discrete, and Poisson needs discrete values:
    combined_df["cases_observed"] = np.round(combined_df["cases_observed"])

    #combined_df.dropna(inplace=True)
    combined_df["ll"] = poisson_ll(combined_df["cases_observed"],
                                   combined_df["poisson_rate_population"])
    #print(combined_df)
    return combined_df["ll"].sum()

def compute_prevalence_likelihood2(combined_df):
    df=combined_df
    ll = gammaln(df['ref.Observations'] + df['sim.Observations'] + 1) - gammaln(df['ref.Observations'] + 1) - gammaln(df['sim.Observations'] + 1)

    ix = df['ref.Trials'] > 0
    ll.loc[ix] += (df.loc[ix]['ref.Observations'] + 1) * np.log(df.loc[ix]['ref.Trials'])

    ix = df['sim.Trials'] > 0
    ll.loc[ix] += (df.loc[ix]['sim.Observations'] + 1) * np.log(df.loc[ix]['sim.Trials'])

    ix = (df['ref.Trials'] > 0) & (df['sim.Trials'] > 0)
    ll.loc[ix] -= (df.loc[ix]['ref.Observations'] + df.loc[ix]['sim.Observations'] + 1) * np.log(df.loc[ix]['ref.Trials'] + df.loc[ix]['sim.Trials'])

    ll=ll.abs()
    ll=ll*-1
    
    
    #print(ll)
    return ll.sum(skipna=False) #mean


def compute_prevalence_likelihood3(combined_df):
    negbinom_ll = np.vectorize(nbinom.logpm) #PMF of negative binomial distribution
    
    combined_df["ll"] = binom_ll(combined_df["ref.Observations"],
                                 combined_df["ref.Trials"],
                                 combined_df["simulation"])
    #print(combined_df)
    return combined_df["ll"].sum()#mean()#
    


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
    
def compute_annual_prev_LL_by_site(site, numOf_param_sets):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "inc_prev_data_final.csv"))
    #sim_df = pd.read_csv(os.path.join(manifest.PROJECT_DIR,'simulations/output/8site_masked2/LF_0/SO', site, "inc_prev_data_final.csv"))
    combined_df = prepare_annual_prevalence_comparison_single_site(sim_df, site)

    ll_by_param_set = combined_df.groupby("param_set") \
        .apply(compute_prevalence_likelihood2) \
        .reset_index() \
        .rename(columns={0: "ll"})
        
    ll_by_param_set, missing_param_sets = identify_missing_parameter_sets(ll_by_param_set, numOf_param_sets)
    
    ll_by_param_set["site"] = site
    ll_by_param_set["metric"] = "prevalence"

    if len(missing_param_sets) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets} for prevalence')
    
    return ll_by_param_set


def compute_annual_prev_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in annual_prevalence_sites:
        df_this_site = compute_annual_prev_LL_by_site(s, numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)


def plot_annual_prevalence_comparison_single_site(site, param_sets_to_plot=[],plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "inc_prev_data_final.csv"))
    #sim_df = pd.read_csv(os.path.join('/projects/b1139/basel-hackathon-2023/simulations/output/3sites_240223/LF_0/SO',site,"inc_prev_data_final.csv"))
    combined_df = prepare_annual_prevalence_comparison_single_site(sim_df, site)
    #print(combined_df)
    if param_sets_to_plot is []:
        param_sets_to_plot = list(set(combined_df["param_set"]))

    #todo Add error bars on data
    # combined_df["reference_std"] = np.sqrt(combined_df["reference"])
    # combined_df["reference_std"] = np.sqrt(combined_df["ref_pop_size"]*combined_df["reference"])/combined_df["ref_pop_size"]

    plt.figure(f"{site}_prevalence")
    plt.fill_between(combined_df["agebin"].to_numpy(), 
                     combined_df["LL"].to_numpy(),
                     combined_df["UL"].to_numpy(),
                     label="reference", alpha=0.2)
    plt.plot(combined_df["agebin"].to_numpy(), combined_df["reference"].to_numpy(), label="reference", marker='o')
    for param_set, sdf in combined_df.groupby("param_set"):
        if param_set in param_sets_to_plot:
            plt.plot(sdf["agebin"].to_numpy(), sdf["simulation"].to_numpy(), label=f"PS {param_set}", marker='s')
    plt.xlabel("Age")
    plt.ylabel("Annual Prevalence")
    plt.title(site)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"prevalence_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"prevalence_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"prevalence_{site}.pdf"))
    plt.close()

def plot_annual_prevalence_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in annual_prevalence_sites:
        plot_annual_prevalence_comparison_single_site(s, param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)


if __name__=="__main__":
    print("Running...")
    
    plot_annual_prevalence_comparison_all_sites([1,2,3])
    print(compute_annual_prev_LL_for_all_sites(2).sort_values(by="ll").to_string())
