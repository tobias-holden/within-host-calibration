import os
import sys
from functools import partial
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sys.path.append("../")

sys.path.append('../../')
from calibration_common.create_plots.helpers_reformat_sim_ref_dfs import get_mean_from_upper_age, \
    match_sim_ref_ages, get_age_bin_averages, combine_higher_dens_freqs
from simulations import manifest

from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

density_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'age_parasite_density'] == 1 :
        density_sites.append(site)

#density_sites = ["sugungum_1970", "rafin_marke_1970","matsari_1970","dapelogo_2007","laye_2007"]

print(f"Density Sites: {density_sites}")
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)

def prepare_parasite_density_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with a parasite density validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Load reference data
    # todo: write a common method to generate age_agg_df for sim, ref and benchmark data
    sim_df = sim_df[sim_df['agebin']!=0.5].reset_index()
    upper_ages = sorted(sim_df['agebin'].unique())
    sim_df['mean_age'] = sim_df['agebin'].apply(get_mean_from_upper_age, upper_ages=upper_ages)
    #print("SIM MEAN AGES")
    #print(sim_df['mean_age'].unique())
    if "level_1" in sim_df.columns:
        #print("is In")
        age_agg_sim_df = sim_df.groupby("param_set")\
            .apply(get_age_bin_averages)\
            .reset_index()\
            .drop(columns="level_1")
    else:
        #print(" not in")
        age_agg_sim_df = sim_df.groupby("param_set")\
            .apply(get_age_bin_averages)\
            .reset_index()\
            .drop(columns="level_1")
            
    #print(age_agg_sim_df)
    
    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['age_parasite_density_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df = ref_df[ref_df['Site'].str.lower() == site.lower()]
    ref_df['Site'] = ref_df['Site'].str.lower()
    upper_ages = sorted(ref_df['agebin'].unique())
    ref_df['mean_age'] = ref_df['agebin'].apply(get_mean_from_upper_age, upper_ages=upper_ages)
    #print("REFERENCE MEAN AGES")
    #print(ref_df['mean_age'].unique())

    # subset simulation output to months in reference dataset
    months = sorted(ref_df['month'].unique())
    #print(site)
    #print(months)
    #print(sim_df)
    #print(age_agg_sim_df)
    #print(sim_df)
    sim_df = age_agg_sim_df[age_agg_sim_df['month'].isin(months)]

    # if the maximum reference density bin is < (maximum simulation density bin / max_magnitude_difference), aggregate all simulation densities >= max ref bin into the max ref bin
    #   the final bin will be all densities equal to or above that value
    max_ref_dens = ref_df['densitybin'].max(skipna=True)

    combine_higher_dens_freqs_simplified = partial(combine_higher_dens_freqs,
                                                   max_ref_dens=max_ref_dens,
                                                   max_magnitude_difference=100)
    #print("1")
    #print(sim_df)
    sim_df = sim_df.groupby("param_set", as_index=False)\
        .apply(combine_higher_dens_freqs_simplified)\
        .reset_index(drop=True)
        
    #sim_df['param_set'] = sim_df.index
    #print('2')
    #print(sim_df)
    # add zeros for unobserved reference densities up to max_ref_dens.
    # Do this by taking the sim dataframe for a single parameter set and using it to add missing zeros in ref_df
    all_zeros_df = sim_df[sim_df["param_set"]==np.min(sim_df["param_set"])].reset_index(drop=True)
    all_zeros_df = all_zeros_df[['month', 'mean_age', 'agebin', 'densitybin', 'Site']]
    ref_df = pd.merge(ref_df, all_zeros_df, how='outer')
    ref_df.fillna(0, inplace=True)

    # fixme - match_sim_ref_ages is currently copied over from the validation framework and it generates an additional
    # fixme - dataframe bench_df that we don't care about right now

    def _match_sim_ref_ages_simple(df):
        #fixme Simple wrapper function since we don't care about bench_df
        return match_sim_ref_ages(ref_df, df)[0]

    sim_df = sim_df.groupby("param_set")\
        .apply(_match_sim_ref_ages_simple)\
        .reset_index(drop=True)#\
        #.drop(columns="index")
    #print("3")
    #print(sim_df)

    # # format reference data
    # if "year" in ref_df.columns:
    #     print(f"A: {site}")
    #     min_yr = np.min(ref_df['year'])
    #     ref_df['year'] = ref_df['year']-min_yr+1
    #     print(ref_df)
    # if not "year" in ref_df.columns:
    #     print(f"B: {site}")
    #     ref_df.insert(0,'year',1)
    #     print(ref_df)
    
    ref_df["site_month"] = ref_df['Site'] + '_month' + ref_df['month'].astype('str')
    #ref_df["site_month"] = ref_df['Site'] + '_year' + ref_df['year'].astype('str') + '_month' + ref_df['month'].astype('str')
    ref_df_asex = ref_df[["asexual_par_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month",
                          "site_month", "bin_total_asex", "count_asex"]] \
        .rename(columns={"asexual_par_dens_freq": "reference",
                         "bin_total_asex": "ref_total",
                         "count_asex": "ref_bin_count"})
    ref_df_gamet = ref_df[["gametocyte_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month",
                           "site_month", "bin_total_gamet", "count_gamet"]] \
        .rename(columns={"gametocyte_dens_freq": "reference",
                         "bin_total_gamet": "ref_total",
                         "count_gamet": "ref_bin_count"})

    # format new simulation output
    # min_yr = np.min(sim_df['year'])
    # sim_df['year'] = sim_df['year']-min_yr+1
    sim_df["site_month"] = sim_df['Site'] + '_month' + sim_df['month'].astype('str')
    #sim_df["site_month"] = sim_df['Site'] + '_year' + sim_df['year'].astype('str') + '_month' + sim_df['month'].astype('str')
    sim_df_asex = sim_df[["param_set", "asexual_par_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month","site_month"]] \
        .rename(columns={"asexual_par_dens_freq": "simulation"})
    sim_df_gamet = sim_df[["param_set", "gametocyte_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month","site_month"]] \
        .rename(columns={"gametocyte_dens_freq": "simulation"})

    # combine reference and simulation dataframes
    combined_df_asex = pd.merge(ref_df_asex, sim_df_asex, how='outer')
    combined_df_asex['metric'] = 'asexual_density'
    #print(combined_df_asex)
    combined_df_gamet = pd.merge(ref_df_gamet, sim_df_gamet, how='outer')
    combined_df_gamet['metric'] = 'gametocyte_density'
    #print(combined_df_gamet)
    return combined_df_asex, combined_df_gamet


def compute_density_likelihood(combined_df):
    """
    Calculate an approximate likelihood for the simulation parameters for each site. This is estimated as the product,
    across age groups, of the probability of observing the reference values if the simulation means represented the
    true population mean
    Args:
        combined_df (): A dataframe containing both the reference and matched simulation output
        sim_column (): The name of the column of combined_df to use as the simulation output
    Returns: A dataframe of loglikelihoods where each row corresponds to a site-month

    """
    #fixme Monique had a different approach in the model-validation framework
    #fixme 230328: JS changed to a simplified approach: naively assume every observation is independent.
    # Likelihood of each observation is likelihood of seeing reference data if simulation is "reality"
    binom_ll = np.vectorize(binom.logpmf) # probability mass function of binomial distribution

    #combined_df.dropna(inplace=True)

    #fixme - Fudge to correct for sim density prevalence values of 1s and 0s (often because of small denominator)
    #fixme - which wreak havoc in likelihoods.  So instead set a range from [0.001,0.999], given typical population size
    #fixme - of 1000 individuals.
    def _correct_extremes(x):
        if x < 0.001:
            return 0.001
        elif x > 0.999:
            return 0.999
        else:
            return x

    combined_df['simulation'] = combined_df['simulation'].apply(_correct_extremes)

    combined_df["ll"] = binom_ll(combined_df["ref_bin_count"],
                                 combined_df["ref_total"],
                                 combined_df["simulation"])
    #print(combined_df)
    return combined_df["ll"].sum()

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
    
def compute_parasite_density_LL_by_site(site, numOf_param_sets):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "parasite_densities_by_age_month.csv"))
    
    
    
    combined_df_asex, combined_df_gamet = prepare_parasite_density_comparison_single_site(sim_df, site)
    
    asex_LL = combined_df_asex.groupby("param_set")\
        .apply(compute_density_likelihood)\
        .reset_index()\
        .rename(columns={0: "ll"})

        
    asex_LL, missing_param_sets_asex = identify_missing_parameter_sets(asex_LL, numOf_param_sets) 
  
    if len(missing_param_sets_asex) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets_asex} for asex parasite density')
              
    asex_LL["metric"] = "asex_density"
   
    
    gamet_LL = combined_df_gamet.groupby("param_set")\
        .apply(compute_density_likelihood)\
        .reset_index()\
        .rename(columns={0: "ll"})
   
    gamet_LL, missing_param_sets_gamet = identify_missing_parameter_sets(gamet_LL, numOf_param_sets) 
    
    if len(missing_param_sets_gamet) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets_gamet} for gamet parasite density')
        
    gamet_LL["metric"] = "gamet_density"  
    
    return_df = pd.concat([asex_LL, gamet_LL], ignore_index=True)
    
    return_df["site"] = site
        
    return return_df


def compute_parasite_density_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in density_sites:
        df_this_site = compute_parasite_density_LL_by_site(s, numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)


def plot_density_comparison_single_site(site, param_sets_to_plot=None, plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "parasite_densities_by_age_month.csv"))
    #sim_df = pd.read_csv(os.path.join('/projects/b1139/basel-hackathon-2023/simulations/output/3sites_240223/LF_0/SO',site,"parasite_densities_by_age_month.csv"))
    combined_df_asex, combined_df_gamet = prepare_parasite_density_comparison_single_site(sim_df, site)
    #print("COMBINED DF ASEX")
    #print(combined_df_asex[combined_df_asex['agebin']==5.0])
    #print("COMBINED DF GAMET")
    #print(combined_df_ga met[combined_df_gamet['agebin']==5.0])
    
    if param_sets_to_plot is None:
        param_sets_to_plot = list(set(combined_df_asex["param_set"]))

    #fixme hack
    # combined_df_asex = combined_df_asex[combined_df_asex["param_set"]==1]
    # combined_df_gamet = combined_df_gamet[combined_df_gamet["param_set"]==1]

    def _plot_parasite_type(parasite_type):
        if parasite_type == "asex":
            df = combined_df_asex
        elif parasite_type == "gamet":
            df = combined_df_gamet
        else:
            raise NotImplemented

        # Shift 0-density to small value to show plots on log scale
        df["densitybin"][df["densitybin"]==0] = 5
        ages = df["mean_age"].unique()
        #print(ages)
        months = df["month"].unique()
        #print(months)
        
        #print(len(ages)*len(months))
        #print(math.ceil(len(ages)*len(months)/3))
        plt.figure(figsize=(20,20))

        foo = df.groupby(["mean_age", "densitybin", "param_set","month"])\
            .agg({"ref_total": "sum",
                  "ref_bin_count": "sum",
                  "simulation": "mean"})\
            .reset_index()
        #print(foo)
        _confidence_interval_vectorized = np.vectorize(partial(proportion_confint, method="wilson"))
        foo["reference"] = foo["ref_bin_count"]/foo["ref_total"]
        foo["reference_low"], foo["reference_high"] = _confidence_interval_vectorized(foo["ref_bin_count"],
                                                                                      foo["ref_total"])
        fig, axs = plt.subplots(3,math.ceil(len(ages)*len(months)/3),figsize=(26,12),num=f"{parasite_type}_density_{site}")
        i = 0
        for a, age_month_df in foo.groupby(["mean_age","month"]):
            age=age_month_df.loc[age_month_df.index[0], 'mean_age']
            month=age_month_df.loc[age_month_df.index[0], 'month']
            have_plotted_ref = False
            for p, param_df in age_month_df.groupby("param_set"):
                if p in param_sets_to_plot:
                    if not have_plotted_ref:
                        #print(i%3)
                        #print(np.trunc(i/3))
                        axs[(i%3),int(np.trunc(i/3))].plot(param_df["densitybin"].to_numpy(), param_df["reference"].to_numpy(), label="ref" if i == 0 else "", marker='o')
                        axs[(i%3),int(np.trunc(i/3))].fill_between(param_df["densitybin"].to_numpy(),
                                         param_df["reference_low"].to_numpy(),
                                         param_df["reference_high"].to_numpy(),
                                         label="95% confidence" if i == 0 else "", alpha=0.2)
                        have_plotted_ref = True

                    axs[(i%3),int(np.trunc(i/3))].plot(param_df["densitybin"].to_numpy(), param_df["simulation"].to_numpy(), label=f"PS {p}" if i == 0 else "", marker='o')
                    axs[(i%3),int(np.trunc(i/3))].set_xscale("log")

                    n_ref_total = param_df["ref_total"].iloc[0]
                    axs[(i%3),int(np.trunc(i/3))].set_title(f"Age {age} Month {month}. Total N = {n_ref_total}")
                    axs[(i%3),int(np.trunc(i/3))].set_ylim([0,1])
                    axs[(i%3),int(np.trunc(i/3))].set_xlim([1,1e7])
                    axs[(i%3),int(np.trunc(i/3))].set_ylabel("Fraction")
                    axs[(i%3),int(np.trunc(i/3))].set_xlabel("Density Bin")
            #axs[(i%3),int(np.trunc(i/3))].xlabel("Parasite density")
            #axs[(i%3),int(np.trunc(i/3))].ylabel("Fraction")
            i += 1
        fig.suptitle(f"{site} - {parasite_type}")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        fig.legend(loc='lower center',ncol=4) 
        
        #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"density_{parasite_type}_{site}.png"))
        plt.savefig(os.path.join(plt_dir,f"density_{parasite_type}_{site}.png"))
        plt.savefig(os.path.join(plt_dir,f"density_{parasite_type}_{site}.pdf"))
        plt.close()

    for parasite_type in ["asex", "gamet"]:
        _plot_parasite_type(parasite_type)

def plot_density_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in density_sites:
        plot_density_comparison_single_site(s,param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)

if __name__ == "__main__":
    #plot_density_comparison_all_sites()
    #print(compute_parasite_density_LL_for_all_sites(numOf_param_sets=5))
    plot_density_comparison_all_sites(param_sets_to_plot = [1], plt_dir=os.path.join(manifest.simulation_output_filepath,"_plots"))
