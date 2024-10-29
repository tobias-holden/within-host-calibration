import os
import sys
sys.path.append('../')
import warnings
import manifest
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import math
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
from scipy.stats import binom
from cycler import cycler
sys.path.append('../../')
from calibration_common.create_plots.helpers_reformat_sim_ref_dfs import get_fraction_in_infectious_bin
from simulations import manifest
from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

infectiousness_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'infectiousness_to_mosquitos'] == 1 :
        infectiousness_sites.append(site)
print(f"Infectiousness Sites: {infectiousness_sites}")
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)


def prepare_infectiousness_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with an prevalence-by-age validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Process simulation data
    #fixme sample_number column will be renamed based on upstream analyzer output


    # Load reference data
    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['infectiousness_to_mosquitos_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df.rename(columns={'site': 'Site'}, inplace=True)
    ref_df = ref_df[ref_df['Site'].str.lower() == str(site).lower()]
    ref_df['Site'] = ref_df['Site'].str.lower()
    ref_months = ref_df['month'].unique()
    ref_ages = ref_df['agebin'].unique()
    # subset simulation to months in reference df
    sim_df = sim_df[sim_df['month'].isin(ref_months)]
    sim_df = sim_df[sim_df['agebin'].isin(ref_ages)]
    inf_df = sim_df
    
    inf_df = (inf_df[inf_df['month'].isin(ref_df['month'].unique()) & inf_df['agebin'].isin(ref_df['agebin'].unique())])

    inf_df = inf_df[inf_df['Pop'] == inf_df['Pop'].max()]
    
    inf_df['counts'] = inf_df['infectiousness_bin_freq'] * inf_df['Pop']
    
    inf_df = inf_df.groupby(['Site', 'param_set', 'month', 'agebin', 'densitybin', 'infectiousness_bin'], as_index=False).agg({'counts': 'sum'})
    
    inf_df['total_count'] = inf_df.groupby(['Site', 'param_set', 'month', 'agebin'])['counts'].transform('sum')

    inf_df['freq_frac_infect'] = inf_df['counts'] / inf_df['total_count']
    
    sim_df=inf_df
    
    sim_df['Site'] = str(site).lower()
        
    # standardize column names and merge simulation and reference data frames
    #print(ref_df.columns)
    #print(("year" in ref_df.columns))
    
    # if "year" in ref_df.columns:
    #     #print(f"A: {site}")
    #     min_yr = np.min(ref_df['year'])
    #     ref_df['year'] = ref_df['year']-min_yr+1
    #     #print(ref_df)
    # if not "year" in ref_df.columns:
    #     #print(f"B: {site}")
    #     ref_df.insert(0,'year',1)
    #     #print(ref_df)
  
    #ref_df["site_month"] = ref_df['Site'] + '_year' + ref_df['year'].astype('str') + '_month' + ref_df['month'].astype('str')
    ref_df["site_month"] = ref_df['Site'] + '_month' + ref_df['month'].astype('str')
    ref_df = ref_df[["freq_frac_infect", "agebin", "densitybin", "fraction_infected_bin", "Site", "month", "site_month",
                     "num_in_group", "count"]]
    ref_df.rename(columns={"freq_frac_infect": "reference",
                           "num_in_group": "ref_total",
                           "count": "ref_bin_count"},
                  inplace=True)

    #sim_df_by_param_set["site_month"] = sim_df_by_param_set['Site'] + '_year' + sim_df_by_param_set['year'].astype('str') + '_month' + sim_df_by_param_set['month'].astype('str')
    sim_df["site_month"] = sim_df['Site'] + '_month' + sim_df['month'].astype('str')
    sim_df.rename(columns={"freq_frac_infect": "simulation",
                           "infectiousness_bin": "fraction_infected_bin"},
                           inplace=True)
  
    ref_df['agebin'] = [int(x) for x in ref_df['agebin']]
    sim_df['agebin'] = [int(x) for x in sim_df['agebin']]
    #print(ref_df)
    #print(sim_df_by_param_set)
    #print(ref_df.columns)
    #print(sim_df_by_param_set.columns)
    #fixme - Note we are dropping nans in both reference and simulation.  Dropping nans in simulation might not be best
    combined_df = pd.merge(sim_df, ref_df, how='outer')#.dropna(subset=["reference"])#, "simulation"])
    #print(combined_df)
    combined_df['metric'] = 'infectiousness'
    #print(combined_df)
    #fixme - Fudge to correct for sim infectiousness values of 1s and 0s (often because of small denominator)
    #fixme - which wreak havoc in likelihoods.  So instead set a range from [0.001,0.999], given typical population size
    #fixme - of 1000 individuals.
    def _correct_extremes(x):
        if x < 0.001:
            return 0.001
        elif x > 0.999:
            return 0.999
        else:
            return x

    combined_df.loc[np.isnan(combined_df['simulation']),'simulation'] = 0.0
    combined_df['simulation'] = combined_df['simulation'].apply(_correct_extremes)
    
    #print(combined_df)
    #print(combined_df)
    return combined_df


def compute_infectiousness_likelihood(combined_df):
    """
    Calculate an approximate likelihood for the simulation parameters for each site. This is estimated as the product,
    across age groups, of the probability of observing the reference values if the simulation means represented the
    true population mean
    Args:
        combined_df (): A dataframe containing both the reference and matched simulation output
        sim_column (): The name of the column of combined_df to use as the simulation output
    Returns: A dataframe of loglikelihoods where each row corresponds to a site-month

    """

    #fixme Jaline and Prashanth used dirichlet_multinomial for infectiousness
    #fixme 230328: JS changed to a simplified approach: naively assume every observation is independent.
    # Likelihood of each observation is likelihood of seeing reference data if simulation is "reality"
    binom_ll = np.vectorize(binom.logpmf) # probability mass function of binomial distribution

    combined_df["ll"] = binom_ll(combined_df["ref_bin_count"],
                                 combined_df["ref_total"],
                                 combined_df["simulation"])
    #print(combined_df)
    return combined_df["ll"].sum()#mean()#

# The following function determines whether any parameters sets were missing for a site,
# if there are missing parameter set, this prepares compute_LL_by_site to shoot out a warning message
# This function additionally adds the missing parameter set to the dataframe with NaN for the ll.
def identify_missing_parameter_sets(combined_df, numOf_param_sets):
    param_list = list(range(1,numOf_param_sets+1))
    #print(param_list)
    #print(combined_df.to_string())
    months=combined_df['month'].unique()
    ages=combined_df['agebin'].unique()
    missing_param_sets = []
    for x in param_list:
        if x not in combined_df['param_set'].values:
            for age in ages:
                for month in months:
                    combined_df.loc[len(combined_df.index)] = [x,month,age,np.nan]
            missing_param_sets.append(x)
    return combined_df, missing_param_sets
    
def compute_infectiousness_LL_by_site(site,numOf_param_sets):
    #print(site)
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "infectiousness_by_age_density_month.csv"))
    #sim_df = pd.read_csv(os.path.join(manifest.PROJECT_DIR,"simulations/output/8site_masked3/LF_5/SO", site, "infectiousness_by_age_density_month.csv"))
    #print(sim_df[sim_df['param_set']==90])

    #print(sim_df)
    combined_df = prepare_infectiousness_comparison_single_site(sim_df, site)
    
    #print(cc.to_string())
    ll_by_param_set = combined_df.groupby(["param_set","month","agebin"]) \
        .apply(compute_infectiousness_likelihood) \
        .reset_index() \
        .rename(columns={0: "ll_spec"})
        
    #print(ll_by_param_set)

    ll_by_param_set, missing_param_sets = identify_missing_parameter_sets(ll_by_param_set, numOf_param_sets)
    
    ll_by_param_set["metric"] = "infectiousness"
    ll_by_param_set["site"] = site
    
    
    if len(missing_param_sets) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets} for infectiousness')
    return ll_by_param_set


def compute_infectious_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in infectiousness_sites:
        df_this_site = compute_infectiousness_LL_by_site(s, numOf_param_sets)
        df_this_site = df_this_site.groupby(["param_set","site","metric"])['ll_spec'] \
            .apply("sum") \
            .reset_index() \
            .rename(columns={"ll_spec":"ll"})
        df_by_site.append(df_this_site)
        
    return pd.concat(df_by_site)


def plot_infectiousness_comparison_single_site(site, 
                                               param_sets_to_plot=None, 
                                               plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "infectiousness_by_age_density_month.csv"))
    #sim_df = pd.read_csv(os.path.join('/projects/b1139/basel-hackathon-2023/simulations/output/3sites_240223/LF_0/SO',site,"infectiousness_by_age_density_month.csv"))
    combined_df = prepare_infectiousness_comparison_single_site(sim_df, site)
    #print(combined_df)
    #fixme hack
    #combined_df = combined_df[combined_df["param_set"] == 1]
    

    if param_sets_to_plot is None:
        param_sets_to_plot = list(set(combined_df["param_set"]))

    #todo Add error bars on data

    # plt.figure(dpi=200, figsize=(18,8))

    combined_df["densitybin"][combined_df["densitybin"]==0] = 0.05
    age_bins = combined_df["agebin"].unique()
    dens_bins = combined_df["densitybin"].unique()
    frac_infected_bins = combined_df["fraction_infected_bin"].unique()
    month_bins = combined_df["month"].unique()

    n_age_bins = len(age_bins)
    n_month_bins = len(month_bins)
    n_frac_infected_bins = len(frac_infected_bins)
    
    n_subplots = n_age_bins * n_month_bins
    #print(n_subplots)
    i = 0
    fig, axs = plt.subplots(3,math.ceil(n_subplots/3),figsize=(16,14),num=f"infectiousness_{site}")
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_cycler = cycler(color=colors[1:])
    for a in age_bins:
        for m in month_bins:
            subplot_df = combined_df[np.logical_and(combined_df["agebin"]==a,
                                                    combined_df["month"]==m)].dropna(subset=["param_set"])
            #print(i)
            axs[(i%3),int(np.trunc(i/3))].set_prop_cycle(custom_cycler)
            param_sets_to_plot = sorted(param_sets_to_plot)
            for p in param_sets_to_plot:
                sdf = subplot_df[subplot_df['param_set']==p]
                sdf.loc[sdf['reference'].isnull(),'reference'] = 0.001
                sdf.loc[sdf['reference']==0.0,'reference'] = 0.001
                sdf_r=sdf[sdf['reference']>0.001]
                sdf_s=sdf[sdf['simulation']>0.001]
                #print(sdf.to_string())
                axs[(i%3),int(np.trunc(i/3))].scatter([np.log10(d) for d in sdf_s["densitybin"].to_numpy()], [str(f) for f in sdf_s["fraction_infected_bin"].to_numpy()], label=f"PS {p}" if i == 0 else "",marker='o',edgecolors='none',s=[sim * 200 for sim in sdf_s['simulation'].to_numpy()],alpha=0.5)
               
            #sdf = sdf.dropna(subset = ['reference'])
            axs[(i%3),int(np.trunc(i/3))].scatter([np.log10(d) for d in sdf_r["densitybin"].to_numpy()], [str(f) for f in sdf_r["fraction_infected_bin"].to_numpy()], label='ref' if i == 0 else "",marker='o',facecolors='none', edgecolors='k',s=[ref * 200 for ref in sdf_r['reference'].to_numpy()],alpha=1.0)
            #axs[(i%3),int(np.trunc(i/3))].set_xscale("log")
            axs[(i%3),int(np.trunc(i/3))].set_title(label=f"Age: {a} Month: {m}")
            axs[(i%3),int(np.trunc(i/3))].set_xlabel("Gametocye Density")
            axs[(i%3),int(np.trunc(i/3))].set_ylabel("Infectiousness Level")
            i+=1
            
    fig.suptitle(f"{site} - Infectiousness")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.legend(loc='lower center',ncol=4) 
    plt.show()

    plt.savefig(os.path.join(plt_dir,f"infectious_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"infectious_{site}.pdf"))
    plt.close()

def plot_infectiousness_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in infectiousness_sites:
        plot_infectiousness_comparison_single_site(s, param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
        

if __name__=="__main__":
    # cc=compute_infectiousness_LL_by_site(site="laye_2007",numOf_param_sets=1000)
    # print(cc.to_string())
    # print(cc.sort_values(by=['param_set']).to_string())
    # cc=cc.groupby(["param_set","site","metric"])['ll_spec'] \
    #          .apply("sum") \
    #          .reset_index() \
    #          .rename(columns={"ll_spec":"ll"})
    # print(cc.sort_values(by=['ll']).to_string())
    sim_df = pd.read_csv(os.path.join("/projects/b1139/within-host-calibration/simulations/output/test_241027/LF_0/SO/laye_2007/infectiousness_by_age_density_month.csv"))
    print(prepare_infectiousness_comparison_single_site(sim_df,'laye_2007'))
    # 
    #plot_infectiousness_comparison_all_sites(param_sets_to_plot=[92])
    #plot_infectiousness_comparison_all_sites(param_sets_to_plot=[1,532],
    #                                          plt_dir=os.path.join(manifest.PROJECT_DIR,'simulations/output/8site_masked3/LF_0'))
    # plot_infectiousness_comparison_all_sites(param_sets_to_plot=[32],
    #                                         plt_dir=os.path.join(manifest.PROJECT_DIR,'simulations/output/8site_masked3/LF_1'))
    #print(compute_infectious_LL_for_all_sites(100).sort_values(by=["ll"]).to_string())
