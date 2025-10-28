import os, sys, shutil
#sys.path.append('/projects/b1139/environments/e/lib/python3.8/site-packages/')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.stats import rankdata
from botorch.utils.transforms import unnormalize

from gpytorch.constraints import Interval, GreaterThan, LessThan

sys.path.append("../")
from calibration_common.batch_generators.expected_improvement import ExpectedImprovement
from calibration_common.batch_generators.turbo_thompson_sampling import TurboThompsonSampling
from calibration_common.batch_generators.batch_generator_array import BatchGeneratorArray

from calibration_common.emulators.GP import ExactGP, ExactMultiTaskGP
from calibration_common.bo import BO
from calibration_common.post_calibration_analysis import post_calibration_analysis

from my_func import my_func as myFunc
from compare_to_data.run_full_comparison import plot_all_comparisons
from compare_to_data.run_full_comparison import compute_LL_across_all_sites_and_metrics
from clean_all import clean_analyzers, clean_logs
from translate_parameters import translate_parameters

import manifest as manifest
import torch
from torch import tensor


torch.set_default_dtype(torch.float64)

exp_label = "ricky_test"
phase_0_exp = ""
phase = 0

output_dir = f"output/{exp_label}"
best_dir = f"output/{exp_label}" 


calib_coord = pd.read_csv(os.path.join(manifest.input_files_path,"calibration_coordinator.csv"),header=None)

# Botorch details
calib_coord.set_index(0, inplace=True)
init_size=int(calib_coord.at["init_size",1])
init_batches =  int(calib_coord.at["init_batches",1]) 
batch_size = int(calib_coord.at["batch_size",1])
max_eval = int(calib_coord.at["max_eval",1])
failure_limit = int(calib_coord.at["failure_limit",1])
success_limit = int(calib_coord.at["success_limit",1])

param_key=pd.read_csv("parameter_key.csv")
n_params=int(param_key.shape[0])


# # Define ECDF function
# def ecdf(values, y_train_ll):
#     combined_values = np.concatenate([values, y_train_ll])  # Combine 'values' from Y1 with 'll' from Y2
#     return rankdata(combined_values) / len(combined_values)  # ECDF based on combined values
# 
# # Calculate ECDF for Y1 using values from both Y1 and Y0
# def compute_ecdf_for_Y1(Y1, Y0):
#     # Initialize a list to store the transformed ECDF values for each row in Y1
#     ecdf_values = []
#     Y1=pd.concat([Y1,Y0])
#     # Group by 'site' and 'metric' and apply ECDF calculation to 'll' column
#     for (site, metric), group in Y1.groupby(['site', 'metric']):
#         # Get 'll' values from the current group in Y1
#         group_values = group['raw_ll'].values
#         #print("group_values")
#         #print(len(group_values))
#         # Get the corresponding 'll' values from Y0 for the same 'site' and 'metric'
#         y_train_ll = Y1[(Y1['site'] == site) & (Y1['metric'] == metric)]['raw_ll'].values
#         # Compute ECDF for this group using both Y1 and Y0 'll' values
#         if metric != "no_blood":
#             ecdf_values_group = ecdf(group_values, [])
#         else:
#             ecdf_values_group=group_values
#         # Add the ECDF values for the group to the list (to match the original number of rows in group)
#         #print("ecdf_values_group")
#         #print(len(ecdf_values_group))
#         ecdf_values.extend(ecdf_values_group[:len(group)])  # # # Only use ECDF values corresponding to the group size
#     # Assign the computed ECDF values to the 'll' column in Y1
#     Y1['ll'] = ecdf_values
#     Y1.loc[Y1['metric'] == 'no_blood', 'll'] = Y1.loc[Y1['metric'] == 'no_blood', 'raw_ll']
# 
#     Y1.loc[(Y1['metric']=="no_blood") & (Y1['ll']==0), 'll'] = 1
#     Y1.loc[(Y1['metric']=="no_blood") & (Y1['ll']<0), 'll'] = -1
#     Y1=Y1.reset_index(drop=True)
#     print(Y1)
#     # Ensure that the function returns the modified Y1 DataFrame
#     return Y1


def ecdf(values):
    # Combine 'values' from Y1 with 'll' from Y0 for ECDF calculation
    combined_values = values
    return rankdata(combined_values) / len(combined_values)  # ECDF based on combined values

def compute_ecdf_for_Y1(Y1, Y0):
    # Preserve the original index for Y1 and Y0 so that we can easily restore it later
    
    Y0['original_index'] = Y0.index
    Y1['original_index'] = Y1.index + len(Y0)
    # Concatenate Y1 and Y0 but keep track of their original indices
    combined_df = pd.concat([Y0, Y1])
    print(combined_df)
    # Initialize a list to store the transformed ECDF values for each row in the combined dataframe
    ecdf_values = []
    
    # Group by 'site' and 'metric' and calculate ECDF for the combined data
    for (site, metric), group in combined_df.groupby(['site', 'metric']):
      
        print(f"Converting raw {metric} LL to ECDF for {site}")
        # Get 'raw_ll' values from the current group
        #group_values = group['raw_ll'].values
        group_values = combined_df[(combined_df['site']==site) & (combined_df['metric']==metric)]['raw_ll'].values
        
        # Handle NaN values: replace NaNs with 0 for all metrics except 'no_blood', where it's -1
        if metric != "no_blood":
            group_values = np.nan_to_num(group_values, nan=0)  # Replace NaNs with 0 for non-'no_blood' metrics
        else:
            group_values = np.where(np.isnan(group_values), -1, group_values)  # Replace NaNs with -1 for 'no_blood'

        
        # Compute ECDF for this group using both Y1 and Y0 'raw_ll' values
        ecdf_values_group = ecdf(group_values)
        print(ecdf_values_group)
        print(type(ecdf_values_group))
        print(f"size: {len(group)}")
        print(f"range: {np.min(ecdf_values_group)}-{np.max(ecdf_values_group)}")
        #print(ecdf_values_group)
        #print(ecdf_values_group.shape)
        #print(len(group))
        # Append ECDF values for the group to the list
        ecdf_values.extend(ecdf_values_group[-len(group):])
        
        # Assign the computed ECDF values back to the combined dataframe
        combined_df.loc[(combined_df["metric"]==metric) & (combined_df['site']==site),'ecdf']=ecdf_values_group
        #print(combined_df.loc[(combined_df["metric"]==metric) & (combined_df['site']==site)].to_string())
        
    #combined_df['ecdf'] = ecdf_values
    # Now, instead of using 'isin()', directly align based on 'original_index'
    Y1['ll'] = combined_df.set_index('original_index').loc[Y1['original_index'], 'ecdf'].values
    Y0['ll'] = combined_df.set_index('original_index').loc[Y0['original_index'], 'ecdf'].values
    #print(Y0.shape)
    #print(Y0)
    # Specific adjustments for 'no_blood' metric cases
    # Only adjust 'll' values for 'no_blood' where 'raw_ll' is NaN
    Y1.loc[(Y1['metric'] == 'no_blood') & (Y1['raw_ll'].isna()), 'll'] = -1
    Y0.loc[(Y0['metric'] == 'no_blood') & (Y0['raw_ll'].isna()), 'll'] = -1

    # Assign 'raw_ll' to 'll' for 'no_blood' metric cases
    Y1.loc[Y1['metric'] == 'no_blood', 'll'] = Y1.loc[Y1['metric'] == 'no_blood', 'raw_ll']
    Y0.loc[Y0['metric'] == 'no_blood', 'll'] = Y0.loc[Y0['metric'] == 'no_blood', 'raw_ll']

    # Adjust ECDF for "no_blood" metric cases
    Y1.loc[(Y1['metric'] == "no_blood") & (Y1['ll'] == 0), 'll'] = 1
    Y1.loc[(Y1['metric'] == "no_blood") & (Y1['ll'] < 0), 'll'] = -1
    Y0.loc[(Y0['metric'] == "no_blood") & (Y0['ll'] == 0), 'll'] = 1
    Y0.loc[(Y0['metric'] == "no_blood") & (Y0['ll'] < 0), 'll'] = -1
    
    #print(Y0)
    #print(Y1)
    
    # Ensure that the function returns the modified Y1 and Y0 DataFrames
    return Y1, Y0

# Define the Problem, it must be a functor
class Problem:
    def __init__(self,workdir="checkpoints/emod"):
        self.dim = n_params  #mandatory dimension
        self.ymax = None #max value
        self.best = None
        self.n = 0
        self.workdir = workdir
        try:
            self.ymax = np.loadtxt(f"{self.workdir}/emod.ymax.txt").astype(float)
            self.n = np.loadtxt(f"{self.workdir}/emod.n.txt").astype(int)
        except IOError:
            self.ymax = None
            self.n = 0

        os.makedirs(os.path.relpath(f'{self.workdir}'), exist_ok=True)

    # The input is a vector that contains multiple set of parameters to be evaluated
    def __call__(self, X):
        # Each set of parameter x is evaluated
        # Note that parameters are samples from the unit cube in Botorch
        # Here we map unnormalizing them before calling the square function
        # Finally, because we want to minimize the function, we negate the return value
        # Y = [-myFunc(x) for x in unnormalize(X, [-5, 5])]
        # We return both X and Y, this allows us to disard points if so we choose
        # To remove a set of parameters, we would remove it from both X and Y

        # Finally, we need to return each y as a one-dimensional tensor (since we have just one dimension)
        # 
        # rewrite myfunc as class so we can keep track of things like the max value - aurelien does plotting each time but only saves when the new max > old max - would also allow for easier saving of outputs if desired. would also potentially help with adding iterations to param_set number so we don't reset each time. not sure yet if better to leave existing myfunc or pull everything into this
        param_key=pd.read_csv("parameter_key.csv")
        wdir=os.path.join(f"{self.workdir}/LF_{self.n}")
        os.makedirs(wdir,exist_ok=True) 
        if self.n>0:
            Y0=myFunc(X,wdir)
            #Y0=compute_LL_across_all_sites_and_metrics(5)
            Y0['round'] = [self.n] * len(Y0)
            Y0['raw_ll'] = Y0['ll']
            X0=torch.load(f"/projects/b1139/within-host-calibration/simulations/output/{exp_label}/X.pt")
            y_train = pd.read_csv(f"{self.workdir}/all_LL.csv")
            y_train = y_train[y_train['round']<self.n]
            y_new = Y0
            
            Y1,Y_train = compute_ecdf_for_Y1(y_new, y_train)
            #print(Y1.shape)
            
            
        else:
            if phase == 1:
                Y0=pd.read_csv(f'/projects/b1139/within-host-calibration/simulations/output/{phase_0_exp}/all_LL.csv')
                X0=torch.load(f'/projects/b1139/within-host-calibration/simulations/output/{phase_0_exp}/X.pt')
                X=torch.cat([X0,X])
                Y0['raw_ll']=Y0['ll']
                Y0['round'] = (Y0['round']*-1) - 1
                Y0['param_set']= Y0['param_set'] * -1
            else:
                Y0=myFunc(X,wdir)
                Y0['round'] = [self.n] * len(Y0)
                Y0['raw_ll']=Y0['ll']
            #if exp_label == "debug":
            #    X0=X0[[range(100)]]
            #    Y0=Y0[Y0['round']==0]
            #    Y0['raw_ll']=Y0['ll']
            #    Y0['round'] = (Y0['round']*-1) - 1
            #    Y0['param_set']= Y0['param_set'] * -1
            #    #print(X.shape)
            
            
            #print(Y0)
            #print(Y0.shape)
            column_names = Y0.columns
            y_empty = pd.DataFrame(columns=column_names)
            Y1,Y_train = compute_ecdf_for_Y1(y_empty, Y0)
            #print(Y1)
            #print(Y1.shape)
            #Y1=Y1[Y1['metric'] != "infectiousness"]
            
            
        ## Apply weights
        ########################################################################
        # in Phase 0 - unweighted #
        # if phase == 0:
        #     Y1['ll'] = Y1['ll']
        #     Y1['raw_ll']=Y1['ll']
        # elif phase == 1:
        #   # Weight as ecdf() vs. all observations plus phase0:
        #     y_train = pd.read_csv('/projects/b1139/within-host-calibration/simulations/output/250307_phase0_fixGarki/all_LL.csv')
        #     if(self.n>0):
        #         y_curr=pd.read_csv(f"{self.workdir}/all_LL.csv")
        #         y_train=pd.concat([y_train,y_curr],join='inner')
        #     Y1['raw_ll']=Y1['ll']
        #     y_train['raw_ll']=y_train['raw_ll']
        #     Y1 = compute_ecdf_for_Y1(Y1, y_train)
        #     Y1=Y1[Y1['metric'] != "infectiousness"]

        ########################################################################
        
        Y1 = pd.concat([Y_train,Y1])
        #print(Y1)
        #Y = Y1.groupby(["param_set","round"]).agg({"ll": lambda x: x.sum(skipna=False)}).sort_values(by=['round','param_set'],ascending=False).reset_index()
        Y1.to_csv(f"{self.workdir}/working_LL.csv",index=False)
        #print(Y1)
        #print(Y1['param_set'])
        #print(type(Y1['param_set']))
        #print(Y1['round'])
        #print(type(Y1['round']))
        
        Y1['param_set'] = Y1['param_set'].astype(int).values
        Y1['round'] = Y1['round'].astype(int).values
        Y1['ll']= Y1['ll'].astype(float)
        #print(Y1)
        Y1['param_set'] = Y1['param_set'].abs()
        Y = Y1.groupby(['round','param_set'])['ll'].sum().reset_index()
        #print(Y)
        print("passed")
        params=Y['param_set']
        Y = Y['ll']
        if self.n==0 and phase==0: #you only want to do this when you include the team default. (because it only has 3 max infections).
            #if you get errors due to miss match in comparisons its because of this.
            # Mask score for team default X_prior
            print(f"score to hide : {Y[0]}")
            Y[0]= float("nan")
            
        if os.path.exists(f"{self.workdir}/all_LL.csv"):
            #Y0['round'] = [self.n] * len(Y0)
            # score_df=pd.read_csv(f"{self.workdir}/all_LL.csv")
            # score_df=pd.concat([score_df,Y1])
            # score_df.to_csv(f"{self.workdir}/all_LL.csv",index=False)
            Y1.to_csv(f"{self.workdir}/all_LL.csv",index=False)
        else:
            #Y1['round'] = [self.n] * len(Y1)
            Y1.to_csv(f"{self.workdir}/all_LL.csv",index=False)
        
        
        #Entering the output phase:
        xc = []
        yc = []
        pc = []
        diff=len(Y)-len(X)      # How many of the parameter values are from BEFORE this round
        #print(f"Diff: {diff}")
        for j in range(len(Y)):
            if j == 0 and self.n == 0:
                continue
            elif pd.isna(Y[j]):
                continue
            else:
                yc.append(Y[j])
                pc.append(params[j])
                if j < diff:
                    #print("add filler param set to X")
                    #important to maintain shape of xc, but most scores arent used (aren't relevant)
                    xc.append(torch.rand(1,self.dim).tolist())
                else:
                    #print(f"xc: {xc}")
                    #print(f"x to add: {X[j-diff]}")
                    xc.append(X[j-diff].tolist())
        ##########################    
        to_keep=len(xc)
        xc2 = [tuple(item) if isinstance(item, list) else item for item in xc]
        yc = [tuple(item) if isinstance(item, list) else item for item in yc]
        xc2 = [tuple(item[0]) if isinstance(item[0], list) else item for item in xc2]
        # Flatten yc by extracting the scalar value from the tuple
        #yc = [item[0] for item in yc]
        # Check the updated xc2 and yc
        #print("xc2:", xc2)
        #print("yc:", yc)
        # Now you should be able to create the dictionary
        if self.n > 0:
            pre_x=xc2[:-batch_size]
            pre_y=yc[:-batch_size]
            pre_p=pc[:-batch_size]
            xc2=xc2[-batch_size:]
            yc=yc[-batch_size:]
            pc=pc[-batch_size:]
            pre_links=dict(zip(pre_x,pre_y))
            pre_pset=dict(zip(pre_p,pre_y))
            self.y_max= max(pre_links.values())
        links = dict(zip(xc2, yc))
        pset=dict(zip(pc,yc))
        
        if self.n > 0 :
            X_out = torch.tensor(xc[-batch_size:],dtype=torch.float64)   # tp append only this round's parameters
            #print(X_out.shape)
            Y_out = torch.tensor(yc[-batch_size:])                       # to append only this round's scores
            Y_out=Y_out.unsqueeze(1)
            torch.save(torch.tensor(yc[:-batch_size]).unsqueeze(1),f"{self.workdir}/Y.pt")          # Update scores from previous rounds
        else :
            #print(xc)
            X_out=torch.tensor(xc,dtype=torch.float64)
            Y_out=torch.tensor(yc)
            Y_out=Y_out.unsqueeze(1)
        #print(Y_out.shape)
      # If new best value is found, save it and some other data
        if self.n == 0:
            self.ymax = max(links.values())

            best_p = max(pset,key=pset.get)
            best_x = max(links,key=links.get)
            self.best = translate_parameters(param_key,best_x,ps_id=best_p)
            np.savetxt(f"{self.workdir}/emod.ymax.txt", [self.ymax])
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", [self.ymax])
            self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv",index=False)
            
            if(phase==0):
                plot_all_comparisons(param_sets_to_plot=[1],plt_dir=self.workdir)
                plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get),1],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
            shutil.copytree(f"{manifest.simulation_output_filepath}",f"{self.workdir}/LF_{self.n}/SO",dirs_exist_ok = True)            
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            clean_analyzers()
            #clean_logs()
        else: 
            if (max(links.values()) > self.ymax) or (self.n==1):
                best_p = max(pset,key=pset.get)
                
                best_x = max(links,key=links.get)
                self.best = translate_parameters(param_key,best_x,best_p)
                self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv",index=False)
                if(best_p>0):
                    plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get)],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
            self.ymax = max(links.values()) #weighted_lf
            np.savetxt(f"{self.workdir}/emod.ymax.txt", [self.ymax])
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", [self.ymax])
            shutil.copytree(f"{manifest.simulation_output_filepath}",f"{self.workdir}/LF_{self.n}/SO",dirs_exist_ok = True)
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            clean_analyzers()
        return X_out, Y_out


problem = Problem(workdir=f"output/{exp_label}")

# Delete everything and restart from scratch 
# Comment this line to restart from the last state instead
#if os.path.exists(output_dir): shutil.rmtree(output_dir)
#if os.path.exists(best_dir): shutil.rmtree(best_dir)

# at beginning of workflow, cleanup all sbatch scripts for analysis
clean_analyzers()

# Create the GP model
# See emulators/GP.py for a list of GP models
# Or add your own, see: https://botorch.org/docs/models
model = ExactGP(noise_constraint=GreaterThan(1e-6))

# Create batch generator(s)
tts = TurboThompsonSampling(batch_size=batch_size, failure_tolerance=failure_limit, 
                            success_tolerance=success_limit,dim=problem.dim)
batch_generator = tts 

# Create the workflow
bo = BO(problem=problem, model=model, batch_generator=batch_generator, checkpointdir=output_dir, max_evaluations=max_eval)

# Sample and evaluate sets of parameters randomly drawn from the unit cube
#bo.initRandom(2)

# centroid of unit parameter cube

param_center = [0.5] * n_params

### Current to parameter_key.csv as of March 7, 2025
###################################################

# # Usual random init sample, with team default Xprior

p0team_default = [#Max Individual Infections
                 -1,
                 #Antibody_Days_To_Long_Term_Decay of 365000.0
                 -1,
                 #Antibody_Long_Term_Decay_Days of 365.0
                 0.0,
                 #Antigen Switch Rate of 7.65e-10
                 0.470915,
                 #Falciparum MSP Variants of 32.0
                 0.28421052631578947,
                 #MSP Merozoite Kill Fraction of 0.511735322
                 0.3724510733333335,
                 #Falciparum Nonspecific Types of 76.0
                 0.7473684210526316,
                 #Nonspecific Antibody Growth Rate Factor of 0.5
                 0.494949494949495,
                 #Nonspecific Antigenicity Factor of 0.4151
                 0.39387500000000003,
                 #Falciparum PfEMP1 Variants of 1070.0
                 0.285,
                 #Antibody iRBC Kill Rate of 1.596
                 0.02742857142857145,
                 #Gametocyte Mosquito Stage Survival Rate of 0.002011099
                 0.4902,
                 #Cytokine Gametocyte Inactivation of 0.01667
                 0.452741,
                 #Max Fever Kill Rate of iRBCs of 1.4
                 0.073064,
                 #Pyrogenic Threshold of 15000.0
                 0.825707,
                 #InnateImmuneDistribution2 of 1.0
                 1.0,
                 #Maternal Antibody Protection of 0.1327
                 0.13183183183183184,
                 #Gametocyte Production Rate of 0.0615
                 0.662799,
                 #Gametocyte Fraction Male of 0.2
                 0.3877551020408163,
                 #Gametocyte Human Stage Survival Rate of 0.588569307
                 0.1968206822222222,
                 #Severe Anemia Threshold of 4.50775825
                 0.8031033000000001,
                 #Severe Anemia Inverse Width of 10.0
                 0.2,
                 #Severe Fever Threshold of 3.983542997
                 0.24588574924999995,
                 #Severe Fever Inverse Width of 27.56535804
                 0.5513071608,
                 #Severe Parasitemia Threshold of 851031.2877
                 0.9999950570878854,
                 #Severe Parasitemia Inverse Width of 56.5754896
                 0.9429248266666667,
                 #RBC Destruction Multiplier of 3.29
                 0.316,
                 #Erythropoiesis Anemia Effect of 3.5
                 0.4
                 ]

xprior = [p0team_default]
## add samples at unit centroid to learn noise
#xprior = x_prior + [param_center]*5
bo.initRandom(init_size,
              n_batches = init_batches,
              Xpriors = xprior)


# Run the optimization loop
bo.run()

##### Post-calibration steps

# Run analysis

post_calibration_analysis(experiment=exp_label,
                          length_scales_by_objective=True,              # Fit single-task GP per within-host site-metric
                          length_scales_by_environment_objective=False, # per environment_calibration score
                          length_scales_plot=False,                     # Plot length-scales from calibration
                          prediction_plot=False,exclude_count=0,        # Plot predictions, starting @ exclude_count
                          timer_plot=False,                             # Plot emulator and acquisition timing
                          n_prior=1)                                 # First n 'masked' priors used to seed calib without scores    


