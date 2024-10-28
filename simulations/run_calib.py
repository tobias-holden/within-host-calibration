import os, sys, shutil
#sys.path.append('/projects/b1139/environments/e/lib/python3.8/site-packages/')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from botorch.utils.transforms import unnormalize

from gpytorch.constraints import Interval, GreaterThan, LessThan

sys.path.append("../")
from calibration_common.batch_generators.expected_improvement import ExpectedImprovement
from calibration_common.batch_generators.turbo_thompson_sampling import TurboThompsonSampling
from calibration_common.batch_generators.batch_generator_array import BatchGeneratorArray

from calibration_common.emulators.GP import ExactGP, ExactMultiTaskGP
from calibration_common.bo import BO

from my_func import my_func as myFunc
from compare_to_data.run_full_comparison import plot_all_comparisons
from compare_to_data.run_full_comparison import compute_LL_across_all_sites_and_metrics
from clean_all import clean_analyzers, clean_logs
from translate_parameters import translate_parameters

import manifest as manifest
import torch
from torch import tensor

from post_calibration_analysis import post_calibration_analysis

torch.set_default_dtype(torch.float64)

exp_label = "test_241028"

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

# Define the Problem, it must be a functor
class Problem:
    def __init__(self,workdir="checkpoints/emod"):
        self.dim = int(param_key.shape[0])  #17 # mandatory dimension
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

        if self.n > 0 :
            Y0 = myFunc(X,wdir)
        else: 
            Y0 = compute_LL_across_all_sites_and_metrics(numOf_param_sets=250)
            
        
        Y1 = Y0
        
        if self.n == 0:
            Y0['round'] = [self.n] * len(Y0)
            Y0.to_csv(f"{self.workdir}/all_LL.csv",index=False)
        else:
            Y0['round'] = [self.n] * len(Y0)
            score_df=pd.read_csv(f"{self.workdir}/all_LL.csv")
            score_df=pd.concat([score_df,Y0])
            score_df.to_csv(f"{self.workdir}/all_LL.csv",index=False)
        
        Y1['ll'] = (Y1['ll']  / (Y1['baseline'])) * (Y1['my_weight']) 
        
        # Temporary fix to recognize that post-weighting zero (0) LL is bad
        Y1.loc[(Y1['metric'] == 'infectiousness') & (Y1['ll'] == 0), 'll'] = -10
        Y1.loc[(Y1['metric'] == 'incidence') & (Y1['ll'] == 0), 'll'] = -10
        Y1.loc[(Y1['metric'] == 'severe_incidence') & (Y1['ll'] == 0), 'll'] = -10
        Y1.loc[(Y1['metric'] == 'prevalence') & (Y1['ll'] == 0), 'll'] = -10
        Y1.loc[(Y1['metric'] == 'asex_density') & (Y1['ll'] == 0), 'll'] = -10
        Y1.loc[(Y1['metric'] == 'gamet_density') & (Y1['ll'] == 0), 'll'] = -10
        
        Y = Y1.groupby("param_set").agg({"ll": lambda x: x.sum(skipna=False)}).reset_index().sort_values(by=['ll'])
        #Ym = Y1.groupby("param_set").agg({"ll": lambda x: x.min(skipna=False)}).reset_index().sort_values(by=['ll'])
        params=Y['param_set']
        Y = Y['ll']
        #Ym=Ym['ll']
        if self.n==0:
            # Mask score for team default X_prior
            Y[0]= float("nan")
         #   Ym[0]= float("nan")
            
        xc = []
        yc = []
        #ym = []
        ysc = []
        pc = []
        
        for j in range(len(Y)):
            if pd.isna(Y[j]):
                continue
            else:
                xc.append(X[j].tolist())
                yc.append([Y[j]])
                #ym.append([Ym[j]])
                sub=Y1[Y1['param_set']==params[j]]
                ysc.append(sub['ll'].to_list())
                pc.append(params[j])
        
        xc2=[tuple(i) for i in xc]
        links=dict(zip(xc2,yc)) 
        pset=dict(zip(pc,yc))
        #links_m=dict(zip(xc2,ym))
        #pset_m=dict(zip(pc,ym))
        
        X_out = torch.tensor(xc,dtype=torch.float64)
        #print("X_out")
        #print(X_out)
        
        Y_out = torch.tensor(yc)
        #Y_m_out = torch.tensor(ym)
        #Y_out = torch.stack([torch.tensor(y) for y in ysc],-1)
        #print("Y_out")
        #print(Y_out)
        #print(Y_out.shape)
        #print("Y_m_out")
        #print(Y_m_out)
        #print(Y_m_out.shape)

        # If new best value is found, save it and some other data
        if self.ymax is None or self.n == 0:
            self.ymax = max(links.values())
            
            best_p = max(pset,key=pset.get)
            best_x = max(links,key=links.get)
            self.best = translate_parameters(param_key,best_x,ps_id=best_p)
            
            np.savetxt(f"{self.workdir}/emod.ymax.txt", [self.ymax])
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", [self.ymax])
            self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv",index=False)
            plot_all_comparisons(param_sets_to_plot=[1],plt_dir=self.workdir)
            plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get),1],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
            shutil.copytree(f"{manifest.simulation_output_filepath}",f"{self.workdir}/LF_{self.n}/SO",dirs_exist_ok = True)            
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            clean_analyzers()
            clean_logs()
        else: 
            if max(links.values())[0] > self.ymax:
                self.ymax = max(links.values()) #weighted_lf  
                best_p = max(pset,key=pset.get)
                best_x = max(links,key=links.get)
                self.best = translate_parameters(param_key,best_x,best_p)
                self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv",index=False)

                plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get)],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
              
            np.savetxt(f"{self.workdir}/emod.ymax.txt", [self.ymax])
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", [self.ymax])
            shutil.copytree(f"{manifest.simulation_output_filepath}",f"{self.workdir}/LF_{self.n}/SO",dirs_exist_ok = True)
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            clean_analyzers()
            clean_logs()
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

# Usual random init sample, with team default Xprior
                       
team_default_params = [0.235457679394, # Antigen switch rate (7.65E-10) 
                       0.166666666667,  # Gametocyte sex ratio (0.2) 
                       0.236120668037,  # Base gametocyte mosquito survival rate (0.00088) **
                       0.394437557888,  # Base gametocyte production rate (0.0615)
                       0.50171665944,   # Falciparum MSP variants (32)
                       0.0750750750751, # Falciparum nonspecific types (76)
                       0.704339142192,  # Falciparum PfEMP1 variants (1070)
                       0.28653200892,   # Fever IRBC kill rate (1.4)
                       0.584444444444,  # Gametocyte stage survival rate (0.5886)
                       0.506803355556,  # MSP Merozoite Kill Fraction (0.511735)
                       0.339794000867,  # Nonspecific antibody growth rate factor (0.5)  
                       0.415099999415,  # Nonspecific Antigenicity Factor (0.4151) 
                       0.492373751573,  # Pyrogenic threshold (15000)
                       -1.0,            # Max Individual Infections (3)
                       0.666666666666,  # Erythropoesis Anemia Effect Size (3.5)
                       0.755555555555,  # RBC Destruction Multiplier (3.9)
                       0.433677]        # Cytokine Gametocyte Inactivation (0.02)

params_241013 = [0.063819259,
                0.311834632,
                0.265195263,
                0.709026171,
                0.159035939,
                0.176874946,
                0.733520807,
                0.976217168,
                0.833900254,
                0.449908713,
                0.694589051,
                0.129187744,
                0.294669431,
                1.000000000, # Max Individual Infections (20)
                0.410575954,
                0.391240481,
                0.199995755]

bo.initRandom(init_size,
              n_batches = init_batches,
              Xpriors = [team_default_params,params_241013])

# Run the optimization loop
bo.run()

##### Post-calibration steps

# Run analysis

# post_calibration_analysis(experiment=exp_label,
#                           length_scales_by_objective=True,      # Fit single-task GP per site-metric
#                           length_scales_plot=True,              # Plot length-scales from calibration
#                           prediction_plot=True,exclude_count=0, # Plot predictions, starting @ exclude_count
#                           timer_plot=True)                      # Plot emulator and acquisition timing
