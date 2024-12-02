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

exp_label = "test_severe_3MII"

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
        # if self.n == 0:
        #     Y0 = compute_LL_across_all_sites_and_metrics(numOf_param_sets=100)
        # else:
        #     Y0 = myFunc(X,wdir)
        Y0 = myFunc(X,wdir)
        Y1 = Y0
        
        if self.n == 0:
            Y0['round'] = [self.n] * len(Y0)
            Y0.to_csv(f"{self.workdir}/all_LL.csv",index=False)
        else:
            Y0['round'] = [self.n] * len(Y0)
            score_df=pd.read_csv(f"{self.workdir}/all_LL.csv")
            score_df=pd.concat([score_df,Y0])
            score_df.to_csv(f"{self.workdir}/all_LL.csv",index=False)
        
        ## Apply weights
        Y1['ll'] = (Y1['ll']) * (Y1['my_weight']) # weighting by general 'order' of baseline score
        #Y1['ll'] = Y1['ll']
        # Temporary fix to recognize that post-weighting zero (0) LL is bad
        Y1.loc[(Y1['metric'] == 'infectiousness') & (Y1['ll'] == 0), 'll'] = -10000
        Y1.loc[(Y1['metric'] == 'incidence') & (Y1['ll'] == 0), 'll'] = -10000
        Y1.loc[(Y1['metric'] == 'severe_incidence') & (Y1['ll'] == 0), 'll'] = -10000
        Y1.loc[(Y1['metric'] == 'prevalence') & (Y1['ll'] == 0), 'll'] = -10000
        Y1.loc[(Y1['metric'] == 'asex_density') & (Y1['ll'] == 0), 'll'] = -10000
        Y1.loc[(Y1['metric'] == 'gamet_density') & (Y1['ll'] == 0), 'll'] = -10000
        
        Y = Y1.groupby("param_set").agg({"ll": lambda x: x.sum(skipna=False)}).reset_index().sort_values(by=['ll'])
        #Ym = Y1.groupby("param_set").agg({"ll": lambda x: x.min(skipna=False)}).reset_index().sort_values(by=['ll'])
        params=Y['param_set']
        Y = Y['ll']
        #Ym=Ym['ll']
        # if self.n==0:
        #     # Mask score for team default X_prior
        #     Y[0]= float("nan")
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
            #clean_logs()
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
            #clean_logs()
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

                       
team_default_params = [0.08869166174886942, # Severe fever inverse width (27.5653580403806)
                       0.1,                 # Severe fever threshold (3.98354299722192)
                       0.32967622160487114, # Severe parasite inverse width (56.5754896048744)
                       0.55130716080761,    # Severe parasite threshold (851031.287744526)
                       0.834479208605029,   # Severe anemia inverse width (10)
                       0.565754896048744]   # Severe anemia threshold (4.50775824973078)   
                

bo.initRandom(init_size,
              n_batches = init_batches,
              Xpriors = [team_default_params])

# Run the optimization loop
bo.run()

##### Post-calibration steps

# Run analysis

post_calibration_analysis(experiment=exp_label,
                          length_scales_by_objective=True,              # Fit single-task GP per within-host site-metric
                          length_scales_by_environment_objective=False, # per environment_calibration score
                          length_scales_plot=False,                      # Plot length-scales from calibration
                          prediction_plot=False,exclude_count=0,         # Plot predictions, starting @ exclude_count
                          timer_plot=False)                              # Plot emulator and acquisition timing
