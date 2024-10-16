import os, sys, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from botorch.utils.transforms import unnormalize

from gpytorch.constraints import Interval, GreaterThan, LessThan

sys.path.append("../")
from batch_generators.expected_improvement import ExpectedImprovement
from batch_generators.turbo_thompson_sampling import TurboThompsonSampling
from batch_generators.batch_generator_array import BatchGeneratorArray

from emulators.GP import ExactGP
from bo import BO
from plot import *

from my_func import my_func as myFunc
from compare_to_data.run_full_comparison import plot_all_comparisons
from compare_to_data.run_full_comparison import compute_LL_across_all_sites_and_metrics
from clean_all import clean_analyzers
from translate_parameters import translate_parameters

# Define the square function in one dimension
#def myFunc(x):
#    return x**2 

# Define the Problem, it must be a functor
class Problem:
    def __init__(self,workdir="checkpoints/emod"):
        self.dim = 15 # mandatory dimension
        self.ymax = None #max value
        self.best = None
        self.n = 0
        self.workdir = workdir
        #print(f"{self.workdir}/emod.ymax.txt")
        try:
            self.ymax = np.loadtxt(f"{self.workdir}/emod.ymax.txt").astype(float)
            self.n = np.loadtxt(f"{self.workdir}/emod.n.txt").astype(int)
            #self.best = np.loadtxt(f"{self.workdir}/emod.best.txt").astype(float)
        except IOError:
            self.ymax = None
            self.n = 0
            #self.best = None
            
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
        #rewrite myfunc as class so we can keep track of things like the max value - aurelien does plotting each time but only saves when the new max > old max - would also allow for easier saving of outputs if desired. would also potentially help with adding iterations to param_set number so we don't reset each time. not sure yet if better to leave existing myfunc or pull everything into this
        param_key=pd.read_csv("test_parameter_key.csv")
        wdir=os.path.join(f"{self.workdir}/LF_{self.n}")
        os.makedirs(wdir,exist_ok=True)
        if os.path.exists(f"{wdir}/all_LL.csv"):
            Y1 = pd.read_csv(f"{wdir}/all_LL.csv")
        else:
            Y1=myFunc(X,wdir)
            #Y1 = compute_LL_across_all_sites_and_metrics(numOf_param_sets=50)
        Y1=myFunc(X,wdir)
        Y = Y1.groupby("param_set").agg({"ll": lambda x: x.sum(skipna=False)}).reset_index().sort_values(by=['ll'])
        params=Y['param_set']
        Y=Y['ll']
        xc = []
        yc = []
        pc = []
        for j in range(len(Y)):
            if pd.isna(Y[j]):
                continue
            else:
                xc.append(X[j].tolist())
                yc.append([Y[j]])
                pc.append(params[j])
                
        xc2=[tuple(i) for i in xc]
        links=dict(zip(xc2,yc)) 
        pset=dict(zip(pc,yc))
        print(max(links.values())[0])
        print(self.ymax)
        print(max(pset,key=pset.get))
        # If new best value is found, save it and some other data
        if self.ymax is None:
            self.ymax = max(links.values())
            best_p = max(pset,key=pset.get)
            best_x = max(links,key=links.get)
            #print(best_x)
            self.best = translate_parameters(param_key,best_x,ps_id=best_p)
            #print(self.best)
            # os.mkdir(os.path.join(f"{self.workdir}/LF_{self.n}"))
            #shutil.copytree(f'{self.workdir}/output/job_{i}', f'{self.workdir}/LF_{self.n}')
            
            np.savetxt(f"{self.workdir}/emod.ymax.txt", self.ymax)
            
            #np.savetxt(f"{self.workdir}/emod.best.txt", self.best)
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", self.ymax)
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.best.txt", self.best)
            self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv")
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            Y1.to_csv(f"{self.workdir}/LF_{self.n}/all_LL.csv")
              
            plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get),1],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
            self.n += 1
            #np.savetxt(f"{self.workdir}/LF_{self.n-1}/emod.n.txt", [self.n])
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            
            
        else: 
            if max(links.values())[0] > self.ymax:
                #print("Here")
                self.ymax = max(links.values()) #weighted_lf  
                best_p = max(pset,key=pset.get)
                best_x = max(links,key=links.get)
                #print(best_x)
                self.best = translate_parameters(param_key,best_x,best_p)
                self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv")
                plot_all_comparisons(param_sets_to_plot=[best_p],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))

            #print(self.ymax)    
            #np.savetxt(f"{self.workdir}/emod.ymax.txt", self.ymax)
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", self.ymax)
                #print(self.best)
                       
            #shutil.copytree(f'{self.workdir}/output/job_{i}', f'{self.workdir}/LF_{self.n}')
            # os.mkdir(os.path.join(f"{self.workdir}/LF_{self.n}"))
            
            #np.savetxt(f"{self.workdir}/emod.best.txt", self.best)
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", self.ymax)
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.best.txt", self.best)
            
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            Y1.to_csv(f"{self.workdir}/LF_{self.n}/all_LL.csv")
            
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            #np.savetxt(f"{self.workdir}/test.txt", ["Testing 1,2,3"])
            #np.savetxt(f"{self.workdir}/LF_{self.n-1}/emod.n.txt", [self.n])
        return torch.tensor(xc,dtype=torch.float64), torch.tensor(yc)

output_dir = "output/8site_big2"

problem = Problem(workdir="output/8site_big2")
# Create the GP model
# See emulators/GP.py for a list of GP models
# Or add your own, see: https://botorch.org/docs/models
model = ExactGP(noise_constraint=GreaterThan(1e-6))
# Create and combine multiple batch generators
#batch_size 64 when running in production
tts = TurboThompsonSampling(batch_size=100, failure_tolerance=4, dim=problem.dim) #64
#ei = ExpectedImprovement(batch_size=50, num_restarts=20, raw_samples=1024, dim=problem.dim)
batch_generator = tts#ei#BatchGeneratorArray([tts, ei])

timer_model = torch.load(f"{output_dir}/timer_model.pt")
timer_batch_generator = torch.load(f"{output_dir}/timer_batch_generator.pt")
iterations = torch.load(f"{output_dir}/iterations.pt")
length_scales=torch.load(f"{output_dir}/length_scales.pt")
X=torch.load(f"{output_dir}/X.pt")
Y=torch.load(f"{output_dir}/Y.pt")
Y_pred_mean=torch.load(f"{output_dir}/Y_pred_mean.pt")
Y_pred_var=torch.load(f"{output_dir}/Y_pred_var.pt")
# Create the workflow
bo = BO(problem=problem, model=model, batch_generator=batch_generator, checkpointdir=output_dir, max_evaluations=1000)
bo.iterations = iterations
bo.length_scales=length_scales
bo.X = X
bo.Y = Y
bo.Y_pred_mean = Y_pred_mean
bo.Y_pred_var = Y_pred_var
# ...
x=pd.read_csv("test_parameter_key.csv")
parameter_labels=x['parameter_label'].to_list()
print("Plotting...")
# Plot
# plot_runtimes(bo)
# plt.savefig(f'{output_dir}/runtime', bbox_inches="tight")
# plot_MSE(bo,n_init=1)
# plt.savefig(f'{output_dir}/mse', bbox_inches="tight")
plot_convergence(bo, negate=True)
plt.savefig(f'{output_dir}/convergence', bbox_inches="tight")
plot_prediction_error(bo)
plt.savefig(f'{output_dir}/pred_error', bbox_inches="tight")
plot_X_flat(bo, param_key = x, labels=parameter_labels)
plt.savefig(f'{output_dir}/x_flat', bbox_inches="tight")
#plot_space(bo, -5**2, 0, labels="X")
#plt.savefig(f'{output_dir}/space', bbox_inches="tight")
#plot_y_vs_posterior_mean(bo,n_init=1)
#plt.savefig(f'{output_dir}/posterior_mean', bbox_inches="tight")
