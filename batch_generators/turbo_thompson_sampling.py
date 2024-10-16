import dataclasses, json
import json
import os
import math
import numpy as np
import pdb
import torch

import gc
from collections import Counter

from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.acquisition.objective import IdentityMCObjective, GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from dataclasses import dataclass
#from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler

from torch.quasirandom import SobolEngine

import pathlib

from .batch_generator import BatchGenerator 

import gpytorch

torch.set_default_dtype(torch.float64)


@dataclass
# class TurboThompsonSampling():
class TurboThompsonSampling(BatchGenerator):
    dim: int
    batch_size: int
    length: float = 1.0
    length_min: float =2**-30
    length_max: float = 1.0
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 8 # 10 #10  # Note: The original paper uses 3
    y_max: float = -float("inf")
    y1_max: float = -float("inf")
    y2_max: float = -float("inf")
    y3_max: float = -float("inf")
    y4_max: float = -float("inf")
    y5_max: float = -float("inf")
    y6_max: float = -float("inf")
    y7_max: float = -float("inf")
    y8_max: float = -float("inf")
    y9_max: float = -float("inf")
    y10_max: float = -float("inf")
    y11_max: float = -float("inf")
    y12_max: float = -float("inf")
    y13_max: float = -float("inf")
    y14_max: float = -float("inf")
    y15_max: float = -float("inf")
    y16_max: float = -float("inf")
    y17_max: float = -float("inf")
    y18_max: float = -float("inf")
    y19_max: float = -float("inf")
    y20_max: float = -float("inf")
    y21_max: float = -float("inf")
    y22_max: float = -float("inf")
    y23_max: float = -float("inf")
    y24_max: float = -float("inf")
    y25_max: float = -float("inf")
    y26_max: float = -float("inf")


    n_candidates: int = None
    dtype = None
    device = None

    def __init__(self, dim=1, batch_size=4, failure_tolerance = None, n_candidates=None, objective=IdentityMCMultiOutputObjective(), dtype=torch.double, device=torch.device("cpu"), length_min=2**-30):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.dtype = dtype
        self.device = device
        self.length_min = length_min
        self.objective = objective
        self.length = 1.0

        if failure_tolerance is None:
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        else:
            self.failure_tolerance = failure_tolerance

        if self.n_candidates is None:
            self.n_candidates = min(5000, max(2000, 200 * self.dim))

    def update(self, X, Y):
        Y = torch.transpose(Y,0,1)
        print(Y.shape)
        # print(self.failure_counter, self.failure_tolerance)
        if Y.dim()>1:
            print("Multi Obj.", flush=True)
            # y1=[y[0] for y in Y]
            # y2=[y[1] for y in Y]
            # y3=[y[2] for y in Y]
            # y4=[y[3] for y in Y]
            # y5=[y[4] for y in Y]
            # y6=[y[5] for y in Y]
            # y7=[y[6] for y in Y]
            # y8=[y[7] for y in Y]
            # y9=[y[8] for y in Y]
            # y10=[y[9] for y in Y]
            # y11=[y[10] for y in Y]
            # y12=[y[11] for y in Y]
            # y13=[y[12] for y in Y]
            # y14=[y[13] for y in Y]
            # y15=[y[14] for y in Y]
            # y16=[y[15] for y in Y]
            # y17=[y[16] for y in Y]
            # y18=[y[17] for y in Y]
            # y19=[y[18] for y in Y]
            # y20=[y[19] for y in Y]
            # y21=[y[20] for y in Y]
            # y22=[y[21] for y in Y]
            # y23=[y[22] for y in Y]
            # y24=[y[23] for y in Y]
            # y25=[y[24] for y in Y]
            # y26=[y[25] for y in Y]
            
            
            # self.y1_max = max(self.y1_max, max(y1).item())
            # self.y2_max = max(self.y2_max, max(y2).item()) 
            # self.y3_max = max(self.y3_max, max(y3).item()) 
            # self.y4_max = max(self.y4_max, max(y4).item()) 
            # self.y5_max = max(self.y5_max, max(y5).item()) 
            # self.y6_max = max(self.y6_max, max(y6).item()) 
            # self.y7_max = max(self.y3_max, max(y7).item()) 
            # self.y8_max = max(self.y4_max, max(y8).item()) 
            # self.y9_max = max(self.y5_max, max(y9).item()) 
            # self.y10_max = max(self.y10_max, max(y10).item()) 
            # self.y11_max = max(self.y11_max, max(y11).item())
            # self.y12_max = max(self.y12_max, max(y12).item()) 
            # self.y13_max = max(self.y13_max, max(y13).item()) 
            # self.y14_max = max(self.y14_max, max(y14).item()) 
            # self.y15_max = max(self.y15_max, max(y15).item()) 
            # self.y16_max = max(self.y16_max, max(y16).item()) 
            # self.y17_max = max(self.y17_max, max(y17).item()) 
            # self.y18_max = max(self.y18_max, max(y18).item()) 
            # self.y19_max = max(self.y19_max, max(y19).item()) 
            # self.y20_max = max(self.y20_max, max(y20).item()) 
            # self.y21_max = max(self.y21_max, max(y21).item())
            # self.y22_max = max(self.y22_max, max(y22).item()) 
            # self.y23_max = max(self.y23_max, max(y23).item()) 
            # self.y24_max = max(self.y24_max, max(y24).item()) 
            # self.y25_max = max(self.y25_max, max(y25).item()) 
            # self.y26_max = max(self.y26_max, max(y26).item()) 
            
            # self.y1_min = min(self.y1_min, min(y1).item())
            # self.y2_min = min(self.y2_min, min(y2).item()) 
            # self.y3_min = min(self.y3_min, min(y3).item()) 
            # self.y4_min = min(self.y4_min, min(y4).item()) 
            # self.y5_min = min(self.y5_min, min(y5).item()) 
            # self.y6_min = min(self.y6_min, min(y6).item()) 
            # self.y7_min = min(self.y3_min, min(y7).item()) 
            # self.y8_min = min(self.y4_min, min(y8).item()) 
            # self.y9_min = min(self.y5_min, min(y9).item()) 
            # self.y10_min = min(self.y10_min, min(y10).item()) 
            # self.y11_min = min(self.y11_min, min(y11).item())
            # self.y12_min = min(self.y12_min, min(y12).item()) 
            # self.y13_min = min(self.y13_min, min(y13).item()) 
            # self.y14_min = min(self.y14_min, min(y14).item()) 
            # self.y15_min = min(self.y15_min, min(y15).item()) 
            # self.y16_min = min(self.y16_min, min(y16).item()) 
            # self.y17_min = min(self.y17_min, min(y17).item()) 
            # self.y18_min = min(self.y18_min, min(y18).item()) 
            # self.y19_min = min(self.y19_min, min(y19).item()) 
            # self.y20_min = min(self.y20_min, min(y20).item()) 
            # self.y21_min = min(self.y21_min, min(y21).item())
            # self.y22_min = min(self.y22_min, min(y22).item()) 
            # self.y23_min = min(self.y23_min, min(y23).item()) 
            # self.y24_min = min(self.y24_min, min(y24).item()) 
            # self.y25_min = min(self.y25_min, min(y25).item()) 
            # self.y26_min = min(self.y26_min, min(y26).item()) 
        #     
        #     i_failed = False
        #     
        #     if max(y1) < self.y1_max: 
        #         i_failed=True
        #     if max(y2) < self.y2_max: 
        #         i_failed=True
        #     if max(y3) < self.y3_max: 
        #         i_failed=True
        #     if max(y4) < self.y4_max: 
        #         i_failed=True
        #     if max(y5) < self.y5_max: 
        #         i_failed=True
        #     if max(y6) < self.y6_max: 
        #         i_failed=True
        #     if max(y7) < self.y7_max: 
        #         i_failed=True
        #     if max(y8) < self.y8_max: 
        #         i_failed=True
        #     if max(y9) < self.y9_max: 
        #         i_failed=True
        #     if max(y10) < self.y10_max: 
        #         i_failed=True
        #     if max(y11) < self.y11_max: 
        #         i_failed=True
        #     if max(y12) < self.y12_max: 
        #         i_failed=True
        #     if max(y13) < self.y13_max: 
        #         i_failed=True
        #     if max(y14) < self.y14_max: 
        #         i_failed=True
        #     if max(y15) < self.y15_max: 
        #         i_failed=True
        #     if max(y16) < self.y16_max: 
        #         i_failed=True
        #     if max(y17) < self.y17_max: 
        #         i_failed=True
        #     if max(y18) < self.y18_max: 
        #         i_failed=True
        #     if max(y19) < self.y19_max: 
        #         i_failed=True
        #     if max(y20) < self.y20_max: 
        #         i_failed=True
        #     if max(y21) < self.y21_max: 
        #         i_failed=True
        #     if max(y22) < self.y22_max: 
        #         i_failed=True
        #     if max(y23) < self.y23_max: 
        #         i_failed=True
        #     if max(y24) < self.y24_max: 
        #         i_failed=True
        #     if max(y25) < self.y25_max: 
        #         i_failed=True
        #     if max(y26) < self.y26_max: 
        #         i_failed=True
        # 
        #     
        #     if i_failed:
        #         self.failure_counter+=1
        #         self.success_counter= 0
        #     else:
        #         self.failure_counter=0
        #         self.success_counter+=1
        #         
        # else:
        print("Single Obj", flush=True)
        Y = torch.sum(Y, dim=1)
        if max(Y).item() > self.y_max: # 1e-3
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0
            
        print("Successes", flush=True)
        print(self.success_counter, flush=True)
        
        print("Failures", flush=True)
        print(self.failure_counter, flush=True)

        # else:
        print("Y", flush=True)
        #print(Y)
        print("Y min:", flush=True)
        print(min(Y).item(), flush=True)
        print("Y max:", flush=True)
        print(max(Y).item(), flush=True)
        print("self.ymax:", flush=True)
        self.y_max = max(self.y_max, max(Y).item())
        print(self.y_max, flush=True)

        print("Turbo length: ", self.length, flush=True)
        # # Force re-exploration after shrinking 6+ times
        # if self.length < 0.001:
        #     self.length = 0.5
        #     self.failure_counter=0
        #     self.success_counter=0
            
        if self.length < self.length_min:
            self.stopping_condition = True

        return self

    def generate_batch(self, model, X, Y):
        
        print("Generating Batch", flush=True)
        
        X_cand = self._create_candidates(model, X, Y)

        # with gpytorch.settings.max_cholesky_size(10000):
        # with torch.no_grad():
        #     thompson_sampling = MaxPosteriorSampling(model=model, replacement=False, objective=self.objective)
        #     X_next = thompson_sampling(X_cand, num_samples=2)
        
        print("Making predictions...",flush=True)
        model.eval()
        model.likelihood.eval()
        print("GC check",flush=True)
        print(Counter((t.shape for t in gc.get_objects() if isinstance(t, torch.Tensor))), flush=True)
        #exit(1)
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(10000):
            predictions = model.likelihood(model(X_cand.to(torch.double)))
            mean = predictions.mean.detach()
            #lower, upper = predictions.confidence_region()
        
        print("Selecting next samples...",flush=True)
        if mean.dim()>1 and mean.shape[-1] > 1:
            mean = mean.sum(dim=1)

        sorted,indices = torch.sort(mean)

        # for j in range(1,self.batch_size+1):
            #print(f"max at {indices[-j]} ({sorted[-j]})")
            
        o = indices[-self.batch_size:].numpy().tolist()
        X_next = X_cand[o]
        return X_next
    
    def _create_candidates2(self, model, X, Y):

        assert (X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))) or X.min() == -1.0
        X[X==-1.0]=0.0
        YO = self.objective(Y)
        print("YO",flush=True)
        print(YO.shape,flush=True)
        YT = torch.transpose(YO,0,1)
        print("YT",flush=True)
        print(YT.shape,flush=True)
        self.update(X, YT)
        if self.stopping_condition:
            return
        # Scale the TR to be proportional to the lengthscales
        yy = YT.sum(dim=1).numpy().tolist()
        print("yy",flush=True)
        print(yy,flush=True)
        x_center = X[yy.index(max(yy)), :].clone()
        print("x_center",flush=True)
        print(x_center,flush=True)
        weights = torch.ones(X.shape[-1])

        try:
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except Exception as e:
            try:
                weights = model.covar_module.data_covar_module.lengthscale.squeeze().detach()
            except Exception as e:
                pass
       
        weights = weights / weights.mean()
        # pdb.set_trace()
        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length / 2.0, 0.0, 1.0)

        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(self.n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(self.n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]
        print("X_cand",flush=True)
        print(X_cand.shape,flush=True)
        return X_cand
   
    def _create_candidates(self, model, X, Y):
        
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        YO = self.objective(Y)

        self.update(X, YO)
        if self.stopping_condition:
            return

        # Scale the TR to be proportional to the lengthscales
        x_center = X[YO.argmax(), :].clone()
        weights = torch.ones(X.shape[-1])

        try:
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except Exception as e:
            try:
                weights = model.covar_module.data_covar_module.lengthscale.squeeze().detach()
            except Exception as e:
                pass
       
        weights = weights / weights.mean()
        # pdb.set_trace()
        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length / 2.0, 0.0, 1.0)

        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(self.n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(self.n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        return X_cand
      
      
    def _create_candidates3(self, model, X, Y):
        print("creating candidates...", flush=True)
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        print("X")
        print(X.shape)
        YO = self.objective(Y)
        print("YO",flush=True)
        print(YO.shape,flush=True)
        print("Updating TTS...",flush=True)
        self.update(X, YO)
        
        if self.stopping_condition:
            return
        
        if YO.dim()>1 and YO.shape[-1] > 1:
            print("collapsing objectives...")
            YO=torch.sum(YO,dim=0)
            
        print("YO",flush=True)
        print(YO.shape,flush=True) 
        
        print("Scaling TR...",flush=True)
        
        # Scale the TR to be proportional to the lengthscales
        YO=YO.numpy().tolist()
        x_center = X[YO.index(min(YO)), :].clone()
        print("x_center")
        print(x_center)
        weights = torch.ones(X.shape[-1])

        try:
            weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except Exception as e:
            try:
                weights = model.covar_module.data_covar_module.lengthscale.squeeze().detach()
            except Exception as e:
                pass
       
        weights = weights / weights.mean()
        # pdb.set_trace()
        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length / 2.0, 0.0, 1.0)
        print(f"bounds: {tr_lb.shape}")
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(self.n_candidates).to(dtype=self.dtype, device=self.device)
        print(f"pert: {pert.shape}")
        tr_lb = torch.transpose(tr_lb,0,1)
        tr_ub = torch.transpose(tr_ub,0,1)
        pert = torch.transpose(pert,0,1)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        pert = torch.transpose(pert,0,1)
        print("Creating Mask...",flush=True)
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(self.n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        print("Creating candidates from mask...",flush=True)
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        return X_cand

    def write_checkpoint(self, checkpointdir):
        with open(checkpointdir + "/TurboThompson.json", "w") as statefile:
            json.dump(dataclasses.asdict(self), statefile, ensure_ascii=False, indent=4)
            return True
        return False

    def read_checkpoint(self, checkpointdir):
        with open(checkpointdir + "/TurboThompson.json") as statefile:
            s = json.load(statefile)
            self.dim = s['dim']
            self.batch_size = s['batch_size']
            self.length = s['length']
            self.length_min = s['length_min']
            self.length_max = s['length_max']
            self.failure_counter = s['failure_counter']
            self.failure_tolerance = s['failure_tolerance']
            self.success_counter = s['success_counter']
            self.success_tolerance = s['success_tolerance']
            self.y_max = s['y_max']
            # self.y1_max = s['y1_max']
            # self.y2_max = s['y2_max']
            # self.y3_max = s['y3_max']
            # self.y4_max = s['y4_max']
            # self.y5_max = s['y5_max']
            # self.y6_max = s['y6_max']
            # self.y7_max = s['y7_max']
            # self.y8_max = s['y8_max']
            # self.y9_max = s['y9_max']
            # self.y10_max = s['y10_max']
            # self.y11_max = s['y11_max']
            # self.y12_max = s['y12_max']
            # self.y13_max = s['y13_max']
            # self.y14_max = s['y14_max']
            # self.y15_max = s['y15_max']
            # self.y16_max = s['y16_max']
            # self.y17_max = s['y17_max']
            # self.y18_max = s['y18_max']
            # self.y19_max = s['y19_max']
            # self.y20_max = s['y20_max']
            # self.y21_max = s['y21_max']
            # self.y22_max = s['y22_max']
            # self.y23_max = s['y23_max']
            # self.y24_max = s['y24_max']
            # self.y25_max = s['y25_max']
            # self.y26_max = s['y26_max']
            self.n_candidates = s['n_candidates']
            return True
        return False
