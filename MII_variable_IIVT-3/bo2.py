import os, sys, time
import pathlib
import torch
import gpytorch
import numpy as np
import random
from torch.quasirandom import SobolEngine
from botorch.acquisition.objective import IdentityMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

torch.set_default_dtype(torch.float64)
sample_cap=100

class BO:
    def __init__(self, 
        problem = None, 
        model = None,
        batch_generator = None,
        max_evaluations = 100,
        checkpointdir = "./",
        objective = IdentityMCMultiOutputObjective(),
        dtype = torch.double, 
        device = torch.device("cpu")
    ):
        self.problem = problem # Functor
        self.model = model # GP model
        self.batch_generator = batch_generator # Batch generator or a batch_generator_array
        self.max_evaluations = max_evaluations # Maximum number of evaluations before stopping
        self.checkpointdir = checkpointdir # Folder to store intermediate results
        self.objective = objective # Botorch objective function to convert a multiple output to a single output
        self.dtype = dtype
        self.device = device
        
        self.X = None
        self.Y = None
        self.timer_model = None
        self.timer_batch_generator = None
        self.iterations = None
        self.length_scales = None
        self.Y_pred_var=None
        self.Y_pred_mean = None

        self.stopping_condition = False

        self.n_init = None
        
        self.n_candidates=1000
        self.length=1.0

    def write_checkpoint(self, checkpointdir):
        if checkpointdir != "":
            pathlib.Path(checkpointdir).mkdir(parents=True, exist_ok=True)

            torch.save(self.X, checkpointdir + "/X.pt")
            torch.save(self.Y, checkpointdir + "/Y.pt")
            torch.save(self.Y_pred_mean, checkpointdir + "/Y_pred_mean.pt")
            torch.save(self.Y_pred_var, checkpointdir + "/Y_pred_var.pt")
            torch.save(self.length_scales, checkpointdir + "/length_scales.pt")
            torch.save(self.iterations, checkpointdir + "/iterations.pt")
            torch.save(self.timer_model, checkpointdir + "/timer_model.pt")
            torch.save(self.timer_batch_generator, checkpointdir + "/timer_batch_generator.pt")
            ## TO ADD: copy contents of initial output directory so we can update later, then may need to write one to open and run only the LL (or other bits as needed), update, and save initial values - probably should do in a separate script that only runs when we want...
            self.batch_generator.write_checkpoint(checkpointdir)

    def read_checkpoint(self, checkpointdir):
        X = None
        Y = None
        checkpoint = False
        if checkpointdir != "":
            try:
                self.X = torch.load(checkpointdir + "/X.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.Y = torch.load(checkpointdir + "/Y.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.Y_pred_mean = torch.load(checkpointdir + "/Y_pred_mean.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.Y_pred_var = torch.load(checkpointdir + "/Y_pred_var.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.length_scales = torch.load(checkpointdir + "/length_scales.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.iterations = torch.load(checkpointdir + "/iterations.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.timer_model = torch.load(checkpointdir + "/timer_model.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                self.timer_batch_generator = torch.load(checkpointdir + "/timer_batch_generator.pt", map_location=torch.device(self.device)).to(dtype=self.dtype, device=self.device)
                checkpoint = True
            except Exception as e: print(e, flush=True)

            try:
                self.batch_generator.read_checkpoint(checkpointdir)
            except Exception as e:
                print(e, flush=True)
                print("Error: Unable to load the batch_generator checkpoint")

        if checkpoint is True:
            print(f"Checkpoint loaded successfully.", flush=True)
            print(f"{len(self.X)}) Best value: {self.objective(self.Y).max():.2e}", flush=True)

        return checkpoint

    def initRandom(self, n_init, n_batches = 1, Xpriors = None):
        self.n_init = n_init
        if self.read_checkpoint(checkpointdir=self.checkpointdir) is False:
            sobol = SobolEngine(dimension=self.problem.dim, scramble=True)
            initX = sobol.draw(n=n_init).to(dtype=self.dtype, device=self.device)
            if Xpriors is not None:
                initX = torch.cat((torch.tensor(Xpriors, dtype=self.dtype, device=self.device),initX), dim=0)

            batch_size = int(len(initX)/n_batches)

            # Evaluate samples in batches to avoid large array jobs in Slurm
            X = torch.tensor([], dtype=self.dtype, device=self.device)
            Y = torch.tensor([], dtype=self.dtype, device=self.device)
            for batch in range(0, n_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                if batch == n_batches-1:
                    batch_end = batch_start + len(initX) - (n_batches-1) * batch_size

                bX, bY = self.problem(initX[batch_start:batch_end])
                X = torch.cat((X, bX.to(dtype=self.dtype, device=self.device)), axis=0)
                Y = torch.cat((Y, bY.to(dtype=self.dtype, device=self.device)), axis=0)

            self.X = X#torch.tensor(X, dtype=self.dtype, device=self.device)
            self.Y = Y#torch.tensor(Y, dtype=self.dtype, device=self.device)#.unsqueeze(-1)
            self.length_scales = torch.full((1, X.shape[-1]), torch.nan, dtype=self.dtype, device=self.device)
            self.Y_pred_mean = torch.tensor([[torch.nan for i in range(self.Y.shape[-1])] for j in range(self.n_init)], dtype=self.dtype, device=self.device)
            self.Y_pred_var = torch.tensor([[torch.nan for i in range(self.Y.shape[-1])] for j in range(self.n_init)], dtype=self.dtype, device=self.device)
            self.iterations = torch.tensor([0]).repeat(self.n_init)
            self.timer_model = torch.tensor([], dtype=self.dtype, device=self.device).float()
            self.timer_batch_generator = torch.tensor([], dtype=self.dtype, device=self.device).float()

            self.write_checkpoint(checkpointdir=self.checkpointdir)
            print(f"{len(self.X)}) Best value: {self.objective(self.Y).max():.2e}", flush=True)

    def initX(self, initX):
        self.n_init = len(X)
        if self.read_checkpoint(checkpointdir=self.checkpointdir) is False:
            batch_size = int(len(initX)/n_batches)

            # Evaluate samples in batches to avoid large array jobs in Slurm
            X = torch.tensor([], dtype=self.dtype, device=self.device)
            Y = torch.tensor([], dtype=self.dtype, device=self.device)
            for batch in range(0, n_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                if batch == n_batches-1:
                    batch_end = batch_start + len(initX) - (n_batches-1) * batch_size

                bX, bY = self.problem(initX[batch_start:batch_end])
                X = torch.cat((X, bX.to(dtype=self.dtype, device=self.device)), axis=0)
                Y = torch.cat((Y, bY.to(dtype=self.dtype, device=self.device)), axis=0)

            self.X = X#torch.tensor(X, dtype=self.dtype, device=self.device)
            self.Y = Y#torch.tensor(Y, dtype=self.dtype, device=self.device)#.unsqueeze(-1)
            self.length_scales = torch.full((1, X.shape[-1]), torch.nan, dtype=self.dtype, device=self.device)
            self.Y_pred_mean = torch.tensor([[float('nan') for i in range(self.Y.shape[-1])] for j in range(self.n_init)], dtype=self.dtype, device=self.device)
            self.Y_pred_var = torch.tensor([[float('nan') for i in range(self.Y.shape[-1])] for j in range(self.n_init)], dtype=self.dtype, device=self.device)
            self.iterations = torch.tensor([0]).repeat(self.n_init)
            self.timer_model = torch.tensor([], dtype=self.dtype, device=self.device).float()
            self.timer_batch_generator = torch.tensor([], dtype=self.dtype, device=self.device).float()

            self.write_checkpoint(checkpointdir=self.checkpointdir)
            print(f"{len(self.X)}) Best value: {self.objective(self.Y).max():.2e}", flush=True)

    # Run an full optimization loop
    def run(self):
        if self.read_checkpoint(checkpointdir=self.checkpointdir) is False or self.n_init is None:
            print("Error: BO is not initialized.\nRun init() before run().")
            return
        while (len(self.X) < self.max_evaluations) and not self.stopping_condition:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            self.step()
            self.write_checkpoint(checkpointdir=self.checkpointdir)
            print(f"{len(self.X)}) Best value: {self.objective(self.Y).max():.2e}", flush=True)
            #print(f"{len(self.X)}) Best value: {self.X(self.Y).max()}", flush=True)
    
    # Run one iteration
    def step(self):
        X = self.X
        print("X")
        print(X.shape)
        Y=self.Y
        YT = torch.transpose(Y,0,1)
        print("YT")
        print(YT.shape)

        Y_train = YT
        
        tic = time.perf_counter()
        self.model.fit(X, Y_train)
        timer_model = time.perf_counter() - tic
        
        # Create batch
        tic = time.perf_counter()
        
        test_X = torch.rand(1000,15)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.model.likelihood(self.model.model(test_X.to(torch.double)))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
        
        print("predictions")
        print(mean.shape)
        doms=[]
        for i in range(mean.shape[-1]):
            print(f"Y {i}")
            sorted, indices = torch.sort(mean[:,i])
            print(f"max at {indices[-1]} ({sorted[-1]})\n")
            doms.append(indices[-1])

        M = mean.sum(dim=1)
        
        sorted,indices = torch.sort(M)
        
        print("Overall")
        print(f"max at {indices[-1]} ({sorted[-1]})")
        
        o = indices[-1].numpy().tolist()

        #doms.append(indices[-1])
        
        doms=torch.stack(doms).numpy().tolist()
        
        print(f"Dominating Points:\n{doms}")
        doms=np.unique(doms)
        doms=np.append(doms,[o]*5)
        print(f"Unique Seed Points:\n{doms}")
        print(test_X[doms,:])
        
        x_dom=test_X[doms,:]
        ii=0
        for x in x_dom:
            
            tr_lb = torch.clamp(x - self.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x + self.length / 2.0, 0.0, 1.0)
    
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
            X_cand = x.expand(self.n_candidates, dim).clone()
            X_cand[mask] = pert[mask]
            print(f"round {ii}")
            print("X_cand")
            print(X_cand.shape)
            if ii ==0:
                next_X=X_cand
                
            else:
                next_X=torch.cat((next_X,X_cand))
            ii +=1
            print(f"Next X shape {next_X.shape}")
        
        capped_X = next_X[random.sample(range(next_X.shape[0]),sample_cap),:]
        print("Capped X")
        print(capped_X.shape)
       
         
        if self.batch_generator.stopping_condition:
            self.stopping_condition = True
            return
            
        timer_batch_generator = time.perf_counter() - tic
        
        # Evaluate selected points
        X_next, Y_next = self.problem(capped_X)
        X_next = X_next.to(dtype=self.dtype, device=self.device)
        Y_next = Y_next.to(dtype=self.dtype, device=self.device)
  
        if len(Y_next) == 0:
            print("Error: empty evalutation", flush=True)
            return self.X, self.Y
        Y_next=torch.transpose(Y_next,0,1)
        print("Y_next")
        print(Y_next.shape)
        print("YT")
        print(YT.shape)
        
        # Append selected evaluations to data
        self.X = torch.cat((X, X_next), dim=0)
        self.Y = torch.cat((YT, Y_next), dim=0)
        self.Y = torch.transpose(self.Y,0,1)
        print("final Y")
        print(self.Y.shape)
        # Append BO statistics
        pred_posterior = self.model.posterior(X_next)
        pred_posterior_mean = pred_posterior.mean
        pred_posterior_var = pred_posterior.variance
        
        print("shapes")
        print(self.Y_pred_var.shape)
        print(pred_posterior_var.shape)
        exit(1)
        # Hack to make single task global GP work with multioutput problem: Need to replace nan predictions for initial samples
        if len(self.Y_pred_mean) == self.n_init and self.Y_pred_mean.shape[-1] != pred_posterior_mean.shape[-1]:
            self.Y_pred_mean = torch.tensor([float('nan')]).repeat(self.n_init).unsqueeze(-1)
            self.Y_pred_var = torch.tensor([float('nan')]).repeat(self.n_init).unsqueeze(-1)
            
        self.Y_pred_mean = torch.cat((self.Y_pred_mean, pred_posterior_mean))
        self.Y_pred_var = torch.cat((self.Y_pred_var, pred_posterior_var))
        self.iterations = torch.cat((self.iterations, torch.tensor([self.iterations[-1]+1]).repeat(len(X_next)).to(self.iterations)))
        
        # Assumes emulator is a GP, which is always the case currently. Change in case we add other emulators
        try:
            length_scales = self.model.model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except Exception as e:
            try:
                length_scales = self.model.model.covar_module.data_covar_module.lengthscale.squeeze().detach()
            except:
                pass

        length_scales = length_scales.unsqueeze(0)

        if length_scales.shape[-1] == 1:
             length_scales = length_scales.unsqueeze(0)

        self.length_scales = torch.cat((self.length_scales, length_scales)).to(self.length_scales)
        
        self.timer_model = torch.cat((self.timer_model, torch.tensor([timer_model]).to(self.timer_model)))
        self.timer_batch_generator = torch.cat((self.timer_batch_generator, torch.tensor([timer_batch_generator]).to(self.timer_batch_generator)))
        
        return self.X, self.Y