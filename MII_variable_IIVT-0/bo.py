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
        max_evaluations = None,
        checkpointdir = "./",
        objective = IdentityMCObjective(),
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
        #print(f"Dim1 : {self.problem.dim}")
        #print(f"Dim2 : {Xpriors.dim}")
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
            self.Y_pred_mean = torch.tensor([[float('nan') for i in range(self.Y.shape[0])] for j in range(self.n_init)], dtype=self.dtype, device=self.device)
            self.Y_pred_var = torch.tensor([[float('nan') for i in range(self.Y.shape[0])] for j in range(self.n_init)], dtype=self.dtype, device=self.device)
            self.iterations = torch.tensor([0]).repeat(self.n_init)
            self.timer_model = torch.tensor([], dtype=self.dtype, device=self.device).float()
            self.timer_batch_generator = torch.tensor([], dtype=self.dtype, device=self.device).float()

            self.write_checkpoint(checkpointdir=self.checkpointdir)
            print(f"{len(self.X)}) Best value: {self.objective(self.Y).max():.2e}", flush=True)

    # Run an full optimization loop
    def run(self):
        print("run", flush=True)
        print(f"X length {len(self.X)}", flush=True)
        print(f"max eval {self.max_evaluations}", flush=True)
        print(f"stop con {self.stopping_condition}", flush=True)
        if self.read_checkpoint(checkpointdir=self.checkpointdir) is False or self.n_init is None:
            print("Error: BO is not initialized.\nRun init() before run().", flush=True)
            return
        print(f"Continue? {(len(self.X) < self.max_evaluations) and not self.stopping_condition}", flush=True)
        while (len(self.X) < self.max_evaluations) and not self.stopping_condition:
            print("empty cache", flush=True)
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("take next step...", flush=True)    
            self.step()
            self.write_checkpoint(checkpointdir=self.checkpointdir)
            print(f"{len(self.X)}) Best value: {self.objective(self.Y).max():.2e}", flush=True)
            #print(f"{len(self.X)}) Best value: {self.X(self.Y).max()}", flush=True)
    
    def step(self):
        print("step", flush=True)
        X = self.X
        Y = self.Y
      
        tic = time.perf_counter()
        self.model.fit(X, Y)
        #print("done fitting", flush=True)
        timer_model = time.perf_counter() - tic
        
        # Create batch
        tic = time.perf_counter()
        print("batch time", flush=True)
        X_next = self.batch_generator.generate_batch(model=self.model.model, X=X, Y=Y)
        if self.batch_generator.stopping_condition:
            self.stopping_condition = True
            return
            
        timer_batch_generator = time.perf_counter() - tic
        
        # Evaluate selected points

        print("Evaluate next points...",flush=True)
        
        X_next, Y_next = self.problem(X_next)
        X_next = X_next.to(dtype=self.dtype, device=self.device)
        Y_next = Y_next.to(dtype=self.dtype, device=self.device)

        if len(Y_next) == 0:
            print("Error: empty evalutation", flush=True)
            return self.X, self.Y
        
        #print(Y.shape)
        #print(Y_next.shape)
        #Y_next=torch.transpose(Y_next,0,1)
        # Append selected evaluations to data
        self.X = torch.cat((X, X_next), dim=0)
        self.Y = torch.cat((Y,Y_next), dim=0)
        # Append BO statistics
        pred_posterior = self.model.posterior(X_next)
        pred_posterior_mean = pred_posterior.mean
        pred_posterior_var = pred_posterior.variance
        #print(self.Y_pred_mean.shape)
        #print(pred_posterior_mean.shape)
        #print(self.Y_pred_var.shape)
        #print(pred_posterior_var.shape)
        # Hack to make single task global GP work with multioutput problem: Need to replace nan predictions for initial samples
        if len(self.Y_pred_mean) == self.n_init and self.Y_pred_mean.shape[-1] != pred_posterior_mean.shape[-1]:
            self.Y_pred_mean = torch.tensor([float('nan')]).repeat(self.n_init).unsqueeze(-1)
            self.Y_pred_var = torch.tensor([float('nan')]).repeat(self.n_init).unsqueeze(-1)
        #print("pred_post_shapes:")
        #print(self.Y_pred_mean.shape)
        #print(torch.sum(pred_posterior_mean,dim=1).shape)
        #exit(1)
        self.Y_pred_mean = torch.cat((self.Y_pred_mean, torch.sum(pred_posterior_mean,dim=1).unsqueeze(dim=1)))
        self.Y_pred_var = torch.cat((self.Y_pred_var, torch.sum(pred_posterior_var,dim=1).unsqueeze(dim=1)))
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
        #print("finish step")
        return self.X, self.Y
