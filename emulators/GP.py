import botorch
import gpytorch
import numpy as np
import pdb
import torch

import gc
from collections import Counter

from botorch.models.gpytorch import GPyTorchModel
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.multitask import MultiTaskGP, KroneckerMultiTaskGP
from botorch.models.transforms.outcome import Standardize, Log
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch

from gpytorch.constraints import Interval, GreaterThan, LessThan
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

torch.set_default_dtype(torch.float64)

class GP():
    def print_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param)

    def posterior(self, X):
        """Get posterior distribution for X using fitted emulator"""
        
        with torch.no_grad():
            post = self.model.posterior(X)

        return post
    
    def posterior_mean(self, X, confidence_bounds = False):
        """Get posterior mean for X using fitted emulator"""
        
        posterior = self.posterior(X)
        mean = posterior.mean.detach()
        if not confidence_bounds:
            return mean
        
        SD = torch.sqrt(posterior.variance)
        return mean, (mean - 1.96 * SD).detach(), (mean + 1.96 * SD).detach()

class ExactGP(GP):
    def __init__(self, noise_constraint = GreaterThan(1e-6), objective = None): # Botorch default, previously we had: Interval(1e-8, 10):
        self.likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        self.noise_constraint = noise_constraint
        self.objective = objective

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        print(f"X shape: {X.shape}", flush=True)
        print(f"Y shape: {Y.shape}", flush=True)
       # Y = torch.transpose(Y,0,1)
       # print(f"Y shape: {Y.shape}", flush=True)
        if self.objective is not None:
            Y = self.objective(Y).unsqueeze(1)
            print(f"Y shape: {Y.shape}", flush=True)
            
        likelihood = GaussianLikelihood(noise_constraint=self.noise_constraint)
        
        self.model = SingleTaskGP(
            X,
            Y,
            likelihood=likelihood,
            outcome_transform=Standardize(m=Y.shape[-1]) # Standardize Y for training, then untransform Y for predictions
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

class ExactGPTurboLocal(GP):
    def __init__(self, turbo, noise_constraint = GreaterThan(1e-6)): # Botorch default, previously we had: Interval(1e-8, 10):
        self.likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        self.noise_constraint = noise_constraint
        self.turbo = turbo

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        likelihood = GaussianLikelihood(noise_constraint=self.noise_constraint)

        length = self.turbo.length
        # print("GP length: ", length)

        x_center = X[Y.argmax(), :].clone()
        weights = torch.ones(X.shape[-1]).to(x_center)

        try:
            weights = self.model.covar_module.base_kernel.lengthscale.squeeze().detach()
        except Exception as e:
            try:
                weights = self.model.covar_module.data_covar_module.lengthscale.squeeze().detach()
            except Exception as e:
                pass
       
        weights = weights / weights.mean()

        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)

        mask = (X >= tr_lb) & (X <= tr_ub)
        supermask = (mask[:] == True)
        supermask = supermask[:,0]

        # print(X[0:5])
        # print(mask[0:5])
        # print(supermask[0:5])

        localX = X[supermask]
        localY = Y[supermask]

        # print("Original: ", len(X))
        print("Reduced: ", len(localX))

        self.model = SingleTaskGP(
            localX,
            localY,
            likelihood=likelihood,
            outcome_transform=Standardize(m=Y.shape[-1]) # Normalize Y for training, then untransform Y for predictions
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        self.tr_lb = tr_lb
        self.tr_ub = tr_ub

class ExactGPFixedNoise(GP):
    def fit(self, X, Y, noise = None):
        self.X = X
        self.Y = Y
        
        if noise is None:
            noise = (torch.zeros(len(X)) + 1e-6).unsqueeze(-1)

        self.model = FixedNoiseGP(
            X,
            Y,
            noise,
            outcome_transform = Standardize(m=Y.shape[-1])
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)
        
        return self.model

class ApproximateGPyTorchModel(GPyTorchModel, ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self,
                                                   inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True)
        super(ApproximateGPyTorchModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self._num_outputs = 1 # attribute required by botorch

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproximateGP(GP):

    def __init__(
        self,
        noise_constraint = Interval(1e-8, 10),
        n_inducing_points = 50
    ):
        self.likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        self.n_inducing_points = n_inducing_points

    def fit(self, X, Y, noise = 0):

        self.X = X
        self.Y = Y
        
        model = ApproximateGPyTorchModel(
            X[:self.n_inducing_points],
        ).double()
        
        likelihood = GaussianLikelihood()
        
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)
        
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y.size(0))

        train_dataset = TensorDataset(X, Y.squeeze())
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        
        num_epochs=100
        loss_best = torch.Tensor([float('Inf')])
        non_improvement_counter = 0
        for i in range(num_epochs):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                output = model(X)
                loss = -mll(output, Y.squeeze())
            
            if loss < loss_best:
                loss_best = loss
                non_improvement_counter = 0
            else:
                non_improvement_counter += 1
            if non_improvement_counter>10:
                break

        self.model = model
        return self.model

class ExactMultiTaskGP(GP):
    def __init__(self, noise_constraint = GreaterThan(1e-6)): # Botorch default, previously we had: Interval(1e-8, 10):
        #self.likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        self.noise_constraint = noise_constraint

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        #YT = torch.transpose(self.Y,0,1)
        print("X", flush=True)
        print(X.shape, flush=True)
        print("Y", flush=True)
        print(Y.shape, flush=True)
        print("Y transposed", flush=True)
        print(torch.transpose(Y,0,1).shape, flush=True)
        Y = torch.transpose(Y,0,1)
        # print(X)
        # print(Y)
        # self.model = MultiTaskGP(#KroneckerMultiTaskGP(
        #     X, 
        #     Y,
        #     task_feature=-1,
        #     outcome_transform=Standardize(m=Y.shape[-1]) # Standardize Y for training, then untransform Y for predictions
        # )

        likelihood = MultitaskGaussianLikelihood(num_tasks=Y.shape[-1], noise_constraint=self.noise_constraint)

        self.model = KroneckerMultiTaskGP(
            X.unsqueeze(-1), 
            Y.unsqueeze(-1),
            likelihood=likelihood,
            outcome_transform=Standardize(m=2) # Standardize Y for training, then untransform Y for predictions
        )
        
        self.model.train()
        self.model.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
#        print("GC check",flush=True)
#        print(Counter((t.shape for t in gc.get_objects() if isinstance(t, torch.Tensor))), flush=True)
        print("fitting...", flush=True)
        #fit_gpytorch_torch(mll)
        fit_gpytorch_model(mll, options={"maxiter":1})

        print("done fitting...",flush=True)
        #exit(1)
        self.model.eval()
        self.model.likelihood.eval()

        
