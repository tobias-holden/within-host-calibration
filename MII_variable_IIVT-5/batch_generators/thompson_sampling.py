import dataclasses
import json
import math
import numpy as np
import os
import pathlib
import torch

from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from dataclasses import dataclass
from torch.quasirandom import SobolEngine

from .batch_generator import BatchGenerator

class ThompsonSampling(BatchGenerator):
    def __init__(self, dim=1, batch_size=4, n_candidates=None, dtype=torch.double, device=torch.device("cpu")):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.dtype = dtype
        self.device = device

        if self.n_candidates is None:
            self.n_candidates = min(5000, max(2000, 200 * self.dim))

    def generate_batch(self, model, X, Y, Y_train):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()

        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * 1.0 / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * 1.0 / 2.0, 0.0, 1.0)

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

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=self.batch_size)
        
        return X_next

    def write_checkpoint(self, checkpointdir):
        return True

    def read_checkpoint(self, checkpointdir):
        return True
