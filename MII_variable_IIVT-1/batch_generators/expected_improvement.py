import os
import math
import json
import numpy as np
import dataclasses, json
from dataclasses import dataclass

import torch
import os
import math
import json
import numpy as np
import dataclasses, json
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective

from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import pathlib

from .batch_generator import BatchGenerator

class ExpectedImprovement(BatchGenerator):
    def __init__(self, dim=1, batch_size=4, num_restarts = 10, raw_samples = 512, objective = None, sampler = None, dtype=torch.double, device=torch.device("cpu")):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.dtype = dtype
        self.device = device
        self.sampler = sampler
        self.objective = objective

        if self.sampler is None:
            self.sampler = SobolQMCNormalSampler(self.raw_samples)#num_samples=512, collapse_batch_dims=True)

        if self.objective is None:
            self.objective = IdentityMCObjective()

    def generate_batch(self, model, X, Y, Y_train):
        super().__init__()

        ei = qExpectedImprovement(model=model, best_f=Y_train.max(), sampler=self.sampler, objective=self.objective, maximize=True)

        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                                torch.ones(self.dim, dtype=self.dtype, device=self.device)]),
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        return X_next

    def write_checkpoint(self, checkpointdir):
        return True

    def read_checkpoint(self, checkpointdir):
        return True
