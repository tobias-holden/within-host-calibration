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
from botorch.acquisition import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import pathlib

from .batch_generator import BatchGenerator

class UpperConfidenceBound(BatchGenerator):
    def __init__(self, dim=1, batch_size=4, num_restarts = 10, raw_samples = 512, beta=0.1, dtype=torch.double, device=torch.device("cpu")):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.beta = beta
        self.dtype = dtype
        self.device = device

    def generate_batch(self, model, X, Y, Y_train):
        super().__init__()
        ucb = qUpperConfidenceBound(model, self.beta)
        X_next, acq_value = optimize_acqf(
            ucb,
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
