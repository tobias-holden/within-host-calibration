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
from botorch.acquisition import qExpectedImprovement
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

class BatchGeneratorArray:
    def __init__(self, batch_generators=None):
        self.batch_generators = batch_generators
        self.stopping_condition = False

    def generate_batch(self, model, X, Y, Y_train):
        X_next = torch.tensor([], dtype=self.batch_generators[0].dtype, device=self.batch_generators[0].device)
        for g in self.batch_generators:
            X_next_generator = g.generate_batch(model, X, Y, Y_train)
            if g.stopping_condition:
                self.stopping_condition = True
                print(f"Stopping condition met by batch generator {g}")
            else:
                X_next = torch.cat((X_next, X_next_generator), dim=0)
        return X_next

    def write_checkpoint(self, checkpointdir):
        r = True
        for g in self.batch_generators:
            if g.write_checkpoint(checkpointdir) is False:
                r = False
        return r

    def read_checkpoint(self, checkpointdir):
        r = True
        for g in self.batch_generators:
            if g.read_checkpoint(checkpointdir) is False:
                r = False
        return r
