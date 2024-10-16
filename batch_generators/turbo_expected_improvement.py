import os
import math
import json
import numpy as np
import dataclasses, json
from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

import pathlib

from .batch_generator import BatchGenerator

@dataclass
# class TurboThompsonSampling():
class TurboExpectedImprovement(BatchGenerator):
    dim: int
    batch_size: int
    length: float = 1.0
    length_min: float =2**-30
    length_max: float = 1.0
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3 # 10 #10  # Note: The original paper uses 3
    y_max: float = -float("inf")
    dtype = None
    device = None

    def __init__(self, dim=1, batch_size=4, failure_tolerance = None, num_restarts = 10, raw_samples = 512, dtype=torch.double, device=torch.device("cpu"), length_min=2**-30):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.length_min = length_min
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        if failure_tolerance is None:
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        else:
            self.failure_tolerance = failure_tolerance

    def update(self, X, Y):
        # print(self.failure_counter, self.failure_tolerance)
        if max(Y) > self.y_max: # 1e-3
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

        self.y_max = max(self.y_max, max(Y).item())
        print(self.length)

        if self.length < self.length_min:
            self.stopping_condition = True

        return self

    def generate_batch(self, model, X, Y, Y_train):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        self.update(X, Y)
        if self.stopping_condition:
            return

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()

        if len(weights.shape) == 0: weights = weights.unsqueeze(-1)

        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length / 2.0, 0.0, 1.0)

        ucb = qExpectedImprovement(model, Y_train.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return X_next

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
            return True
        return False
