import dataclasses
import json
import math
import numpy as np
import os
import pathlib
import pdb
import torch

from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from dataclasses import dataclass
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from .turbo_thompson_sampling import TurboThompsonSampling
from ..emulators.GP import ExactGP

class TurboThompsonSamplingLocal(TurboThompsonSampling):

    def __init__(self, model_local=ExactGP(), max_observations_local=300, covariance_distance=True, **kwargs):
        self.max_observations_local = max_observations_local
        self.model_local = model_local
        self.covariance_distance = covariance_distance
        super().__init__(**kwargs)

    def generate_batch(self, model, X, Y, Y_train):
        
        X_cand = self._create_candidates(model, X, Y)
        
        YO = self.objective(Y)

        # Determine the observations that have highest covariance to x_center
        # in the global GP
        if self.covariance_distance:
            covariance_to_center = model.forward(X).covariance_matrix[YO.argmax(), :]
            idx_highest_covariance = covariance_to_center.argsort(descending=True)[:self.max_observations_local-1]
            # # Make sure best point is included in local model, as kernel matrix gives the variance, not the covariance
            if YO.argmax() in idx_highest_covariance:
                idx_highest_covariance = covariance_to_center.argsort(descending=True)[:self.max_observations_local]
            else:
                idx_highest_covariance = torch.cat([idx_highest_covariance, YO.argmax().unsqueeze(-1)])
            X_loc, Y_loc = X[idx_highest_covariance], Y[idx_highest_covariance]
        else: # euclidean distance 
            distance_to_center = np.linalg.norm(X.cpu()[YO.argmax()] - X.cpu(), axis=-1)
            idx_closest_distance = distance_to_center.argsort()[:self.max_observations_local]
            X_loc, Y_loc = X[idx_closest_distance], Y[idx_closest_distance]

        # Rescale local points and candidate points to unit cube
        X_loc_lb = X_loc.min(dim=0)[0]
        X_loc_ub = X_loc.max(dim=0)[0]
        X_loc_bounds = torch.cat([X_loc_lb, X_loc_ub]).reshape(2,-1)
        X_loc_norm = normalize(X_loc, X_loc_bounds)
        X_cand_norm = normalize(X_cand, X_loc_bounds)
        
        # Fit the local GP
        self.model_local.fit(X_loc_norm, Y_loc)
        
        # Sample on the candidate points
        with torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model_local.model, replacement=False, objective=self.objective)
            X_next = thompson_sampling(X_cand_norm, num_samples=self.batch_size)
       
        # Transform back to original space
        X_next = unnormalize(X_next, X_loc_bounds)
        
        return X_next
