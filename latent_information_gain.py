#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for Self-Correcting Bayesian Optimization [hvarfner2023scorebo]_.

References

.. [hvarfner2023scorebo]
    C. Hvarfner, E. Hellsten, F. Hutter, L. Nardi.
    Self-Correcting Bayesian Optimization thorugh Bayesian Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2023.

Contributor: hvarfner
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from botorch import settings
from botorch.models import NeuralProcessModel
from torch import Tensor

import torch
#reference: https://arxiv.org/abs/2106.02770 

class LatentInformationGain:
    def __init__(
        self, 
        model: NeuralProcessModel, 
        num_samples: int = 10
    ) -> None:
        """
        Latent Information Gain (LIG) Acquisition Function, designed for the
        NeuralProcessModel.

        Args:
            model: Trained NeuralProcessModel.
            num_samples (int): Number of samples for calculation.
        """
        self.model = model
        self.num_samples = num_samples

    def acquisition(self, candidate_x, context_x, context_y):
        """
        Conduct the Latent Information Gain acquisition function for the inputs.

        Args:
            candidate_x: Candidate input points, as a Tensor.
            context_x: Context input points, as a Tensor.
            context_y: Context target points, as a Tensor.

        Returns:
            torch.Tensor: The LIG score of computed KLDs.
        """

        # Encoding and Scaling the context data
        z_mu_context, z_logvar_context = self.model.data_to_z_params(context_x, context_y)
        kl = 0.0
        for _ in range(self.num_samples):
            # Taking reparameterized samples
            samples = self.model.sample_z(z_mu_context, z_logvar_context)

            # Using the Decoder to take predicted values
            y_pred = self.model.decoder(candidate_x, samples)

            # Combining context and candidate data
            combined_x = torch.cat([context_x, candidate_x], dim=0)
            combined_y = torch.cat([context_y, y_pred], dim=0)

            # Computing posterior variables
            z_mu_posterior, z_logvar_posterior = self.model.data_to_z_params(combined_x, combined_y)
            std_prior = 0.1 + 0.9 * torch.sigmoid(z_logvar_context)
            std_posterior = 0.1 + 0.9 * torch.sigmoid(z_logvar_posterior)

            p = torch.distributions.Normal(z_mu_posterior, std_posterior)
            q = torch.distributions.Normal(z_mu_context, std_prior)

            kl_divergence = torch.distributions.kl_divergence(p, q).sum()
            kl += kl_divergence

        # Average KLD
        return kl / self.num_samples
