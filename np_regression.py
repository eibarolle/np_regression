r"""
Neural Process Regression models based on PyTorch models.

References:

.. [Wu2023arxiv]
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Contributor: eibarolle
"""

import copy
import numpy as np
from numpy.random import binomial
import torch
import torch.nn as nn
import matplotlib.pyplot as plts
# %matplotlib inline
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.acquisition.objective import PosteriorTransform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from typing import Callable, List, Optional, Tuple
from torch.nn import Module, ModuleDict, ModuleList
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors import MultivariateNormalPosterior

device = torch.device("cpu")
# Account for different acquisitions

#reference: https://chrisorm.github.io/NGP.html
class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int], 
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""
        A modular implementation of a Multilayer Perceptron (MLP).
        
        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            if init_func is not None:
                init_func(layer.weight)
            layers.append(layer)
            layers.append(activation())
            prev_dim = hidden_dim

        final_layer = nn.Linear(prev_dim, output_dim)
        if init_func is not None:
            init_func(final_layer.weight)
        layers.append(final_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class REncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""Encodes inputs of the form (x_i,y_i) into representations, r_i.

        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation=activation, init_func=init_func)
        
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass for representation encoder.

        Args:
            inputs: Input tensor

        Returns:
            torch.Tensor: Encoded representations
        """
        return self.mlp(inputs)

class ZEncoder(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""Takes an r representation and produces the mean & standard 
        deviation of the normally distributed function encoding, z.
        
        Args:
            input_dim: An int representing r's aggregated dimensionality.
            output_dim: An int representing z's latent dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.mean_net = MLP(input_dim, output_dim, hidden_dims, activation=activation, init_func=init_func)
        self.logvar_net = MLP(input_dim, output_dim, hidden_dims, activation=activation, init_func=init_func)
        
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass for latent encoder.

        Args:
            inputs: Input tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Mean of the latent Gaussian distribution.
                - Log variance of the latent Gaussian distribution.
        """
        return self.mean_net(inputs), self.logvar_net(inputs)
    
class Decoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = nn.init.normal_
    ) -> None:
        r"""Takes the x star points, along with a 'function encoding', z, and makes predictions.
        
        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation=activation, init_func=init_func)
        
    def forward(
        self,
        x_pred: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass for decoder.

        Args:
            x_pred: No. of data points, by x_dim
            z: No. of samples, by z_dim

        Returns:
            torch.Tensor: Predicted target values.
        """
        z_expanded = z.unsqueeze(0).expand(x_pred.size(0), -1)
        xz = torch.cat([x_pred, z_expanded], dim=-1)
        return self.mlp(xz)

def MAE(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    r"""Mean Absolute Error loss function.

    Args:
        pred: The predicted values tensor.
        target: The target values tensor.

    Returns:
        torch.Tensor: A tensor representing the MAE.
    """
    loss = torch.abs(pred-target)
    return loss.mean()

class NeuralProcessModel(Model):
    def __init__(
        self,
        r_hidden_dims: List[int], 
        z_hidden_dims: List[int], 
        decoder_hidden_dims: List[int],
        x_dim: int,
        y_dim: int,
        r_dim: int,
        z_dim: int,
        activation: Callable = nn.Sigmoid,
        init_func: Optional[Callable] = torch.nn.init.normal_,
    ) -> None:
        r"""Diffusion Convolutional Recurrent Neural Network Model Implementation.

        Args:
            r_hidden_dims: Hidden Dimensions/Layer list for REncoder
            z_hidden_dims: Hidden Dimensions/Layer list for ZEncoder
            decoder_hidden_dims: Hidden Dimensions/Layer for Decoder
            x_dim: Int dimensionality of input data x.
            y_dim: Int dimensionality of target data y.
            r_dim: Int dimensionality of representation r.
            z_dim: Int dimensionality of latent variable z.
            activation: Activation function applied between layers, defaults to nn.Sigmoid.
            init_func: A function initializing the weights, defaults to nn.init.normal_.
        """
        super().__init__()
        self.r_encoder = REncoder(x_dim+y_dim, r_dim, r_hidden_dims, activation=activation, init_func=init_func) 
        self.z_encoder = ZEncoder(r_dim, z_dim, z_hidden_dims, activation=activation, init_func=init_func) 
        self.decoder = Decoder(x_dim + z_dim, y_dim, decoder_hidden_dims, activation=activation, init_func=init_func) 
        self.z_mu_all = None
        self.z_logvar_all = None
        self.z_mu_context = None
        self.z_logvar_context = None
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) # Look at BoTorch native versions
        #self.train(n_epochs, x_train, y_train)
    
    def data_to_z_params(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xy_dim: int = 1,
        r_dim: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute latent parameters from inputs as a latent distribution.

        Args:
            x: Input tensor
            y: Target tensor
            xy_dim: Combined Input Dimension as int, defaults as 1
            r_dim: Combined Target Dimension as int, defaults as 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - x_c: Context input data.
                - y_c: Context target data.
                - x_t: Target input data.
                - y_t: Target target data.
        """
        xy = torch.cat([x,y], dim=xy_dim)
        rs = self.r_encoder(xy)
        r_agg = rs.mean(dim=r_dim)
        return self.z_encoder(r_agg) 
    
    def sample_z(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n: int = 1,
        min_std: float = 0.1,
        scaler: float = 0.9
    ) -> torch.Tensor:
        r"""Reparameterization trick for z's latent distribution.

        Args:
            mu: Tensor representing the Gaussian distribution mean.
            logvar: Tensor representing the log variance of the Gaussian distribution.
            n: Int representing the # of samples, defaults to 1.
            min_std: Float representing the minimum possible standardized std, defaults to 0.1.
            scaler: Float scaling the std, defaults to 0.9.

        Returns:
            torch.Tensor: Samples from the Gaussian distribution.
    """
        if min_std <= 0 or scaler <= 0:
            raise ValueError()
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(self.z_dim).normal_()).to(device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n,self.z_dim).normal_()).to(device)
        
        std = min_std + scaler * torch.sigmoid(logvar) 
        return mu + std * eps

    def KLD_gaussian(
        self,
        min_std: float = 0.1,
        scaler: float = 0.9
    ) -> torch.Tensor:
        r"""Analytical KLD between 2 Gaussian Distributions.

        Args:
            min_std: Float representing the minimum possible standardized std, defaults to 0.1.
            scaler: Float scaling the std, defaults to 0.9.
            
        Returns:
            torch.Tensor: A tensor representing the KLD.
        """
        
        if min_std <= 0 or scaler <= 0:
            raise ValueError()
        std_q = min_std + scaler * torch.sigmoid(self.z_logvar_all)
        std_p = min_std + scaler * torch.sigmoid(self.z_logvar_context)
        p = torch.distributions.Normal(self.z_mu_context, std_p)
        q = torch.distributions.Normal(self.z_mu_all, std_q)
        return torch.distributions.kl_divergence(p, q).sum()
    
    def posterior(
        self, 
        X: torch.Tensor, 
        covariance_multiplier: float,
        observation_constant: float,
        observation_noise: bool = False, 
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> MultivariateNormalPosterior:
        r"""Computes the model's posterior distribution for given input tensors.

        Args:
            X: Input Tensor
            covariance_multiplier: Float scaling the covariance.
            observation_constant: Float representing the noise constant.
            observation_noise: Adds observation noise to the covariance if True, defaults to False.
            posterior_transform: An optional posterior transformation, defaults to None.

        Returns:
            MultivariateNormalPosterior: The posterior distribution object 
            utilizing MultivariateNormal.
        """
        mean = self.decoder(X, self.sample_z())
        covariance = torch.eye(X.size(0)) * covariance_multiplier
        if (observation_noise):
            covariance = covariance + observation_constant
        mvn = MultivariateNormal(mean, covariance)
        posterior = MultivariateNormalPosterior(mvn)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior
        
    def condition_on_observations(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor
    ) -> "NeuralProcessModel":
        r"""Condition the model on new observations.

        Args:
            X: Input tensor.
            Y: Target tensor.

        Returns:
            NeuralProcessModel: The current model with new conditioned train data.
        """
        new_copy = copy.deepcopy(self)
        new_copy.x_train = X
        new_copy.y_train = Y
        return new_copy
    
    def load_state_dict(
        self, 
        state_dict: dict, 
        strict: bool = True
    ) -> None:
        """
        Initialize the fully Bayesian model before loading the state dict.

        Args:
            state_dict (dict): A dictionary containing the parameters.
            strict (bool): Case matching strictness.
        """
        super().load_state_dict(state_dict, strict=strict)

    def transform_inputs(
        self,
        X: torch.Tensor,
        input_transform: Optional[Module] = None,
    ) -> torch.Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            torch.Tensor: A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    def forward(
        self,
        x_t: torch.Tensor,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        input_dim: int = 0,
        target_dim: int = 1
    ) -> torch.Tensor:
        r"""Forward pass for the model.

        Args:
            x_t: Target input data.
            x_c: Context input data.
            y_c: Context target data.
            input_dim: Input dimension concatenated
            target_dim: Target dimension concatendated

        Returns:
            torch.Tensor: Predicted target values.
        """
        if any(tensor.numel() == 0 for tensor in [x_t, x_c, y_c]):
            raise ValueError()
        if input_dim not in [0, 1]:
            raise ValueError()
        if x_c.size(1 - input_dim) != x_t.size(1 - input_dim):
            raise ValueError()
        if x_c.size(1 - target_dim) != y_c.size(1 - target_dim):
            raise ValueError()
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(torch.cat([x_c, x_t], dim = input_dim), torch.cat([y_c, x_t], dim = target_dim))
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        z = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, z)
    
    def random_split_context_target(
        x: torch.Tensor,
        y: torch.Tensor,
        n_context: int,
        axis: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Helper function to split randomly into context and target.

        Args:
            x: Input data tensor.
            y: Target data tensor.
            n_context (int): Number of context points.
            axis: Dimension axis as int

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - x_c: Context input data.
                - y_c: Context target data.
                - x_t: Target input data.
                - y_t: Target target data.
        """
    
        ind = torch.arange(x.shape[axis])
        mask = torch.randperm(ind.size(0))[:n_context]  
        x_c = x.index_select(axis, mask)
        y_c = y.index_select(axis, mask)
        x_t = torch.index_select(x, axis, torch.setdiff1d(ind, mask))
        y_t = torch.index_select(y, axis, torch.setdiff1d(ind, mask))

        return x_c, y_c, x_t, y_t
    