r"""
Neural Process Regression models based on PyTorch models.

References:

.. [Wu2023arxiv]
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Contributor: eibarolle
"""

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

device = torch.device("cpu")

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
            activation: Activation function applied between layers.
            init_func: A function initializing the weights.
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
        hidden_dims: List[int]
    ) -> None:
        r"""Encodes inputs of the form (x_i,y_i) into representations, r_i.

        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
        """
        super().__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dims)
        
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
    ) -> None:
        r"""Takes an r representation and produces the mean & standard 
        deviation of the normally distributed function encoding, z.
        
        Args:
            input_dim: An int representing r's aggregated dimensionality.
            output_dim: An int representing z's latent dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
        """
        super().__init__()
        self.mean_net = MLP(input_dim, output_dim, hidden_dims)
        self.logvar_net = MLP(input_dim, output_dim, hidden_dims)
        
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
        hidden_dims: List[int]
    ) -> None:
        r"""Takes the x star points, along with a 'function encoding', z, and makes predictions.
        
        Args:
            input_dim: An int representing the total input dimensionality.
            output_dim: An int representing the total encoded dimensionality.
            hidden_dims: A list of integers representing the # of units in each hidden dimension.
        """
        super().__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dims)
        
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
        pred: The redicted values tensor.
        target: The target values tensor.

    Returns:
        torch.Tensor: A tensor representing the MAE.
    """
    loss = torch.abs(pred-target)
    return loss.mean()

class NeuralProcessModel(Model):
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        r_hidden_dims: List[int], 
        z_hidden_dims: List[int], 
        decoder_hidden_dims: List[int],
        n_epochs: int = 5000,
        x_dim: int = 2,
        y_dim: int = 100,
        r_dim: int = 8,
        z_dim: int = 8,
        init_func: Optional[Callable] = torch.nn.init.normal_,
    ) -> None:
        r"""Diffusion Convolutional Recurrent Neural Network Model Implementation.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            r_hidden_dims: Hidden Dimensions/Layer list for REncoder
            z_hidden_dims: Hidden Dimensions/Layer list for ZEncoder
            decoder_hidden_dims: Hidden Dimensions/Layer for Decoder
            x_dim: Int dimensionality of input data x.
            y_dim: Int dimensionality of target data y.
            r_dim: Int dimensionality of representation r.
            z_dim: Int dimensionality of latent variable z.
            init_func: A function initializing thee weights.
        """
        super().__init__()
        self.r_encoder = REncoder(x_dim+y_dim, r_dim, r_hidden_dims) 
        self.z_encoder = ZEncoder(r_dim, z_dim, z_hidden_dims) 
        self.decoder = Decoder(x_dim + z_dim, y_dim, decoder_hidden_dims) 
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #self.train(n_epochs, x_train, y_train)
    
    def data_to_z_params(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute latent parameters from inputs as a latent distribution.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - x_c: Context input data.
                - y_c: Context target data.
                - x_t: Target input data.
                - y_t: Target target data.
        """
        xy = torch.cat([x,y], dim=1)
        rs = self.r_encoder(xy)
        r_agg = rs.mean(dim=0)
        return self.z_encoder(r_agg) 
    
    def sample_z(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        r"""Reparameterization trick for z's latent distribution.

        Args:
            mu: Tensor representing the Gaussian distribution mean.
            logvar: Tensor representing the log variance of the Gaussian distribution.
            n: Int representing the # of samples.
            z_dim: Int dimensionality of latent variable z.

        Returns:
            torch.Tensor: Samples from the Gaussian distribution.
    """
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(self.z_dim).normal_()).to(device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n,self.z_dim).normal_()).to(device)
        
        std = 0.1 + 0.9 * torch.sigmoid(logvar)
        return mu + std * eps

    def KLD_gaussian(self) -> torch.Tensor:
        r"""Analytical KLD between 2 Gaussian Distributions.

        Returns:
            torch.Tensor: A tensor representing the KLD.
        """
        std_q = 0.1+ 0.9*torch.sigmoid(self.z_logvar_all)
        std_p = 0.1+ 0.9*torch.sigmoid(self.z_logvar_context)
        p = torch.distributions.Normal(self.z_mu_context, std_p)
        q = torch.distributions.Normal(self.z_mu_all, std_q)
        return torch.distributions.kl_divergence(p, q).sum()
    
    def posterior(
        self, 
        X: torch.Tensor, 
        observation_noise: bool = False, 
        posterior_transform: Optional[PosteriorTransform] = None
    ) -> GPyTorchPosterior:
        r"""Computes the model's posterior distribution for given input tensors.

        Args:
            X: Input Tensor
            observation_noise: Adds observation noise to the covariance if true.
            posterior_transform: An optional posterior transformation.

        Returns:
            GPyTorchPosterior: The posterior distribution object utilizing
            GPyTorch and MultivariateNormal.
        """
        mean = self.decoder(X, self.sample_z())
        covariance = torch.eye(X.size(0)) * 0.1
        if (observation_noise):
            covariance = covariance + 0.01
        mvn = MultivariateNormal(mean, covariance)
        return GPyTorchPosterior(mvn)
        
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
        self.train_data = (X, Y)
        return self
    
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
    ) -> torch.Tensor:
        r"""Forward pass for the model.

        Args:
            x_t: Target input data.
            x_c: Context input data.
            y_c: Context target data.

        Returns:
            torch.Tensor: Predicted target values.
        """
        
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(torch.cat([x_c, x_t]), torch.cat([y_c, x_t]))
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        z = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, z)
    
    def random_split_context_target(
        x: torch.Tensor,
        y: torch.Tensor,
        n_context: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Helper function to split randomly into context and target.

        Args:
            x: Input data tensor.
            y: Target data tensor.
            n_context (int): Number of context points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - x_c: Context input data.
                - y_c: Context target data.
                - x_t: Target input data.
                - y_t: Target target data.
        """
        ind = np.arange(x.shape[0])
        mask = np.random.choice(ind, size=n_context, replace=False)
        return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)
    
    # def train(
    #     self,
    #     n_epochs: int,
    #     x_train: torch.Tensor,
    #     y_train: torch.Tensor,
    #     n_display: int = 500,
    #     N = 100000,
    # ) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
    #     r"""Training loop for the NP model.

    #     Args:
    #         n_epochs: An int representing the # of training epochs.
    #         x_train: Training input data.
    #         y_train: Training target data.
    #         n_display: Frequency of logs.
    #         N: An int representing population size.

    #     Returns:
    #         Tuple[List[float], torch.Tensor, torch.Tensor]: 
    #             - train_losses: Recorded training losses.
    #             - z_mu_all: Posterior mean of z.
    #             - z_logvar_all: Posterior mog variance of z.
    #     """
    #     train_losses = []
        
    #     for t in range(n_epochs): 
    #         self.optimizer.zero_grad()
    #         #Generate data and process
    #         x_context, y_context, x_target, y_target = self.random_split_context_target(
    #                                 x_train, y_train, int(len(y_train)*0.1)) #0.25, 0.5, 0.05,0.015, 0.01
    #         # print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)    

    #         x_c = torch.from_numpy(x_context).float().to(device)
    #         x_t = torch.from_numpy(x_target).float().to(device)
    #         y_c = torch.from_numpy(y_context).float().to(device)
    #         y_t = torch.from_numpy(y_target).float().to(device)

    #         x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
    #         y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

    #         y_pred = self.forward(x_t, x_c, y_c, x_ct, y_ct)

    #         train_loss = N * MAE(y_pred, y_t)/100 + self.KLD_gaussian()
            
    #         train_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #10
    #         self.optimizer.step()

    #         if t % (n_display/10) ==0:
    #             train_losses.append(train_loss.item())
            
    #     return train_losses, self.z_mu_all, self.z_logvar_all
    