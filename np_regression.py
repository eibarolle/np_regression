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
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from typing import Callable, List, Optional, Tuple
import torch.nn as nn
from sklearn import preprocessing
from scipy.stats import multivariate_normal

device = torch.device("cpu")


"""# CNP"""

#reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        init_func: Optional[Callable] = torch.nn.init.normal_,
        l1_size: int = 16,
        l2_size: int = 8
    ) -> None:
        r"""Encodes inputs of the form (x_i,y_i) into representations, r_i.

        Args:
            in_dim: An int representing the total input dimensionality.
            out_dim: An int representing the total encoded dimensionality.
            init_func: A function initializing the weights.
            l1_size: An int representing the L1 Regression size.
            l2_size: An int representing the L2 Regression size.
        """
        super(REncoder, self).__init__()
        
        self.l1 = torch.nn.Linear(in_dim, l1_size)
        self.l2 = torch.nn.Linear(l1_size, l2_size)
        self.l3 = torch.nn.Linear(l2_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
        
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
        return self.l3(self.a2(self.l2(self.a1(self.l1(inputs)))))

class ZEncoder(torch.nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        init_func: Optional[Callable] = torch.nn.init.normal_,
    ) -> None:
        r"""Takes an r representation and produces the mean & standard 
        deviation of the normally distributed function encoding, z.
        
        Args:
            in_dim: An int representing r's aggregated dimensionality.
            out_dim: An int representing z's latent dimensionality.
            init_func: A function initializing the weights.
        """
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.logvar1_size = out_dim
        
        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.logvar1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.logvar1.weight)
        
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
        return self.m1(inputs), self.logvar1(inputs)
    
class Decoder(torch.nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        init_func: Optional[Callable] = torch.nn.init.normal_,
        l1_size: int = 8,
        l2_size: int = 16
    ) -> None:
        r"""Takes the x star points, along with a 'function encoding', z, and makes predictions.
        
        Args:
            in_dim: An int representing the total input dimensionality.
            out_dim: An int representing the predicted outputs' dimensionality.
            init_func: A function initializing the weights.
            l1_size: An int representing the L1 Regression size.
            l2_size: An int representing the L2 Regression size.
        """
        super(Decoder, self).__init__()
        
        self.l1 = torch.nn.Linear(in_dim, l1_size)
        self.l2 = torch.nn.Linear(l1_size, l2_size)
        self.l3 = torch.nn.Linear(l2_size, out_dim)
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
        
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        
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
        zs_reshaped = z.unsqueeze(-1).expand(z.shape[0], x_pred.shape[0]).transpose(0,1)
        xpred_reshaped = x_pred
        
        xz = torch.cat([xpred_reshaped, zs_reshaped], dim=1)

        return self.l3(self.a2(self.l2(self.a1(self.l1(xz))))).squeeze(-1)

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

class NeuralProcessModel(nn.Module):
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
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
            x_dim: Int dimensionality of input data x.
            y_dim: Int dimensionality of target data y.
            r_dim: Int dimensionality of representation r.
            z_dim: Int dimensionality of latent variable z.
            init_func: A function initializing thee weights.
        """
        super().__init__()
        self.repr_encoder = REncoder(x_dim+y_dim, r_dim) 
        self.z_encoder = ZEncoder(r_dim, z_dim) 
        self.decoder = Decoder(x_dim + z_dim, y_dim) 
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train(n_epochs, x_train, y_train)
    
    def data_to_z_params(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute latent parameters from inputs.

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
        rs = self.repr_encoder(xy)
        r_agg = rs.mean(dim=0) # Average over samples
        return self.z_encoder(r_agg) # Get mean and variance for q(z|...)
    
    def sample_z(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        r"""Reparameterization trick for z.

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
        
        # std = torch.exp(0.5 * logvar)
        std = 0.1+ 0.9*torch.sigmoid(logvar)
        return mu + std * eps

    def KLD_gaussian(self) -> torch.Tensor:
        r"""Analytical KLD between 2 Gaussian Distributions.

        Returns:
            torch.Tensor: A tensor representing the KLD.
        """
        mu_q, logvar_q, mu_p, logvar_p = self.z_mu_all, self.z_logvar_all, self.z_mu_context, self.z_logvar_context

        std_q = 0.1+ 0.9*torch.sigmoid(logvar_q)
        std_p = 0.1+ 0.9*torch.sigmoid(logvar_p)
        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        return torch.distributions.kl_divergence(p, q).sum()
        

    def forward(
        self,
        x_t: torch.Tensor,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_ct: torch.Tensor,
        y_ct: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass for the model.

        Args:
            x_t: Target input data.
            x_c: Context input data.
            y_c: Context target data.
            x_ct: Combined input data.
            y_ct: Combined target data.

        Returns:
            torch.Tensor: Predicted target values.
        """
        
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x_ct, y_ct)
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, self.zs)
    
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
    
    def train(
        self,
        n_epochs: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        n_display: int = 500,
        N = 100000,
    ) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
        r"""Training loop for the NP model.

        Args:
            n_epochs: An int representing the # of training epochs.
            x_train: Training input data.
            y_train: Training target data.
            n_display: Frequency of logs.
            N: An int representing population size.

        Returns:
            Tuple[List[float], torch.Tensor, torch.Tensor]: 
                - train_losses: Recorded training losses.
                - z_mu_all: Posterior mean of z.
                - z_logvar_all: Posterior mog variance of z.
        """
        train_losses = []
        
        for t in range(n_epochs): 
            self.optimizer.zero_grad()
            #Generate data and process
            x_context, y_context, x_target, y_target = self.random_split_context_target(
                                    x_train, y_train, int(len(y_train)*0.1)) #0.25, 0.5, 0.05,0.015, 0.01
            # print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)    

            x_c = torch.from_numpy(x_context).float().to(device)
            x_t = torch.from_numpy(x_target).float().to(device)
            y_c = torch.from_numpy(y_context).float().to(device)
            y_t = torch.from_numpy(y_target).float().to(device)

            x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
            y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

            y_pred = self.forward(x_t, x_c, y_c, x_ct, y_ct)

            train_loss = N * MAE(y_pred, y_t)/100 + self.KLD_gaussian()
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5) #10
            self.optimizer.step()

            if t % (n_display/10) ==0:
                train_losses.append(train_loss.item())
            
        return train_losses, self.z_mu_all, self.z_logvar_all
    