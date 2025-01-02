# NP Regression Implementation

## NeuralProcessModel
Neural Process (NP) is a type of deep generative model that represents distributions over functions. It introduces a global latent variable $z$ to capture the stochasticity and learns the conditional distribution $p(x_{1:T} | \theta)$ by optimizing the evidence lower bound (ELBO):

Here, $p(z)$ is the prior distribution for the latent variable. We use $x_{1:T}$ as a shorthand for $(x_1, \dots, x_T)$. The prior distribution $p(z)$ is conditioned on a set of context points $\theta_c$, $x_c^{1:T}$ as $p(z | x_c^{1:T}, \theta_c)$.

For more applicable spatiotemporal computations, the Latent Information Gain acquisition function is utilized to calculate the expected KL Divergences.

## Latent Information Gain

In the high-dimensional spatiotemporal domain, Expected Information Gain becomes less informative for useful observations, and it can be difficult to calculate its parameters. To overcome these limitations, we propose a novel acquisition function by computing the expected information gain in the latent space rather than the observational space. To design this acquisition function, we prove the equivalence between the expected information gain in the observational space and the expected KL divergence between the latent process' prior and posterior, that is

$$ \text{EIG}(\hat{x}_{1:T}, \theta) := \mathbb{E} \left[ H(\hat{x}_{1:T}) - H(\hat{x}_{1:T} \mid z_{1:T}, \theta) \right] = \mathbb{E}_{p(\hat{x}_{1:T} \mid \theta)} \text{KL} \left( p(z_{1:T} \mid \hat{x}_{1:T}, \theta) \,\|\, p(z_{1:T}) \right) $$


Inspired by this fact, we propose a novel acquisition function computing the expected KL divergence in the latent processes and name it LIG. Specifically, the trained NP model produces a variational posterior given the current dataset. For every parameter $$\theta$$ remained in the search space, we can predict $$\hat{x}_{1:T}$$ with the decoder. We use $$\hat{x}_{1:T}$$ and $$\theta$$ as input to the encoder to re-evaluate the posterior. LIG computes the distributional difference with respect to the latent process.

[Wu2023arxiv]:
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Created By: Ernest Ibarolle

Original Study/Reference: https://arxiv.org/pdf/2106.02770
