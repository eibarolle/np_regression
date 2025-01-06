# NP Regression Implementation

## class MLP(input_dim, output_dim, hidden_dims, activation = nn.Sigmoid, init_func = nn.init.normal_)
- Bases: nn.Module
- A modular implementation of a Multilayer Perceptron (MLP).
- Parameters:
  - input_dim (int)- Input dimensionality.
  - output_dim (int)- Output dimensionality.
  - hidden_dims (List[int])- List specifying the number of units in each hidden layer
  - activation (Callable)- Activation function applied between layers. Defaults to nn.Sigmoid
  - init_func (Optional[Callable])- Function to initialize weights. Defaults to nn.init.normal_

- forward(self, x)
  - Performs the forward pass.
  - Parameters:
    - x (torch.Tensor)- Input tensor.
  - Returns:
    - A tensor representing the model after applying the MLP.
  - Return type:
    - torch.Tensor
      
## class REncoder(input_dim, output_dim, hidden_dims, activation = nn.Sigmoid, init_func = nn.init.normal_)
- Bases: nn.Module
- Encodes inputs of the form (x_i,y_i) into representations, r_i.
- Parameters:
  - input_dim(int)- Total input dimensionality.
  - output_dim(int)- Total encoded dimensionality.
  - hidden_dims(List[int])- # of units in each hidden dimension.
  - activation(Callable)- Activation function applied between layers, defaults to nn.Sigmoid.
  - init_func(Optional[Callable])- A function initializing the weights, defaults to nn.init.normal_.
- forward(self, x)
  - Forward pass for the representation encoder.
  - Parameters:
    - x (torch.Tensor)- Input tensor.
  - Returns:
    - A tensor representing the encoded representations of the REncoder.
  - Return type:
    - torch.Tensor
            
## class ZEncoder(input_dim, output_dim, hidden_dims, activation = nn.Sigmoid, init_func = nn.init.normal_)
- Bases: nn.Module
- Takes an r representation and produces the mean & standard deviation of the normally distributed function encoding, z.
- Parameters:
  - input_dim(int)- r's aggregated dimensionality.
  - output_dim(int)- z's latent dimensionality.
  - hidden_dims(List[int])- # of units in each hidden dimension.
  - activation(Callable)- Activation function applied between layers, defaults to nn.Sigmoid.
  - init_func(Optional[Callable])- A function initializing the weights, defaults to nn.init.normal_.
      
- forward(self, x)
  - Forward pass for latent encoder.
  - Parameters:
    - x (torch.Tensor)- Input tensor.
    - Returns:
      - A tuple consisting of the mean of the latent Gaussian distribution and the log variance of the latent Gaussian distribution.
    - Return type:
      - Tuple[torch.Tensor, torch.Tensor]

## class Decoder(input_dim, output_dim, hidden_dims, activation = nn.Sigmoid, init_func = nn.init.normal_)
- Bases: nn.Module
- Takes the x star points, along with a 'function encoding', z, and makes predictions.
- Parameters:
  - input_dim(int)- Total input dimensionality.
  - output_dim(int)- Total encoded dimensionality.
  - hidden_dims (List[int])- # of units in each hidden dimension.
  - activation(Callable)- Activation function applied between layers, defaults to nn.Sigmoid.
  - init_func(Optional[Callable])- A function initializing the weights, defaults to nn.init.normal_.
      
- forward(self, x_pred, z)
  - Forward pass for the decoder.
  - Parameters:
    - x_pred (torch.Tensor)- No. of data points, by x_dim
    - z (torch.Tensor)- No. of samples, by z_dim
    - Returns:
      - A tensor representing the model after applying the MLP.
    - Return type:
      - torch.Tensor
            
## MAE(pred, target) 
- Mean Absolute Error loss function.
- Parameters:
  - pred (torch.Tensor)- Predicted values.
  - target (torch.Tensor) - Target values.
- Returns:
  - A tensor representing the Mean Absolute Error (MAE).
- Return type:
  - torch.Tensor

## class NeuralProcessModel(self, r_hidden_dims, z_hidden_dims, decoder_hidden_dims, x_dim, y_dim, r_dim, z_dim, activation = nn.Sigmoid, init_func = torch.nn.init.normal_) 
- Base: botorch.models.model.Model
- Diffusion Convolutional Recurrent Neural Network Model Implementation.
- Parameters:
  - r_hidden_dims (List[int]) - Hidden dimensions/layers for the REncoder.
  - z_hidden_dims (List[int]) - Hidden dimensions/layers for the ZEncoder.
  - decoder_hidden_dims (List[int]) - Hidden dimensions/layers for the Decoder.
  - x_dim (int) - Dimensionality of input data x.
  - y_dim (int) - Dimensionality of target data y.
  - r_dim (int) - Dimensionality of representation r.
  - z_dim (int) - Dimensionality of latent variable z.
  - activation(Callable)- Activation function applied between layers, defaults to nn.Sigmoid.
  - init_func (Optional[Callable]) - Function for initializing weights. Defaults to torch.nn.init.normal_.
- Property r_encoder: REncoder
  - The Representation Encoder of the model.
- Property z_encoder: ZEncoder
  - The Latent ZEncoder of the model.
- Property decoder: Decoder
  - The Decoder of the model.
- Property z_mu_all: torch.Tensor
  - Gaussian Distribution Latent Mean of the full data.
- Property z_logvar_all: torch.Tensor
  - Gaussian Distribution Latent Log Variance of the full data.
- Property z_mu_context: torch.Tensor
  - Gaussian Distribution Latent Mean of the context data.
- Property z_logvar_context: torch.Tensor
  - Gaussian Distribution Latent Log Variance of the context data.

- data_to_z_params(x, y, xy_dim = 1, r_dim = 0)
  - Compute latent parameters from inputs as a latent distribution.
  - Parameters:
    - x (torch.Tensor)- Input tensor.
    - y (torch.Tensor)- Target tensor.
    - xy_dim (int)- Combined Input Dimension, defaults as 1.
    - r_dim (int)-  Combined Target Dimension, defaults as 0.
  - Returns:
    - The mean and log variance of the latent Gaussian distribution as a tuple.
  - Return type:
    - Tuple[torch.Tensor, torch.Tensor] 

- sample_z(mu, logvar, n = 1, min_std = 0.1, scaler = 0.9)
  - Reparameterization trick for sampling from the latent Gaussian distribution.
  - Parameters:
    - mu (torch.Tensor)- Mean of the Gaussian distribution.
    - logvar (torch.Tensor)- Log variance of the Gaussian distribution.
    - n (int)- # of samples. Defaults to 1.
    - min_std (float)- Minimum standard deviation. Defaults to 0.1.
    - scaler (float)- Scaling factor for the standard deviation. Defaults to 0.9.
  - Returns:
    - Samples from the Gaussian distribution.
  - Return type:
    - torch.Tensor

- KLD_gaussian(min_std = 0.1, scaler = 0.9)
  - Analytical KL divergence between two Gaussian distributions.
  - Parameters:
    - min_std (float)- Minimum standard deviation. Defaults to 0.1.
    - scaler (float)- Scaling factor for the standard deviation. Defaults to 0.9.
  - Returns:
    - KL divergence value.
  - Return type:
    - torch.Tensor

- posterior(X, covariance_multiplier, observation_constant, observation_noise = False, posterior_transform = None)
  - Computes the model's posterior distribution for given input tensors.
  - Parameters:
    - X (torch.Tensor)- Input tensor.
    - covariance_multiplier (float)- Scaling factor for the covariance matrix.
    - observation_constant (float)- Noise constant added to the covariance matrix.
    - observation_noise (bool)- If True, adds observation noise to the covariance matrix, defaults to False.
    - posterior_transform (Optional[PosteriorTransform])- Transformation applied to the posterior distribution, defaults to None.
  - Returns:
    - The posterior distribution object utilizing MultivariateNormal.
  - Return type:
    - MultivariateNormalPosterior

- condition_on_observations(X, Y)
  - Condition the model on new observations.
  - Parameters:
    - X (torch.Tensor)- Input tensor.
    - Y (torch.Tensor)- Target tensor.
  - Returns:
    - New instance of the model conditioned on the given data.
  - Return type:
    - NeuralProcessModel

- load_state_dict(state_dict, strict = True)
  - Initialize the fully Bayesian model before loading the state dictionary.
  - Parameters:
    - state_dict (dict)- Dictionary containing the parameters.
    - strict (bool)- Whether to strictly enforce matching keys. Defaults to True.

- transform_inputs(X, input_transform = None)
  - Transform inputs.
  - Parameters:
    - X (torch.Tensor)- Input tensor.
    - input_transform (Optional[Module])- A Module that performs the input transformation.
  - Returns:
    - A tensor of transformed inputs
  - Return type:
    - torch.Tensor

- forward(x_t, x_c, y_c, input_dim = 0, target_dim = 1)
  - Forward pass for the model.
  - Parameters:
    - x_t (torch.Tensor)- Target input tensor.
    - x_c (torch.Tensor)- Context input tensor.
    - y_c (torch.Tensor)- Context target tensor.
    - input_dim (int)- Dimension along which input data is concatenated. Defaults to 0.
    - target_dim (int)- Dimension along which target data is concatenated. Defaults to 1.
  - Returns:
    - Predicted target values.
  - Return type:
    - torch.Tensor

- random_split_context_target(x, y, n_context, axis)
  - Helper function to split data into context and target sets.
  - Parameters:
    - x (torch.Tensor)- Input data tensor.
    - y (torch.Tensor)- Target data tensor.
    - n_context (int)- Number of context points.
    - axis (int)- Dimension axis
  - Returns:
    - Context and target data tuples:
      - x_c: Context input data.
      - y_c: Context target data.
      - x_t: Target input data.
      - y_t: Target target data.
  - Return type:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]



## class LatentInformationGain(model, num_samples = 10, min_std = 0.1, scaler = 0.9)
- Latent Information Gain (LIG) Acquisition Function, designed for the NeuralProcessModel.
- Parameters:
  - model (NeuralProcessModel)- Trained NeuralProcessModel.
  - num_samples (int)- Number of samples for calculation, defaults to 10.
  - min_std (float)- The minimum possible standardized std, defaults to 0.1.
  - scaler (float)- Scaling the std, defaults to 0.9.
- def acquisition(self, candidate_x, context_x, context_y):
  - Conduct the Latent Information Gain acquisition function for the inputs.
  - Parameters:
    - candidate_x (torch.Tensor): Candidate input points.
    - context_x (torch.Tensor): Context input points.
    - context_y (torch.Tensor): Context target points
  - Returns:
    - The LIG score of computed KLDs.
  - Return type:
    - torch.Tensor

[Wu2023arxiv]:
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Created By: Ernest Ibarolle

Original Study/Reference: https://arxiv.org/pdf/2106.02770
