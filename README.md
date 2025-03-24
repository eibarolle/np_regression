# NP Regression Implementation

### Created By: Ernest Ibarolle

### Original Study/Reference: https://arxiv.org/pdf/2106.02770

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
- forward(self, inputs)
  - Forward pass for the representation encoder.
  - Parameters:
    - inputs (torch.Tensor)- Input tensor.
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
      
- forward(self, inputs)
  - Forward pass for latent encoder.
  - Parameters:
    - inputs (torch.Tensor)- Input tensor.
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

## class NeuralProcessModel(self, train_X, train_Y, r_hidden_dims = [16, 16], z_hidden_dims = [32, 32], decoder_hidden_dims = [16, 16], x_dim = 2, y_dim = 1, r_dim = 64, z_dim = 8, n_context = 20, activation = nn.Sigmoid, init_func = torch.nn.init.normal_, likelihood = Likelihood, input_transform = InputTransform) 
- Base: botorch.models.model.Model
- Diffusion Convolutional Recurrent Neural Network Model Implementation.
- Parameters:
  - train_X (torch.Tensor) - A 'batch_shape x n x d' tensor of training features.
  - train_Y (torch.Tensor) - A 'batch_shape x n x d' tensor of training observations.    
  - r_hidden_dims (List[int]) - Hidden dimensions/layers for the REncoder, defaults to [16, 16].
  - z_hidden_dims (List[int]) - Hidden dimensions/layers for the ZEncoder, defaults to [32, 32].
  - decoder_hidden_dims (List[int]) - Hidden dimensions/layers for the Decoder, defaults to [16, 16].
  - x_dim (int) - Dimensionality of input data x, defaults to 2.
  - y_dim (int) - Dimensionality of target data y, defaults to 1.
  - r_dim (int) - Dimensionality of representation r, defaults to 64.
  - z_dim (int) - Dimensionality of latent variable z, defaults to 8.
  - n_context (int) - Number of context points, defaults to 20.
  - activation(Callable)- Activation function applied between layers, defaults to nn.Sigmoid.
  - init_func (Optional[Callable]) - Function for initializing weights. Defaults to torch.nn.init.normal_.
  - likelihood (Likelihood) - A likelihood distribution. If omitted, use a standard GaussianLikelihood.
  - input_transform (InputTransform) - An input transform that is applied in the model's forward pass.
- Property r_encoder: REncoder
  - The Representation Encoder of the model.
- Property z_encoder: ZEncoder
  - The Latent ZEncoder of the model.
- Property decoder: Decoder
  - The Decoder of the model.
- Property z_dim: int
  - The z_dim of the model.  
- Property z_mu_all: torch.Tensor
  - Gaussian Distribution Latent Mean of the full data.
- Property z_logvar_all: torch.Tensor
  - Gaussian Distribution Latent Log Variance of the full data.
- Property z_mu_context: torch.Tensor
  - Gaussian Distribution Latent Mean of the context data.
- Property z_logvar_context: torch.Tensor
  - Gaussian Distribution Latent Log Variance of the context data.
- Property likelihood: Likelihood
  - Likelihood distributon used for observation noise.
- Property input_transform: InputTransform
  - Input Transformer applied in the forward pass.

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

- sample_z(mu, logvar, n = 1, min_std = 0.01, scaler = 0.5)
  - Reparameterization trick for sampling from the latent Gaussian distribution.
  - Parameters:
    - mu (torch.Tensor)- Mean of the Gaussian distribution.
    - logvar (torch.Tensor)- Log variance of the Gaussian distribution.
    - n (int)- # of samples. Defaults to 1.
    - min_std (float)- Minimum standard deviation. Defaults to 0.01.
    - scaler (float)- Scaling factor for the standard deviation. Defaults to 0.5.
  - Returns:
    - Samples from the Gaussian distribution.
  - Return type:
    - torch.Tensor

- KLD_gaussian(min_std = 0.01, scaler = 0.5)
  - Analytical KL divergence between two Gaussian distributions.
  - Parameters:
    - min_std (float)- Minimum standard deviation. Defaults to 0.01.
    - scaler (float)- Scaling factor for the standard deviation. Defaults to 0.5.
  - Returns:
    - KL divergence value.
  - Return type:
    - torch.Tensor

- posterior(X, covariance_multiplier, output_indices, observation_noise = False, posterior_transform = None)
  - Computes the model's posterior distribution for given input tensors.
  - Parameters:
    - X (torch.Tensor) - Input tensor.
    - covariance_multiplier (float) - Scaling factor for the covariance matrix.
    - output_indices (list_int) - Ignored (defined in parent Model, but not used here).
    - observation_constant (float) - Noise constant added to the covariance matrix.
    - observation_noise (bool) - If True, adds observation noise to the covariance matrix, defaults to False.
    - posterior_transform (Optional[PosteriorTransform])- Transformation applied to the posterior distribution, defaults to None.
  - Returns:
    - The posterior distribution object utilizing MultivariateNormal.
  - Return type:
    - GPyTorchPosterior

- transform_inputs(X, input_transform = None)
  - Transform inputs.
  - Parameters:
    - X (torch.Tensor)- Input tensor.
    - input_transform (Optional[Module])- A Module that performs the input transformation.
  - Returns:
    - A tensor of transformed inputs
  - Return type:
    - torch.Tensor

- forward(train_X, train_Y, axis = 0)
  - Forward pass for the model.
  - Parameters:
    - train_X (torch.Tensor) - A 'batch_shape x n x d' tensor of training features.
    - train_Y (torch.Tensor) - A 'batch_shape x n x d' tensor of training observations.
    - axis (int)- Dimension axis as int. Defaults to 0.
  - Returns:
    - Predicted target value distribution.
  - Return type:
    - MultivariateNormal

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
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]



## class LatentInformationGain(model, num_samples = 10, min_std = 0.01, scaler = 0.5)
- Latent Information Gain (LIG) Acquisition Function, designed for the NeuralProcessModel.
- Bases: botorch.acquisition.AcquisitionFunction
- Parameters:
  - model (Type[Any]) - Trained model.
  - num_samples (int) - Number of samples for calculation, defaults to 10.
  - min_std (float)- The minimum possible standardized std, defaults to 0.01.
  - scaler (float)- Scaling the std, defaults to 0.5.
- def acquisition(self, candidate_x):
  - Conduct the Latent Information Gain acquisition function for the inputs.
  - Parameters:
    - candidate_x (torch.Tensor): Candidate input points, as a Tensor. Ideally in the shape (N, q, D), and assumes N = 1 if the given dimensions are 2D.
  - Returns:
    - The LIG scores of computed KLDs, in the shape (N, q).
  - Return type:
    - torch.Tensor
