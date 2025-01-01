import unittest
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from botorch_community.models import NeuralProcessModel
from botorch.posteriors import GPyTorchPosterior
from torch import Tensor

class TestNeuralProcessModel(unittest.TestCase):
    def initialize(self):
        self.x_dim = 2
        self.y_dim = 1
        self.r_dim = 8
        self.z_dim = 8
        self.model = NeuralProcessModel(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            r_dim=self.r_dim,
            z_dim=self.z_dim,
        )
        self.x_data = np.random.rand(100, self.x_dim)
        self.y_data = np.random.rand(100, self.y_dim)

    def test_r_encoder(self):
        input = torch.rand(10, self.x_dim + self.y_dim)
        output = self.model.r_encoder(input)
        self.assertEqual(output.shape, (10, self.r_dim))
        self.assertTrue(torch.is_tensor(output))

    def test_z_encoder(self):
        input = torch.rand(10, self.r_dim)
        mean, logvar = self.model.z_encoder(input)
        self.assertEqual(mean.shape, (10, self.z_dim))
        self.assertEqual(logvar.shape, (10, self.z_dim))
        self.assertTrue(torch.is_tensor(mean))
        self.assertTrue(torch.is_tensor(logvar))

    def test_decoder(self):
        x_pred = torch.rand(10, self.x_dim)
        z = torch.rand(self.z_dim)
        output = self.model.decoder(x_pred, z)
        self.assertEqual(output.shape, (10, self.y_dim))
        self.assertTrue(torch.is_tensor(output))

    def test_sample_z(self):
        mu = torch.rand(self.z_dim)
        logvar = torch.rand(self.z_dim)
        samples = self.model.sample_z(mu, logvar, n=5)
        self.assertEqual(samples.shape, (5, self.z_dim))
        self.assertTrue(torch.is_tensor(samples))

    def test_KLD_gaussian(self):
        self.model.z_mu_all = torch.rand(self.z_dim)
        self.model.z_logvar_all = torch.rand(self.z_dim)
        self.model.z_mu_context = torch.rand(self.z_dim)
        self.model.z_logvar_context = torch.rand(self.z_dim)
        kld = self.model.KLD_gaussian()
        self.assertGreaterEqual(kld.item(), 0)
        self.assertTrue(torch.is_tensor(kld))

    def test_data_to_z_params(self):
        x = torch.rand(10, self.x_dim)
        y = torch.rand(10, self.y_dim)
        mu, logvar = self.model.data_to_z_params(x, y)
        self.assertEqual(mu.shape, (self.z_dim,))
        self.assertEqual(logvar.shape, (self.z_dim,))
        self.assertTrue(torch.is_tensor(mu))
        self.assertTrue(torch.is_tensor(logvar))

    def test_forward(self):
        x_t = torch.rand(10, self.x_dim)
        x_c = torch.rand(5, self.x_dim)
        y_c = torch.rand(5, self.y_dim)
        x_ct = torch.cat([x_c, x_t], dim=0)
        y_ct = torch.cat([y_c, torch.rand(10, self.y_dim)], dim=0)
        output = self.model(x_t, x_c, y_c, x_ct, y_ct)
        self.assertEqual(output.shape, (10, self.y_dim))

    def test_random_split_context_target(self):
        x_c, y_c, x_t, y_t = self.model.random_split_context_target(
            self.x_data, self.y_data, n_context=20
        )
        self.assertEqual(x_c.shape[0], 20)
        self.assertEqual(y_c.shape[0], 20)
        self.assertEqual(x_t.shape[0], 80)
        self.assertEqual(y_t.shape[0], 80)

    # def test_train(self):
    #     x_train = self.x_data
    #     y_train = self.y_data
    #     losses, z_mu, z_logvar = self.model.train(
    #         n_epochs=10, x_train=x_train, y_train=y_train, n_display=2
    #     )
    #     self.assertIsInstance(losses, list)
    #     self.assertTrue(len(losses) > 0)
    #     self.assertTrue(all(isinstance(loss, float) for loss in losses))
    #     self.assertEqual(z_mu.shape, (self.z_dim,))
    #     self.assertEqual(z_logvar.shape, (self.z_dim,))
        
    def test_posterior(self):
        X = torch.rand(5, 3)
        posterior = self.model.posterior(X, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        mvn = posterior.mvn
        self.assertTrue(torch.allclose(mvn.mean, torch.zeros(5)))
        self.assertEqual(mvn.covariance_matrix.size(), (5, 5))
        expected_covariance = torch.eye(5) * 0.11
        self.assertTrue(torch.allclose(mvn.covariance_matrix, expected_covariance))
    
    def test_condition_on_observations(self):
        X = torch.rand(10, 3)
        Y = torch.rand(10, 1)
        self.model.condition_on_observations(X, Y)
        self.assertTrue(torch.equal(self.model.train_data[0], X))
        self.assertTrue(torch.equal(self.model.train_data[1], Y))
    
    def test_load_state_dict(self):
        state_dict = {"param1": torch.tensor([1.0]), "param2": torch.tensor([2.0])}
        self.model.load_state_dict(state_dict)
        self.assertTrue(torch.equal(self.model.param1, torch.tensor([1.0])))
        self.assertTrue(torch.equal(self.model.param2, torch.tensor([2.0])))
    
    def test_transform_inputs(self):
        X = torch.rand(5, 3)
        self.assertTrue(torch.equal(self.model.transform_inputs(X), X * 2))
    

if __name__ == "__main__":
    unittest.main()
