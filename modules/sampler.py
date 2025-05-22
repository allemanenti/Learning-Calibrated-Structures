from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from torch.distributions import Bernoulli

from torch_geometric.utils import dense_to_sparse

from einops import rearrange

from utils.utils import inverse_sigmoid

class ParametrizedBernoulliSampler(Module):
    def __init__(self, 
                 num_nodes: int,
                 gradient_estimator: str,
                 initial_probs: Union[float, Tensor] = 0.05,
                 ):
        super(ParametrizedBernoulliSampler, self).__init__()

        self.num_nodes = num_nodes
        self.gradient_estimator = gradient_estimator

        if gradient_estimator == 'st':
            self.dense_full_ed_idx = self.compute_full_ei(num_nodes)

        # if initial theta is a float, randomly initialize adj around it
        if isinstance(initial_probs, float):
            if initial_probs < 0.05:
                initial_noise = initial_probs
            else:
                initial_noise = 0.05
            initial_probs = torch.ones(num_nodes, num_nodes) * initial_probs \
                            + (torch.rand(num_nodes, num_nodes) - 0.5) * 2 * initial_noise
        # else, use the provided tensor as adj
        elif isinstance(initial_probs, Tensor):
            if initial_probs.shape != (num_nodes, num_nodes):
                raise ValueError(f'initial_theta must be of shape ({num_nodes}, {num_nodes})')
        else:
            raise ValueError(f'ParametrizedBernoulliSampler initial_theta must \
                              be a float or a tensor of shape ({num_nodes}, {num_nodes})')
            
        # Initialize scores based on the provided initial_theta and the scores_transform
        self.scores = Parameter(inverse_sigmoid(initial_probs), requires_grad=True)

    def compute_full_ei(self, num_nodes):
        
        dense_ed_idx_0 = torch.arange(num_nodes).unsqueeze(1).expand(-1, num_nodes)
        dense_ed_idx_0 = rearrange(dense_ed_idx_0, 'n_nodes_0 n_nodes_1 -> (n_nodes_0 n_nodes_1)')
        dense_ed_idx_1 = torch.arange(num_nodes).unsqueeze(0).expand(num_nodes, -1)
        dense_ed_idx_1 = rearrange(dense_ed_idx_1, 'n_nodes_0 n_nodes_1 -> (n_nodes_0 n_nodes_1)')
        dense_ed_idx = torch.stack([dense_ed_idx_0, dense_ed_idx_1])

        return dense_ed_idx

    def compute_edge_probs(self):
        probs = torch.sigmoid(self.scores)
        return probs

    def forward(self, x, n_adjs=1, **sampler_fwd_kwargs):
        """
        Args:
            x (Tensor): Input tensor.
            number_of_sampled_adjs (int): Number of sampled adjacency matrices.
        Returns:
            a list of `number_of_sampled_adjs` tuples containing:
                - tuple containing: (edge indices of shape `[2, num_edges]`, edge weights of shape `[num_edges]`)
                - log likelihood: shape `[num_edges]` for score function or `None` for straight-through 
        """
        probs = self.compute_edge_probs()
        dist = Bernoulli(probs=probs)
        samples = dist.sample([n_adjs])
        
        if self.gradient_estimator == 'st': # Straight-through estimator
            st_samples = samples + (probs - probs.detach())
            st_samples = rearrange(st_samples, 'n_adjs n_nodes_0 n_nodes_1 -> n_adjs (n_nodes_0 n_nodes_1)')
            if self.dense_full_ed_idx.device != st_samples.device:
                self.dense_full_ed_idx = self.dense_full_ed_idx.to(st_samples.device)
            return [((self.dense_full_ed_idx, st_samples[i]), None) for i in range(n_adjs)]
        elif self.gradient_estimator == 'sf': # Score-function estimator
            log_likelihood = dist.log_prob(samples).sum(-2)
            return [(dense_to_sparse(samples[i]), log_likelihood[i]) for i in range(n_adjs)]
        else:
            raise ValueError(f'Gradient estimator {self.gradient_estimator} not implemented for {self.__class__.__name__}')
        