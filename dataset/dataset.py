import os

import torch
from torch.utils.data import Dataset
from types import SimpleNamespace
from einops import rearrange

from torch_geometric.utils import add_self_loops
from torch_geometric.nn import models
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

from utils.utils import get_module_from_name

class StochasticGPVAR_Creator():
    """
    Create a dataset of n_data datapoints.
    Each edge has probability sample_probability of being sampled. 
    Each output Y_i is generated with the following model: X_i ~ N(0, input_noise^2)
    A_i ~ Bernoulli(sample_probabilities)
    Y_i = GNN_name(**GNN_kwargs)(X_i, A_i) + out_noise_type(output_noise)
    """
    def __init__(self, 
                 num_communities: int,
                 GNN_name: str,
                 GNN_kwargs: dict,
                 n_data: int,
                 n_input_features: int,
                 input_noise: float,
                 output_noise: float,
                 out_noise_type: str,
                 sample_probability: float,
                 seed=None,
                 verbose=True,
                 save_model_params=False):
        
        self.num_communities = num_communities
        self.ground_truth_GNN = get_module_from_name(models, GNN_name)
        self.ground_truth_GNN = self.ground_truth_GNN(**GNN_kwargs)
        self.n_data = n_data
        self.n_in_features = n_input_features
        self.input_noise = input_noise
        self.output_noise = output_noise
        self.out_noise_type = out_noise_type
        self.sample_probability = sample_probability
        self.seed = seed
        self.num_nodes = 6 * num_communities
        self.verbose = verbose
        self.save_model_params = save_model_params
        self.GNN_name = GNN_name
        self.GNN_kwargs = GNN_kwargs


    def create_dataset(self, train_size=0.7, val_size=0.15):
        # Check if data exist
        self.dataset_save_path = self.compute_save_path()
        if os.path.exists(self.dataset_save_path):
            if self.verbose:
                print('Dataset already exists')
                print('Load dataset from' + self.dataset_save_path)
        else:
            if self.verbose:
                print('Creating dataset...')
            if self.seed is not None:
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)  

            # create static adjacency matrix 
            node_idx, edge_index, _ = build_tri_community_graph(
                num_communities=self.num_communities)
            
            # add self loops
            base_edge_index, _ = add_self_loops(edge_index=torch.tensor(edge_index),
                                        num_nodes=self.num_nodes)

            self.base_edge_index = base_edge_index

            self.X = torch.randn(self.num_nodes, self.n_data, self.n_in_features) * self.input_noise
            
            self.clean_outputs, self.noisy_output = self.generate_data()

            # create folder
            os.makedirs(self.dataset_save_path)
            # save data
            if self.save_model_params:
                torch.save(self.ground_truth_GNN.state_dict(), self.dataset_save_path + '/model_params.pt')
            torch.save(self.base_edge_index, self.dataset_save_path + '/base_edge_index.pt')
            np.save(self.dataset_save_path + '/X.npy', self.X.detach().numpy())
            np.save(self.dataset_save_path + '/clean_outputs.npy', self.clean_outputs.detach().numpy())
            np.save(self.dataset_save_path + '/noisy_outputs.npy', self.noisy_output.detach().numpy())
            np.save(self.dataset_save_path + '/sampled_indices.npy', self.sampled_indices.detach().numpy())
            if self.verbose:
                print('\nDataset saved at ' + self.dataset_save_path + '\n')

        

        dataset = StochasticGPVAR_Dataset(self.dataset_save_path)

        train_size = int(len(dataset) * train_size)
        val_size = int(len(dataset) * val_size)
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        
        # visualize outputs
        y_with_noise = train_dataset.dataset.noisy_outputs.clone()
        plt.figure(figsize=(7.5, 6))
        plt.hist(y_with_noise.flatten(), bins=100)
        plt.title('Histogram of train y with noise')
        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        plt.savefig('./figures/_y_with_noise_histogram.png')
        plt.clf()
        plt.close()

        return train_dataset, val_dataset, test_dataset, self.dataset_save_path


    def generate_data(self): #, compute_optimal_values=False):
        compute_optimal_values = False
        num_samples_per_x = 128 if compute_optimal_values else 1

        # X shape must be [N nodes, batch_size, in_channels]
        self.sampled_indices = torch.bernoulli(torch.ones(self.n_data, num_samples_per_x, self.base_edge_index.shape[1]) * self.sample_probability)

        out_shape = self.ground_truth_GNN(self.X[:, 0, :], self.base_edge_index).shape
        y = torch.zeros([self.n_data, num_samples_per_x] + [sh for sh in out_shape])

        for i in tqdm(range(self.n_data), disable=not self.verbose):
            for s in range(num_samples_per_x):
                samples = self.sampled_indices[i, s] == 1
                y[i, s] = self.ground_truth_GNN(self.X[:, i, :], self.base_edge_index[:, samples])
        
        if self.out_noise_type == 'gaussian':
            noisy_y = y + torch.randn(y.shape) * self.output_noise
        elif self.out_noise_type == 'uniform':
            noisy_y = y + (torch.rand(y.shape) * 2 - 1) * self.output_noise

        if num_samples_per_x == 1:
            y = y[:, 0]
            noisy_y = noisy_y[:, 0]
        else:
           raise ValueError('num_samples_per_x must be 1 for now')

        return y, noisy_y
    
    def collate_fn(self, batch):
        """
        Collate function for the dataset
        """
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)

        x = rearrange(x, 'bs n_nodes t n_in_features -> bs t n_nodes n_in_features')
        y = rearrange(y, 'bs n_nodes t n_out_features -> bs t n_nodes n_out_features')

        return BatchDict(x, y)

    def compute_save_path(self):
        """
        Returns the name of the dataset file
        """
        n_comm = str(self.num_communities)
        n_data = str(self.n_data)
        n_in_features = str(self.n_in_features)
        in_noise = str(self.input_noise)
        out_noise = str(self.output_noise)
        out_noise_type = self.out_noise_type
        sample_prob = str(self.sample_probability)
        GNN_name = self.GNN_name
        GNN_kwargs = ''
        for k, v in self.GNN_kwargs.items():
            if isinstance(v, list):
                v = '_'.join([str(i) for i in v])
            elif isinstance(v, str):
                v = v.replace(' ', '_')
            elif isinstance(v, bool):
                v = str(int(v))
            else:
                v = str(v)
            GNN_kwargs += '_' + str(k) + '_' + str(v)

        package_filename = os.path.dirname(os.path.realpath(__file__))
        name = f'StochasticGPVAR_{GNN_name}_ncomm_{n_comm}_n_{n_data}_n_feat_{n_in_features}_in_noise_{in_noise}_out_noise_{out_noise}_{out_noise_type}_sample_prob_{sample_prob}' + GNN_kwargs

        save_path = os.path.join(package_filename, '.storage', name)
        return save_path
    

    def select_pyg_GNN(self, GNN_name):
        if hasattr(models, GNN_name):
            return getattr(models, GNN_name)
        else:
            raise ValueError(f'GNN {GNN_name} not found in torch_geometric.models. Available models are: {dir(models)}')

class BatchDict():
    """
    BatchDict is a dictionary that contains the batch data.
    It is used to store the batch data in a dictionary format.
    """
    def __init__(self, x, y):
        self.input = {'x': x}
        self.target = {'y': y}
        self.has_mask = False
        self.transform = self.transform_fn

    def transform_fn(self):
        raise NotImplementedError("Transform function of BatchDict called but not implemented")

# Dataset class
class StochasticGPVAR_Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.X = torch.from_numpy(np.load(path + '/X.npy'))
        self.clean_outputs = torch.from_numpy(np.load(path + '/clean_outputs.npy'))
        self.noisy_outputs = torch.from_numpy(np.load(path + '/noisy_outputs.npy'))
        self.sampled_indices = torch.from_numpy(np.load(path + '/sampled_indices.npy'))

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        return self.X[:, idx, :].unsqueeze(-1), self.noisy_outputs[idx].unsqueeze(-1)


# Code taken from torch-spatiotemporl library https://torch-spatiotemporal.readthedocs.io/en/latest/index.html
def _create_community():
    r"""

    .. code::

                  2
                 / \
                1 - 4
               / \ / \
        ... - 0 - 3 - 5 - ...
    """
    nodes = np.arange(6)
    # yapf: disable
    edges = np.asarray([[0, 1], [1, 2], [3, 4],  # slashes
                        [1, 3], [2, 4], [4, 5],  # backslashes
                        [0, 3], [1, 4], [3, 5]])  # horizontal
    # yapf: enable
    return nodes, edges


def build_tri_community_graph(num_communities):
    r"""
    A family of planar graphs composed of a number of communities.
    Each community takes the form of a 6-node triangle:

    .. code::

            2
           / \
          1 - 4
         / \ / \
        0 - 3 - 5

    All communities are arranged as a line

    .. code::

        c0 - c1 - c2 -  ....

    Args:
        num_communities (int): number of communities in the created graph.

    Returns:
        tuple: Returns a tuple containing the list of nodes, list of edges and
            list of edge weights (which is :obj:`None`).
    """
    nodes = []
    edges = []
    for c in range(num_communities):
        n, e = _create_community()
        n += c * 6
        e += c * 6
        nodes += list(n)
        edges += list(e)
        if c > 0:
            edges.append([n[0] - 1, n[0]])  # connect the two communities
    node_idx = np.stack(nodes, 0)
    edge_idx = np.stack(edges, 1)
    edge_idx = np.concatenate([edge_idx, edge_idx[::-1]], 1)
    edge_idx = np.unique(edge_idx, axis=1)
    return node_idx, edge_idx, None