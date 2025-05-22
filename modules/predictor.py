from torch.nn import Module, Parameter
from torch import rand_like, randn_like, randn, stack

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data as PyGData

from einops import rearrange

class PredictionModel(Module):
    """
    Args:
        Sampler (Module): The adjacency matrix sampler.
        GNN (Module): The GNN model.
        gradient_estimator (str): The gradient estimator to use. ('sf' or 'st')
        output_noise (float): The amount of noise to add to the output.
        out_noise_type (str): The type of noise to add to the output. ('gaussian' or 'uniform')
    """
    def __init__(self, 
                 Sampler,
                 GNN, 
                 gradient_estimator: str,
                 output_noise: float = 0.0,
                 out_noise_type: str = 'gaussian',
                 ):
        
        super(PredictionModel, self).__init__()
        self.gradient_estimator = gradient_estimator

        if output_noise > 0.0:
            self.add_noise = True
            self.noise_param = Parameter(randn(1) * output_noise, requires_grad=True)
            self.output_noise_type = out_noise_type
            assert self.output_noise_type in ['gaussian', 'uniform'], 'Only gaussian and uniform noise are supported for now'
        else:
            self.add_noise = False

        # Parametrize the adjacency matrix
        self.sampler = Sampler
        # Processing function
        self.GNN = GNN  
        
        
    def forward(self, x, n_adjs=1, **sampler_fwd_kwargs):
        """
        x: torch.Tensor -- The input tensor of shape (batch_size, n_nodes, input_dim).
        n_adjs: int -- number of sampled adjs to use.
        """ 
        # Sample the adjs  
        sampler_list = self.sampler(x, n_adjs, **sampler_fwd_kwargs) # sampler list: [((ei, ew), log_likelihood) for i in sampled_adjs]
        bs = x.shape[0]
        n_nodes = x.shape[1]

        # Prepare log likelihoods
        log_likelihoods = []
        for _, ll in sampler_list:
            log_likelihoods.append(ll)
        if ll is not None:
            log_likelihoods = stack(log_likelihoods)
        else:
            # case of st gradient estimator
            log_likelihoods = None

        # Prepare batched data
        data_list = []
        for i in range(bs):
            for adj_coo, _ in sampler_list:
                data_list.append(
                    PyGData(x=x[i,:,:], edge_index=adj_coo[0], edge_weight=adj_coo[1])) 
        disjoint_batch = PyGBatch.from_data_list(data_list)

        # Make predictions
        y_pred = self.GNN(disjoint_batch.x, disjoint_batch.edge_index, disjoint_batch.edge_weight)


        y_pred = rearrange(y_pred, '(bs n_adjs n_nodes) output_dim -> bs n_adjs n_nodes output_dim',
                                bs=bs, n_adjs=len(sampler_list), n_nodes=n_nodes)

        # Add noise
        if self.add_noise:
            if self.output_noise_type == 'gaussian':
                noise = randn_like(y_pred) * self.noise_param
            elif self.output_noise_type == 'uniform':
                noise = (rand_like(y_pred) - 0.5) * self.noise_param * 2
            y_pred = y_pred + noise

        return y_pred, log_likelihoods