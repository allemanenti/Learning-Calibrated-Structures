import torch

from einops import rearrange
# -------------------------------------------------
# BASE LOSS CLASSES 
# -------------------------------------------------
# y_true: [batch_size, n_nodes, output_dim]
# y_pred: [batch_size, n_adj, n_nodes, output_dim]
# log_likelihoods: [n_adj, num_nodes]
# -------------------------------------------------

# expected values loss
class ExpectedValuesLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(ExpectedValuesLoss, self).__init__()
        self.cfg = cfg
        self.gradient_estimator = cfg.model.sampler.kwargs['gradient_estimator']
        self.variance_reduction = cfg.optimization.score_function['variance_reduction']
    
    def forward(self, y_pred, y_true, log_likelihoods):
        pp_ae, pp_se = pp_ae_and_se(y_pred, y_true) 
        metrics = {'point_prediction_ae': pp_ae, 'point_prediction_se': pp_se}

        _, n_adjs, _, _ = y_pred.shape
        y_true = y_true.unsqueeze(1)  # [batch_size, 1, n_nodes, output_dim]

        prediction_loss = self.prediction_loss_fn(y_pred, y_true) # [batch_size, n_adj]
        
        if self.gradient_estimator == 'sf':
            log_likelihoods = log_likelihoods.sum(-1) # [n_adj]
            if self.variance_reduction:
                unbiased_beta = self.compute_baseline(prediction_loss, n_adjs) # [batch_size, n_adjs]
            else:
                unbiased_beta = 0.
            graph_loss = (log_likelihoods.unsqueeze(0) * (prediction_loss.detach() - unbiased_beta)).mean()
            total_loss = graph_loss + prediction_loss.mean()

            metrics['graph_loss'] = graph_loss.item()
            metrics['prediction_loss'] = prediction_loss.mean().item()

            return total_loss, metrics

        elif self.gradient_estimator == 'st':
            return prediction_loss.mean(), metrics

    def compute_baseline(self, prediction_loss, n_adjs):
        biased_sum = prediction_loss.detach().sum(dim=-1, keepdim=True) # [batch_size, 1]
        unbiased_beta = (biased_sum - prediction_loss.detach()) / (n_adjs - 1) # [batch_size, n_adjs]
        return unbiased_beta
    
    def prediction_loss_fn(self, y_pred, y_true):
        raise NotImplementedError("Implement prediction_loss_fn in subclasses")


# discrepancy loss
class DiscrepancyLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(DiscrepancyLoss, self).__init__()
        self.cfg = cfg
        self.gradient_estimator = cfg.model.sampler.kwargs['gradient_estimator']
        self.variance_reduction = cfg.optimization.score_function['variance_reduction']
    
    def forward(self, y_pred, y_true, log_likelihoods):
        bs, n_adjs, n_nodes, out_feat = y_pred.shape

        pp_ae, pp_se = pp_ae_and_se(y_pred, y_true)
        metrics = {'point_prediction_ae': pp_ae, 'point_prediction_se': pp_se}

        y_pred = rearrange(y_pred, 'bs n_adjs n_nodes out_feat -> bs n_adjs (n_nodes out_feat)', 
                           bs=bs, n_adjs=n_adjs, n_nodes=n_nodes, out_feat=out_feat)
        y_true = rearrange(y_true, 'bs n_nodes out_feat -> bs 1 (n_nodes out_feat)',
                            bs=bs, n_nodes=n_nodes, out_feat=out_feat)
        log_likelihoods = log_likelihoods.sum(-1) # [n_adjs]


        pp_kernel_matrix_ = self.kernel_vec(y_pred, y_pred, self.kernel_params)
        pp_kernel_matrix = pp_kernel_matrix_ * (1 - torch.eye(n_adjs, device=pp_kernel_matrix_.device))

        pt_kernel_matrix = self.kernel_vec(y_pred, y_true, self.kernel_params)[..., 0]

        if self.gradient_estimator == 'st':
            pp_prediction_loss = pp_kernel_matrix.sum() / (bs * (n_adjs**2 - n_adjs))
            pt_prediction_loss = pt_kernel_matrix.sum() / (bs * n_adjs)
            total_loss = pp_prediction_loss - 2*pt_prediction_loss
            return total_loss, metrics

        elif self.gradient_estimator == 'sf':
            log_likelihood_matrix_ = log_likelihoods.unsqueeze(-1) + log_likelihoods.unsqueeze(-2)
            log_likelihood_matrix = log_likelihood_matrix_ * (1 - torch.eye(n_adjs, device=log_likelihood_matrix_.device))

            # Compute loss components for graph learning...
            if self.variance_reduction:
                beta_pp, beta_pt = self.compute_baseline(pp_kernel_matrix, pt_kernel_matrix, n_adjs, bs)
            else:
                beta_pp, beta_pt = 0., 0.
        
            pp_MMD_graph = (log_likelihood_matrix * (pp_kernel_matrix.detach() - beta_pp)).sum() / (bs * (n_adjs**2 - n_adjs))
            pt_MMD_graph = (log_likelihoods * (pt_kernel_matrix.detach() - beta_pt)).sum() / (bs * n_adjs)
            
            # ...and for predictor's parameters learning
            pp_prediction_loss = pp_kernel_matrix.sum() / (bs * (n_adjs**2 - n_adjs))
            pt_prediction_loss = pt_kernel_matrix.sum() / (bs * n_adjs)

            prediction_loss = pp_prediction_loss - 2*pt_prediction_loss 
            graph_loss = pp_MMD_graph - 2*pt_MMD_graph
            total_loss = graph_loss + prediction_loss
            return total_loss, metrics
        
    def compute_baseline(self, pp_kernel_matrix, pt_kernel_matrix, n_adjs, bs):
        # beta for pp
        pp_sum = pp_kernel_matrix.detach().sum()
        pp_1_sum = pp_kernel_matrix.detach().sum(dim=(0,1), keepdim=True)
        pp_2_sum = pp_kernel_matrix.detach().sum(dim=(0,2), keepdim=True)
        pp_batch_sum = pp_kernel_matrix.detach().sum(dim=0, keepdim=True)

        pp_beta_sum_matrix = pp_sum - pp_1_sum - pp_2_sum + pp_batch_sum
        beta_pp = pp_beta_sum_matrix / (bs * (n_adjs**2 - 3*n_adjs + 3)) # [1, n_adjs, n_adjs]

        # beta for pt
        pt_biased_sum = pt_kernel_matrix.detach().sum(dim=-1, keepdim=True)
        beta_pt = (pt_biased_sum - pt_kernel_matrix.detach()) / (n_adjs - 1) # [bs, n_adjs]
        return beta_pp, beta_pt 
    

# point prediction loss
class PointPredictionLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(PointPredictionLoss, self).__init__()
        self.cfg = cfg
        self.gradient_estimator = cfg.model.sampler.kwargs['gradient_estimator']
        self.variance_reduction = cfg.optimization.score_function['variance_reduction']
    
    def forward(self, y_pred, y_true, log_likelihoods):
        bs, n_adjs, n_nodes, out_feat = y_pred.shape
        log_likelihoods = log_likelihoods.sum(-1) # [n_adj]
        pp_ae, pp_se = pp_ae_and_se(y_pred, y_true)
        metrics = {'point_prediction_ae': pp_ae, 'point_prediction_se': pp_se}

        if self.gradient_estimator == 'st':
            total_loss = pp_se.mean()
        elif self.gradient_estimator == 'sf':
            y_pred = rearrange(y_pred, 'bs n_adjs n_nodes out_feat -> bs n_adjs (n_nodes out_feat)',
                                bs=bs, n_adjs=n_adjs, n_nodes=n_nodes, out_feat=out_feat)
            y_true = rearrange(y_true, 'bs n_nodes out_feat -> bs (n_nodes out_feat)',
                                bs=bs, n_nodes=n_nodes, out_feat=out_feat)
            E_y_pred = y_pred.detach().mean(dim=1) # [bs, n_nodes*out_feat]
            diff_ = 2 * (E_y_pred - y_true)

            if self.variance_reduction:
                beta = self.compute_baseline(y_pred, n_adjs) 
            else:
                beta = 0.
            sf_loss = ((y_pred.detach() - beta) * log_likelihoods.unsqueeze(0).unsqueeze(2)).mean(dim=1) # [bs, n_nodes*out_feat]
            total_loss = (sf_loss * diff_).mean() + pp_se.clone().mean()
        
        return total_loss, metrics

    def compute_baseline(self, y_pred, n_adjs):
        biased_sum = y_pred.detach().sum(dim=1, keepdim=True)
        unbiased_beta = (biased_sum - y_pred.detach()) / (n_adjs - 1) 
        return unbiased_beta # [batch_size, n_adjs]

# -------------------------------------------------
# utility functions
# -------------------------------------------------

def pp_ae(y_pred, y_true):
    return torch.abs(torch.quantile(y_pred, 0.5, dim=1) - y_true)

def pp_se(y_pred, y_true):
    return ((torch.mean(y_pred, dim=1) - y_true)**2).mean(dim=(-1, -2))

def pp_ae_and_se(y_pred, y_true):
    return pp_ae(y_pred, y_true), pp_se(y_pred, y_true)

def rational_quadratic_kernel(x, y, kernel_params):
    """
    Rational quadratic kernel
    **kwargs must have alpha and sigma parameters stored
    """
    n_ = torch.norm(x-y, dim=-1)**2
    c_ = 2*kernel_params['alpha']*(kernel_params['sigma']**2) 
    return (1 + n_ / ( c_ )) ** (-kernel_params['alpha'])

def energy_distance_kernel_vec(x, y, kernel_params):
    """
    Energy distance kernel
    **kwargs are not used
    """
    d_ = x.unsqueeze(-2) - y.unsqueeze(-3) # difference
    n_ = (d_**2).sum(axis=-1) # norm
    k_ = torch.sqrt(n_ + 1e-6)
    return -k_

# -------------------------------------------------
# specific losses
# -------------------------------------------------

# expected values losses
class EV_MAE(ExpectedValuesLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

    def prediction_loss_fn(self, y_pred, y_true):
        return torch.abs(y_pred - y_true).mean(dim=(-1, -2)) # average over nodes and output_dim


class EV_MSE(ExpectedValuesLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def prediction_loss_fn(self, y_pred, y_true):
        return torch.square(y_pred - y_true).mean(dim=(-1, -2)) # average over nodes and output_dim
    


# discrepancy losses
class MMD(DiscrepancyLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kernel_vec = rational_quadratic_kernel
        self.kernel_params = {
            'alpha': cfg.optimization.loss.kwargs['alpha'],
            'sigma': cfg.optimization.loss.kwargs['sigma']
        }

class ENG_DIST(DiscrepancyLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kernel_vec = energy_distance_kernel_vec
        self.kernel_params = None  

# point prediction loss
class PP_MSE(PointPredictionLoss):
    def __init__(self, cfg):
        super().__init__(cfg)