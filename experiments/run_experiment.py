import os
import torch

from torch_geometric.nn import models
from torch_geometric.utils import to_dense_adj

from einops import rearrange
import numpy as np

from tqdm import tqdm
import hydra
from omegaconf import open_dict
                
from loss import loss as implemented_losses
from modules import sampler as implemented_samplers
from modules.predictor import PredictionModel

from utils.utils import nice_dict, module_from_kwargs, get_module_from_name
from utils.data_utils import get_dataloaders

                
def run_experiment(cfg):
    # Set seed, device and update and print config
    device, cfg = set_experiment(cfg)
    sanity_check_gnn_kwargs(cfg)

    # Create dataset
    # modify get_dataloaders function to use your dataset, if you use another dataset set
    train_loader, val_loader, test_loader, dataset_save_path = get_dataloaders(cfg) 
    true_thetas = torch.load(dataset_save_path + '/base_edge_index.pt').to(device)
    true_thetas = (to_dense_adj(true_thetas) * cfg.dataset.params.sample_probability).squeeze(0)
    # true_thetas = None

    # Create model
    predictor = create_predictor(cfg, device, dataset_save_path)

    # Create optimizer
    optimizer = get_module_from_name(torch.optim, cfg.optimization.optimizer.name)
    valid_kwargs = module_from_kwargs(optimizer, cfg.optimization.optimizer.kwargs, return_module=False)
    valid_kwargs['params'] = predictor.parameters()
    optimizer = optimizer(**valid_kwargs)
    
    epochs = cfg.optimization.hparams.epochs
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        for i, batch in enumerate(train_loader):
            in_epoch_perc = (i+1) / len(train_loader)
            epoch_step = epoch + (in_epoch_perc)
            
            # Forward
            loss, metrics = forward_pass(predictor, batch, device, cfg)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch_step:7.2f} | Train Loss: {loss.item():.4f} | Train MSE PP: {metrics['point_prediction_se'].mean().item():.4f}")

        val_loss, val_metrics, mae_on_theta = validate(predictor, val_loader, device, cfg, true_thetas)
        print(f"\nValidation Loss: {val_loss:.4f} \
              \nValidation MSE PP: {val_metrics[0]:.2f} +- {val_metrics[1]:.2f} \
              \nMAE on theta: {mae_on_theta:.3f}")

    test_loss, test_metrics, mae_on_theta = validate(predictor, test_loader, device, cfg, true_thetas)
    print(f"\nTest Loss: {test_loss:.4f} \
          \nTest MSE PP: {test_metrics[0]:.2f} +- {test_metrics[1]:.2f} \
          \nMAE on theta: {mae_on_theta:.3f}")
    

def compute_loss(y_pred, y, log_likelihoods, cfg):
    bs, n_sampled_adjs, n_nodes, out_feat = y_pred.shape
    loss = get_module_from_name(implemented_losses, cfg.optimization.loss.name)
    loss = loss(cfg)
    loss, metrics = loss(y_pred, y, log_likelihoods) 
    return loss, metrics


def create_predictor(cfg, device, dataset_save_path):     
    # sampler
    Sampler = get_module_from_name(implemented_samplers, cfg.model.sampler.name)
    sampler = module_from_kwargs(Sampler, cfg.model.sampler.kwargs)
    if cfg.model.sampler.trainable == False: 
        for param in sampler.parameters():
            param.requires_grad = False
    # gnn
    Gnn = get_module_from_name(models, cfg.model.gnn.name)
    gnn = Gnn(**cfg.model.gnn.kwargs)
    if cfg.model.gnn.trainable == False: # load GNN model used to generate data
        gnn.load_state_dict(torch.load(dataset_save_path + '/model_params.pt'))
        for param in gnn.parameters():
            param.requires_grad = False
    # predictor model
    predictor = PredictionModel(Sampler=sampler,
                                GNN=gnn, 
                                gradient_estimator=cfg.model.sampler.kwargs['gradient_estimator'],
                                output_noise=cfg.model.gnn['output_noise'],
                                out_noise_type=cfg.model.gnn['out_noise_type'],
                                )
    predictor.to(device)
    return predictor


def forward_pass(predictor, batch, device, cfg):
    x, y = prepare_batch(batch, device)

    # Forward 
    y_pred, log_likelihoods = predictor(x, n_adjs=cfg.model.sampler.kwargs['n_adjs'])
    return compute_loss(y_pred, y, log_likelihoods, cfg)


def prepare_batch(batch, device):
    x = rearrange(batch.input['x'], 'bs in_feat n_nodes 1 -> bs n_nodes in_feat').to(device)
    y = rearrange(batch.target['y'], 'bs out_feat n_nodes 1 -> bs n_nodes out_feat').to(device)
    return x, y


def sanity_check_gnn_kwargs(cfg):
    # Compare kwargs in dataset.params.GNN_kwargs with cfg.model.gnn.kwargs
    if cfg.dataset.params.GNN_kwargs is not None:
        for k, v in cfg.dataset.params.GNN_kwargs.items():
            if k not in cfg.model.gnn.kwargs:
                pass
            elif cfg.model.gnn.kwargs[k] != v:
                raise ValueError(f"Value {cfg.model.gnn.kwargs[k]} for key {k} in cfg.model.gnn.kwargs does not match {v} in dataset.params.GNN_kwargs")


def set_experiment(cfg):
    print('Experiment started')

    # Set device
    if cfg.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
        torch.cuda.set_device(cfg.gpu) 
    
    device  = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 100000, (1,)).item()
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)     

    with open_dict(cfg):
        cfg.model.sampler.kwargs['num_nodes'] = cfg.dataset.params.num_communities * 6

    str_cfg = nice_dict(cfg)
    print('Config file:\n', str_cfg)

    return device, cfg


def validate(predictor, val_loader, device, cfg, true_thetas):
    mean_val_loss = []
    val_se = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            v_loss, v_metrics = forward_pass(predictor, batch, device, cfg)
            mean_val_loss.append(v_loss.item())
            val_se.append(v_metrics['point_prediction_se'])

    mean_val_loss = np.mean(mean_val_loss)

    val_serrors = torch.cat(val_se)
    val_mean_se = val_serrors.mean()
    val_sdv_se = val_serrors.std() 

    if true_thetas is not None:
        mae_on_theta = torch.abs(true_thetas - predictor.sampler.compute_edge_probs()).mean()
    else:
        mae_on_theta = torch.nan

    return mean_val_loss, (val_mean_se, val_sdv_se), mae_on_theta



@hydra.main(config_path="../config", config_name="base")
def hydra_wrap(cfg):
    str_cfg = nice_dict(cfg)
    print('Config file:\n', str_cfg)

    try:
        run_experiment(cfg)
    except Exception as e:
        print(e)
        print('RUN FAILED.\n cfg:\n', cfg)

if __name__ == '__main__':
    torch.set_num_threads(16)
    hydra_wrap()