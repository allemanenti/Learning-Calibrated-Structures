from torch.utils.data import DataLoader
from dataset.dataset import StochasticGPVAR_Creator

from utils.utils import module_from_kwargs

def get_dataloaders(cfg):
    """
    Function to get the dataloaders for the given dataset configuration.
    
    Args:
        cfg: Configuration object from experiment 
    Returns:
        train_loader: DataLoader for the training set
        val_loader: DataLoader for the validation set
        test_loader: DataLoader for the test set
    """   
    if cfg.dataset.name == 'SGPVAR':
        sgpvar = module_from_kwargs(StochasticGPVAR_Creator, cfg.dataset.params)
        train_dataset, val_dataset, test_dataset, dataset_save_path = module_from_kwargs(sgpvar.create_dataset, cfg.optimization.hparams)
        train_loader = DataLoader(train_dataset, batch_size=cfg.optimization.hparams.batch_size, shuffle=True, collate_fn=sgpvar.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=cfg.optimization.hparams.batch_size, shuffle=False, collate_fn=sgpvar.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg.optimization.hparams.batch_size, shuffle=False, collate_fn=sgpvar.collate_fn)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} is not implemented")
        
    print('Data loaded')
    
    return train_loader, val_loader, test_loader, dataset_save_path