# Official repository for the paper "Learning Latent Graph Structures and Their Uncertainty" (ICML 2025)

Official repository for the paper "Learning Latent Graph Structures and Their Uncertainty" (ICML 2025).

This repository contains a refactoring of the code used in the paper "Learning Latent Graph Structures and Their Uncertainty" (ICML 2025). The code is designed to be modular and easy to use, allowing for quick experimentation with different GNNs and Datasets.

## Requirements
The code has been tested with the following versions of Python and libraries:   

- `Python 3.11.11`
- `torch 2.7.0`
- `torch-geometric 2.6.1`

A fast installation of the required packages can be done using the following commands:

```bash
conda create LLGS python=3.11
conda activate LLGS

pip install torch
pip install torch-geometric
pip install einops
pip install hydra-core --upgrade
pip install hydra-joblib-launcher --upgrade
pip install matplotlib
```

# How to run the code 
Inside the conda environment, run the following command to start the training 

```bash
python experiments/run_experiment.py
```

This will start the training of the model with the parameters defined in `config/base.yaml`. You can modify the parameters in this file, in the `config/` subfolders files or in the command line. For example, to change the number of epochs to 5, you can run the following command:

```bash
python experiments/run_experiment.py optimization.hparams.epochs=5
```

We recommend using the ENG_DIST loss instead of the MMD as no additional hyperparameters are needed. 

The code uses the `hydra` library for configuration management. If you are not familiar with `hydra`, you can check the [documentation](https://hydra.cc/docs/intro/) for more information.

## Running different GNNs
To run different GNNs, you can modify `model.gnn.name` parameter in `model_base.yaml` with any of the [torch-geometric Models](https://pytorch-geometric.readthedocs.io/en/2.6.1/modules/nn.html#models).
 You should also modify the `model.gnn.kwargs` parameters accordingly. Since the dataset is build using another GNN, you should also modify the `dataset.params` in `dataset_base.yaml` accordingly or disable the sanity check performed by the `sanity_check_gnn_kwargs(cfg)` function.

## Adding new datasets
To use the code with new datasets, you need to

1. Comment lines 30-31 in `experiments/run_experiment.py` to avoid loading the true graph as comparison and uncomment line 32.
2. Modify the `get_dataloaders` function in `utils/data_utils.py` to load your dataset. The function should return the train, val and test dataloaders. 
A batch element in the loader should have the following structure:
```python
for batch in loader:
    break

x = batch.input['x']  # has shape [batch_size, input_features, num_nodes, 1]
y = batch.target['y'] # has shape [batch_size, output_features, num_nodes, 1]
```

