import torch
import inspect
import omegaconf

def nice_dict(d, tab=""):
    """
    
    Convert a dictionary to a nicely formatted string. 
    """
    d = alphabetically_sorted_dict(d)
    tab_ = tab
    s = ""
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
            s += f"\n{tab_}{k}:"
            s += nice_dict(v, tab=tab_+ "  ")
        else:
            s += f"\n{tab_}{k}: {v}"
    return s

def alphabetically_sorted_dict(d):
    """
    Sorts a dictionary by its keys in alphabetical order (and recursively all dicts inside).
    """
    return {k: alphabetically_sorted_dict(v) if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig) else v \
             for k, v in sorted(d.items())}


def inverse_sigmoid(x):
    """
    Inverse of the sigmoid function.
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Inverse sigmoid of the input tensor.
    """
    return torch.log(x / (1 - x))


def module_from_kwargs(cls, kwargs, return_module=True):
    """
    Filters the kwargs to only include those that are valid for the given class and 
    returns an instance of the class with those kwargs if return_module is True.
    
    Args:
        cls: The class to filter the kwargs for.
        kwargs: The kwargs to filter.
        
    Returns:
        The cls/function with the filtered kwargs.
    """
    valid_kwargs = inspect.signature(cls.__init__).parameters
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs and k != 'self'}
    if return_module:
        return cls(**valid_kwargs)
    else:
        return valid_kwargs


def get_module_from_name(module, name):
    """
    Get a class or function
    from a module by its name.
    Args:
        module: The module to get the class/function from.
        name: The name of the class/function to get.
    Returns:
        The class/function with the given name from the module.
    """
    if not isinstance(module, list):
        module = [module]

    for m in module:
        if hasattr(m, name):
            return getattr(m, name)
    
    raise ValueError(f"Module {name} not found in passed modules {module}")