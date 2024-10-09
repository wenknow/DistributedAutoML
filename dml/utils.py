import numpy as np
import torch
import random

def tensor_to_numpy(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

def numpy_to_tensor(array):
    return torch.from_numpy(array)

def ensure_numpy(data):
    if isinstance(data, torch.Tensor):
        return tensor_to_numpy(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def ensure_tensor(data):
    if isinstance(data, np.ndarray):
        return numpy_to_tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        return torch.tensor(data)
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)