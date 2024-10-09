import random
import torch
from deap import gp
import torch 
from configs.config import config
device = torch.device(config.device)

def safe_div(x, y):
    epsilon = 1e-8
    return x / (y + epsilon)

def safe_add(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x + y

def safe_sub(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x - y

def safe_mul(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x * y

def safe_div(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    epsilon = 1e-8
    return x / (y + epsilon)

def safe_sigmoid(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.sigmoid(x)

def safe_relu(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.relu(x)

def safe_tanh(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.tanh(x)


def safe_div(x, y):
    return x / (y + 1e-8)

def safe_log(x):
    return torch.log(torch.abs(x) + 1e-8)

def safe_sqrt(x):
    return torch.sqrt(torch.abs(x))

def safe_exp(x):
    return torch.exp(torch.clamp(x, -100, 100))

def create_pset():
    pset = gp.PrimitiveSet("MAIN", 2)
    
    # Basic arithmetic operations
    pset.addPrimitive(safe_add, 2)
    pset.addPrimitive(safe_sub, 2)
    pset.addPrimitive(safe_mul, 2)
    pset.addPrimitive(safe_div, 2)
    
    # Advanced mathematical operations
    pset.addPrimitive(safe_log, 1)
    pset.addPrimitive(safe_sqrt, 1)
    pset.addPrimitive(safe_exp, 1)
    
    # Trigonometric functions
    pset.addPrimitive(torch.sin, 1)
    pset.addPrimitive(torch.cos, 1)
    pset.addPrimitive(torch.tan, 1)
    
    # Activation functions
    pset.addPrimitive(torch.sigmoid, 1)
    pset.addPrimitive(torch.relu, 1)
    pset.addPrimitive(torch.tanh, 1)
    
    # Statistical functions
    pset.addPrimitive(torch.mean, 1)
    pset.addPrimitive(torch.std, 1)
    
    # Constants
    device = torch.device("cpu")
    pset.addEphemeralConstant("rand_const", lambda: torch.tensor(random.uniform(-1, 1), device=device))
    pset.addTerminal(torch.tensor(1.0, device=device), name="one")
    pset.addTerminal(torch.tensor(0.0, device=device), name="zero")
    pset.addTerminal(torch.tensor(0.5, device=device), name="half")
    
    # Rename arguments
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    
    return pset
