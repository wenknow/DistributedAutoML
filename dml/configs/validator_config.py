import numpy as np

def constrained_decay(n: int, ratio: float = 5.0):
    smallest = 2 / (n * (1 + ratio))
    largest = smallest * ratio
    decay = np.linspace(largest, smallest, n)
    normalized_decay = decay / np.sum(decay)
   
    return normalized_decay.tolist() 

class ValidatorConfig:
    device = "cpu"

    validation_interval = 20
    validator_type = "loss"
    top_k = 50
    min_score = 0.0
    top_k_weight = constrained_decay(50, 5.0)
    
    # Dataset-specific settings
    dataset_weights = {
        "mnist": 1.0,
        "cifar10": 1.5,
        "cifar100": 2.0
    }
    
    # Training settings
    training_iterations = 300
    # {
    #     "mnist": 100,
    #     "cifar10": 200,
    #     "cifar100": 300
    # }
    
    validation_iterations = 30
    # {
    #     "mnist": 10,
    #     "cifar10": 20,
    #     "cifar100": 30
    # }
    
    # Resource limits
    max_gene_size = 1024*20
    time_penalty_factor = 0.5
    time_penalty_max_time = 7200
    architectures = {
        "cifar10": [ "mlp" ],
        "imagenette": [ "resnet", "mobilenet_v3", "efficientnet_v2" ],
        "fgvc_aircraft": ["resnet", "mobilenet_v3", "efficientnet_v2"]
    }
    architectures_weights = {
        "cifar100":0.333,
        "imagenet":0.333,
        "fgvc_aircraft":0.333
    }

    seed = 42