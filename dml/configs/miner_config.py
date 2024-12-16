import time 

class MinerConfig:
    #TODO limit validator memory allowed to prevent DOS attacks 
    device = "cpu"
    batch_size = 8
    checkpoint_save_dir = "checkpoints"
    check_registration_interval = 500
    evaluation_iterations = 10
    gp_tree_height = 90
    generations = 1
    migration_interval = 100
    migrants_per_round = 10
    miner_type = "optimizer"
    num_processes = 1
    pool_url = None #"http://127.0.0.1:5000"
    population_size = 100 # Per process pop = population_size // num_processes
    push_platform = "hf"
    save_temp_only = True
    seed = int(time.time())
    tournament_size = 2    
    training_iterations = 10
    architectures = {
        "cifar10": [ "mlp" ],
        "imagenette": [ "resnet", "mobilenet_v3", "efficientnet_v2" ],
        "flowers102": ["resnet", "mobilenet_v3", "efficientnet_v2"]
    }
    architectures_weights = {
        "cifar100":0.333,
        "imagenet":0.333,
        "flowers102":0.333
    }

