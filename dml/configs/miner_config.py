class MinerConfig:
    #TODO limit validator memory allowed to prevent DOS attacks 
    population_size = 50
    num_meta_levels = 1
    genome_length = 5
    tournament_size = 2
    generations = 10000
    generation_iters = 1000
    num_scalars = 1
    num_vectors = 5
    num_tensors = 1
    scalar_size = 1
    vector_size = (128,)
    tensor_size = (128, 128)
    input_addresses = [1, 2]
    output_addresses = [1]
    miner_type = "loss"
    migration_server_url = "http://127.0.0.1:5000"
    migration_interval = 10
    pool_url = None#"http://127.0.0.1:5000"
    push_platform = "hf"
    mutation_log_interval=99
    check_registration_interval = 500
    gp_tree_height = 15
    seed = 42