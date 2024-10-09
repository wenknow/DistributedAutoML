class ValidatorConfig:
    validation_interval = 21600  # Interval between validations in seconds
    validator_type = "loss"
    top_k = 10  # Number of top miners to distribute scores to
    min_score = 0.0  # Minimum score for miners not in the top-k
    top_k_weight = [0.5, 0.2] + [0.2/7] * 7 
    time_penalty_factor = 0.5
    time_penalty_max_time = 604800  #1week
    max_gene_size = 1024*20
    seed = 42

    