from dml.miners import MinerFactory
from dml.chain.btt_connector import BittensorNetwork
from dml.configs.config import config
import multiprocessing as mp
def main(config):
    bt_config = config.get_bittensor_config()
    BittensorNetwork.initialize(bt_config)

    config.bittensor_network = BittensorNetwork

    miner = MinerFactory.get_miner(config)
    listener = miner.setup_logging()
    best_genome = miner.mine()
    if best_genome is not None:
        print(f"Best genome fitness: {best_genome.fitness.values[0]:.4f}")
        print(f"Baseline accuracy: {miner.baseline_accuracy:.4f}")
        print(f"Improvement over baseline: {best_genome.fitness.values[0] - miner.baseline_accuracy:.4f}")
    else:
        print("No best genome found.")

    print(f"Best genome fitness: {best_genome.fitness.values[0]:.4f}")
    print(f"Baseline accuracy: {miner.baseline_accuracy:.4f}")
    print(f"Improvement over baseline: {best_genome.fitness.values[0] - miner.baseline_accuracy:.4f}")
    return best_genome

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Set the start method before any multiprocessing code

    miner_type = "loss"  # Change this to "loss" or "simple" as needed
    best_genome = main(config)