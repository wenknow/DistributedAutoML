from dml.miners import MinerFactory
from dml.chain.btt_connector import BittensorNetwork
from dml.configs.config import config
def main(config):
    bt_config = config.get_bittensor_config()
    BittensorNetwork.initialize(bt_config)

    config.bittensor_network = BittensorNetwork

    miner = MinerFactory.get_miner(config)
    best_genome = miner.mine()

    print(f"Best genome fitness: {best_genome.fitness.values[0]:.4f}")
    print(f"Baseline accuracy: {miner.baseline_accuracy:.4f}")
    print(f"Improvement over baseline: {best_genome.fitness.values[0] - miner.baseline_accuracy:.4f}")
    return best_genome

if __name__ == "__main__":
    miner_type = "loss"  # Change this to "loss" or "simple" as needed
    best_genome = main(config)