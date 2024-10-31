from abc import ABC, abstractmethod
from copy import deepcopy
from huggingface_hub import HfApi, Repository
import os
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np
import random
import math
import pickle
import dill
import operator
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import time

from multiprocessing import Queue, Process, Pool, Value
import queue

from deap import algorithms, base, creator, tools, gp
from functools import partial

from dml.models import BaselineNN, EvolvableNN
from dml.ops import create_pset
from dml.gene_io import save_individual_to_json, load_individual_from_json, safe_eval
from dml.gp_fix import SafePrimitiveTree
from dml.destinations import PushMixin, PoolPushDestination, HuggingFacePushDestination
from dml.utils import set_seed, calculate_tree_depth
from logging.handlers import QueueHandler, QueueListener



LOCAL_STORAGE_PATH = "./checkpoints_1"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

def create_n_evaluate(self, individual, train_loader, val_loader):

    model = create_model(individual)
    try:
        self.train(model, train_loader=train_loader)
        fitness = self.evaluate(model, val_loader=val_loader)
    except:
        return (0.0,)

# Define functions used in evaluation
def create_model(individual, config):
    evolved_function = gp.compile(expr=individual, pset=config.pset)
    model = BaselineNN(input_size=28*28, hidden_size=128, output_size=10).to(config.device)
    return model, evolved_function


def run_island(island_id, seed, config, migration_in_queue, migration_out_queue, stats_queue, global_best_queue, 
               island_state_dict, checkpoint_lock,shutdown_event):
    try:
        # Initialize random seeds
        # setup_process_logging(island_id)
        random.seed(seed + island_id)
        torch.manual_seed(seed + island_id)
        np.random.seed(seed + island_id)

        # Initialize DEAP components within the process
        pset = create_pset()
        toolbox = base.Toolbox()

        # Define creators (ensure this is done only once per process)
        try:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        except AttributeError:
            pass  # FitnessMax already created

        try:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        except AttributeError:
            pass  # Individual already created

        # Register functions in toolbox
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", lambda ind: create_n_evaluate(ind, train_loader, val_loader, config))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=config.Miner.gp_tree_height))

        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=config.Miner.gp_tree_height))

        # Load data
        train_loader, val_loader = load_data(config)

        # Restore island state if provided
        with checkpoint_lock:
            island_state = island_state_dict.get(island_id, None)

        if island_state:
            population = island_state['population']
            local_best = island_state['local_best']
            local_best_fitness = island_state['local_best_fitness']
            generation = island_state['generation']
            random.setstate(island_state['random_state'])
            torch.set_rng_state(island_state['torch_rng_state'])
            np.random.set_state(island_state['numpy_rng_state'])
            logging.info(f"Island {island_id} resumed from generation {generation}")
        else:
            # Initialize fresh state
            population = toolbox.population(n=config.Miner.population_per_island)
            local_best = None
            local_best_fitness = -float('inf')
            generation = 0
            logging.info(f"Island {island_id} starting fresh")

            # Island-specific variables
        local_best = None
        local_best_fitness = -float('inf')
        generation = 0

        while generation < config.Miner.generations:
            # Evaluate individuals
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(
                lambda ind: create_n_evaluate(ind, train_loader, val_loader, config),
                invalid_ind
            )
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if fit[0] > local_best_fitness:
                    local_best = deepcopy(ind)
                    local_best_fitness = fit[0]

            # Selection
            population = toolbox.select(population, len(population))
            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.Miner.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < config.Miner.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Replace population
            population[:] = offspring

            # Save island state at the end of each generation
            island_state = {
                'population': population,
                'local_best': local_best,
                'local_best_fitness': local_best_fitness,
                'generation': generation + 1,
                'random_state': random.getstate(),
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
            }

            with checkpoint_lock:
                island_state_dict[island_id] = island_state
                # Optionally call save_checkpoint here

            generation += 1
    except Exception as e:
        # Log exception with traceback
        logging.exception(f"Exception occurred in island {island_id}: {e}")
        # Handle migrations, statistics, etc.
        # ...
    finally:
        logging.info(f"Island {island_id} finished.")

def train(model, loss_function, train_loader, config):
    set_seed_out(config.seed)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = loss_function
    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        if idx >= config.Miner.training_iterations:
            break
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, config):
    set_seed_out(config.seed)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            if idx >= config.Miner.evaluation_iterations:
                break
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

def setup_process_logging(island_id):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'island_{island_id}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Remove any existing handlers
    logger.handlers = []
    logger.addHandler(handler)

def create_n_evaluate(individual, train_loader, val_loader, config):
    try:
        model, loss_function = create_model(individual, config)
        train(model, loss_function, train_loader, config)
        fitness = evaluate(model, val_loader, config)
    except Exception as e:
        logging.error(f"Error evaluating individual: {e}")
        fitness = 0.0
    return fitness,

def load_data(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(config.Miner.seed))
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, generator=torch.Generator().manual_seed(config.Miner.seed))
    return train_loader, val_loader

def set_seed_out(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BaseMiner(ABC, PushMixin):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.seed = self.config.Miner.seed
        set_seed(self.seed)
        self.setup_logging()
        self.metrics_file = config.metrics_file
        self.metrics_data = []
                # Initialize variables
        self.population = None
        self.hof = None
        self.best_individual_all_time = None
        self.start_generation = 0

        self.push_destinations = []

        # DEAP utils
        self.initialize_deap()

            # Load checkpoint if exists
        checkpoint_file = os.path.join(LOCAL_STORAGE_PATH, 'evolution_checkpoint.pkl')
        if os.path.exists(checkpoint_file):
            logging.info("Loading checkpoint...")
            self.start_generation = self.load_checkpoint(checkpoint_file)
            logging.info(f"Resuming from generation {self.start_generation}")
        else:
            self.population = self.toolbox.population(n=self.config.Miner.population_size)
            self.hof = tools.HallOfFame(1)
            self.best_individual_all_time = None
            self.start_generation = 0
            logging.info("Starting evolution from scratch")

    def initialize_deap(self):
        self.toolbox = base.Toolbox()
        self.pset = create_pset()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.create_n_evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(
                operator.attrgetter("height"), self.config.Miner.gp_tree_height
            ),
        )
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(
                operator.attrgetter("height"), self.config.Miner.gp_tree_height
            ),
        )

    def log_metrics(self, metrics):
        self.metrics_data.append(metrics)

        # Save to CSV every 10 generations
        if len(self.metrics_data) % 10 == 0:
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.metrics_file, index=False)

    def emigrate_genes(self, best_gene):
        # Submit best gene
        gene_data = []
        for gene in best_gene:
            gene_data.append(save_individual_to_json(gene=gene))
        response = requests.post(
            f"{self.migration_server_url}/submit_gene", json=gene_data
        )
        if response.status_code == 200:
            return True
        else:
            return False
    def immigrate_genes(self):
        # Get mixed genes from server
        response = requests.get(f"{self.migration_server_url}/get_mixed_genes")
        received_genes_data = response.json()

        if not self.migration_server_url:
            return []

        return [
            load_individual_from_json(
                gene_data=gene_data, function_decoder=self.function_decoder
            )
            for gene_data in received_genes_data
        ]

    # One migration cycle
    def migrate_genes(self, best_gene):
        self.emigrate_genes(best_gene)
        return self.immigrate_genes()

    def create_baseline_model(self):
        return BaselineNN(input_size=28 * 28, hidden_size=128, output_size=10).to(
            self.device
        )

    def measure_baseline(self):
        set_seed(self.seed)
        train_loader, val_loader = self.load_data()
        baseline_model = self.create_baseline_model()
        self.train(baseline_model, train_loader)
        self.baseline_accuracy = self.evaluate(baseline_model, val_loader)
        logging.info(f"Baseline model accuracy: {self.baseline_accuracy:.4f}")

    def save_checkpoint(self, population, hof, best_individual_all_time, generation, random_state, torch_rng_state, numpy_rng_state, checkpoint_file):
        checkpoint = {
            'population': population,
            'hof': hof,
            'best_individual_all_time': best_individual_all_time,
            'generation': generation,
            'random_state': random_state,
            'torch_rng_state': torch_rng_state,
            'numpy_rng_state': numpy_rng_state
        }
        with open(checkpoint_file, 'wb') as cp_file:
            dill.dump(checkpoint, cp_file)


    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as cp_file:
            checkpoint = dill.load(cp_file)
        
        self.population = checkpoint['population']
        self.hof = checkpoint['hof']
        self.best_individual_all_time = checkpoint['best_individual_all_time']
        generation = checkpoint['generation']
        
        # Restore random states
        random.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        
        return generation


    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_model(self, genome):
        pass

    @abstractmethod
    def train(self, model, train_loader):
        pass

    @abstractmethod
    def evaluate(self, model, val_loader):
        pass

    def create_n_evaluate(self, individual, train_loader, val_loader):

        model = self.create_model(individual)
        try:
            self.train(model, train_loader=train_loader)
            fitness = self.evaluate(model, val_loader=val_loader)
        except:
            return (0.0,)

        return (fitness,)
    def log_mutated_child(self, offspring, generation):
        unpacked_code = self.unpacker.unpack_function_genome(offspring)
        log_filename = f"mutated_child_gen_{generation}.py"
        with open(log_filename, "w") as f:
            f.write(unpacked_code)
        logging.info(
            f"Logged mutated child for generation {generation} to {log_filename}"
        )

    def mine(self):
        self.measure_baseline()
        train_loader, val_loader = self.load_data()
        
        checkpoint_file = os.path.join(LOCAL_STORAGE_PATH, 'evolution_checkpoint.pkl')
        
        # Initialize variables
        if self.population is None:
            # Check if checkpoint exists
            if os.path.exists(checkpoint_file):
                # Load checkpoint
                logging.info("Loading checkpoint...")
                self.start_generation = self.load_checkpoint(checkpoint_file)
                logging.info(f"Resuming from generation {self.start_generation}")
            else:
                # No checkpoint, start fresh
                self.population = self.toolbox.population(n=self.config.Miner.population_size)
                self.hof = tools.HallOfFame(1)
                self.best_individual_all_time = None
                self.start_generation = 0
                logging.info("Starting evolution from scratch")
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        for generation in tqdm(range(self.start_generation, self.config.Miner.generations)):
            # Evaluate the entire population
            invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, train_loader, val_loader), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for i in range(0, len(offspring), 2):
                if random.random() < 0.5:
                    if i + 1 < len(offspring):
                        child1, child2 = offspring[i], offspring[i+1]
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(lambda ind: self.toolbox.evaluate(ind, train_loader, val_loader), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the population
            self.population[:] = offspring

            # Update the hall of fame with the generated individuals
            self.hof.update(self.population)

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.population if not math.isinf(ind.fitness.values[0])]
            length = len(fits)
            mean = sum(fits) / length if length > 0 else float('nan')
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5 if length > 0 else float('nan')
            
            logging.info(f"Generation {generation}: Valid fits: {len(fits)}/{len(self.population)}")
            logging.info(f"Min {min(fits) if fits else float('nan')}")
            logging.info(f"Max {max(fits) if fits else float('nan')}")
            logging.info(f"Avg {mean}")
            logging.info(f"Std {std}")

            best_individual = tools.selBest(self.population, 1)[0]

            if self.best_individual_all_time is None or best_individual.fitness.values[0] > self.best_individual_all_time.fitness.values[0]:
                self.best_individual_all_time = deepcopy(best_individual)
                logging.info(f"New best gene found. Pushing to {self.config.gene_repo}")
                self.push_to_remote(self.best_individual_all_time, f"{generation}_{self.best_individual_all_time.fitness.values[0]:.4f}")

            if generation % self.config.Miner.check_registration_interval == 0:
                self.config.bittensor_network.sync()

            logging.info(f"Generation {generation}: Best accuracy = {best_individual.fitness.values[0]:.4f}")
            
            # Save checkpoint at the end of the generation
            random_state = random.getstate()
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            self.save_checkpoint(self.population, self.hof, self.best_individual_all_time, generation + 1, random_state, torch_rng_state, numpy_rng_state, checkpoint_file)

        # Evolution finished
        logging.info("Evolution finished")
        
        # Remove the checkpoint file if evolution is complete
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        return self.best_individual_all_time


    @staticmethod
    def setup_logging(log_file="miner.log"):
        logger = logging.getLogger()
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
            # Create a logging queue
        log_queue = mp.Queue(-1)

        # Create a handler that writes to the log queue
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        # Create a listener that writes from the log queue to the console or a file
        listener = QueueListener(log_queue, logging.StreamHandler())
        listener.start()

        return listener

class BaseHuggingFaceMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(HuggingFacePushDestination(config.gene_repo))


class BaseMiningPoolMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(
            PoolPushDestination(config.Miner.pool_url, config.bittensor_network.wallet)
        )
        self.pool_url = config.Miner.pool_url

    # TODO add a timestamp or sth to requests to prevent spoofing signatures
    def register_with_pool(self):
        data = self._prepare_request_data("register")
        response = requests.post(f"{self.pool_url}/register", json=data)
        return response.json()["success"]

    def get_task_from_pool(self):
        data = self._prepare_request_data("get_task")
        response = requests.get(f"{self.pool_url}/get_task", json=data)
        return response.json()

    def submit_result_to_pool(self, best_genome):
        data = self._prepare_request_data("submit_result")
        data["result"] = save_individual_to_json(best_genome)
        response = requests.post(f"{self.pool_url}/submit_result", json=data)
        return response.json()["success"]

    def get_rewards_from_pool(self):
        data = self._prepare_request_data("get_rewards")
        response = requests.get(f"{self.pool_url}/get_rewards", json=data)
        return response.json()

    def update_config_with_task(self, task):
        # Update miner config with task-specific parameters if needed
        pass


class IslandMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.num_islands = config.Miner.num_processes
        self.migration_interval = config.Miner.migration_interval
        self.population_per_island = config.Miner.population_size // self.num_islands
        self.migrants_per_round = getattr(config.Miner, 'migrants_per_round', 1)
        self._shutdown_event = mp.Event()  # Use Event for process-safe shutdown

        self.best_global_fitness = mp.Value('d', -float('inf'))
        self.island_state_dict = mp.Manager().dict()  # Use Manager to share state
        self.checkpoint_lock = mp.Lock()
        self.checkpoint_file = os.path.join(LOCAL_STORAGE_PATH, 'island_evolution_checkpoint.pkl')

    def save_checkpoint(self):
        with self.checkpoint_lock:
            checkpoint = {
                'island_states': dict(self.island_state_dict),
                'best_global_fitness': self.best_global_fitness.value,
            }
            with open(self.checkpoint_file, 'wb') as cp_file:
                dill.dump(checkpoint, cp_file)
            logging.info("Checkpoint saved.")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as cp_file:
                    checkpoint = dill.load(cp_file)
                self.island_state_dict.update(checkpoint['island_states'])
                self.best_global_fitness.value = checkpoint['best_global_fitness']
                logging.info("Loaded checkpoint successfully")
                return True
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
                logging.info("Starting fresh due to failed checkpoint load")
                self.island_state_dict = mp.Manager().dict()
                self.best_global_fitness.value = -float('inf')
                return False
        else:
            logging.info("No checkpoint found, starting fresh")
            self.island_state_dict = mp.Manager().dict()
            self.best_global_fitness.value = -float('inf')
            return False

    def mine(self):
        try:
            # Create queues for migration and statistics
            migration_in_queues = [Queue() for _ in range(self.num_islands)]
            migration_out_queue = Queue()
            stats_queue = Queue()

            # Create shared variables for best fitness and genome
            self.best_global_fitness = Value('d', -float('inf'))
            manager = mp.Manager()
            self.best_global_genome = manager.list([None])

            # Create and start processes for each island
            processes = []
            for i in range(self.num_islands):
                p = Process(
                    target=run_island,
                    args=(
                        i,
                        self.seed,
                        self.config,
                        migration_in_queues[i],
                        migration_out_queue,
                        stats_queue
                    )
                )
                processes.append(p)
                p.start()

            # Wait for all processes to finish
            for p in processes:
                p.join()

            # Return the best genome found
            return self.best_global_genome[0]

        except KeyboardInterrupt:
            logging.info("Received interrupt signal, initiating shutdown...")
            self.shutdown()
            raise

        finally:
            # Clean up processes
            self.shutdown()
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=5)
            logging.info("All processes cleaned up")


    def shutdown(self):
        """Signal all islands to shutdown gracefully"""
        self._shutdown_event.set()
        logging.info("Shutdown signal sent to all islands")


class ActivationMiner(BaseMiner):

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_data = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        val_data = datasets.MNIST("../data", train=False, transform=transform)
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=128,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return train_loader, val_loader

    def create_model(self, individual):
        set_seed(self.seed)
        return EvolvableNN(
            input_size=28 * 28,
            hidden_size=128,
            output_size=10,
            evolved_activation=self.toolbox.compile(expr=individual),
        ).to(self.device)

    def train(self, model, train_loader):
        set_seed(self.seed)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == self.config.Miner.training_iterations:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def evaluate(self, model, val_loader):
        set_seed(self.seed)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if idx >= self.config.Miner.evaluation_iterations:
                    return correct / total
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total


class LossMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        # self.seed_population()

    # def seed_population(self, mse = True):
    #     population = self.toolbox.population(n=50)
    #     mse_population = [seed_with_mse(len(ind), ind.memory, ind.function_decoder,
    #                                     ind.input_addresses, ind.output_addresses) for ind in population]
    #     return mse_population

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_data = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        val_data = datasets.MNIST("../data", train=False, transform=transform)
        train_loader = DataLoader(
            train_data,
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=128,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return train_loader, val_loader

    def create_model(self, individual):
        set_seed(self.seed)
        return BaselineNN(input_size=28 * 28, hidden_size=128, output_size=10).to(
            self.device
        ), self.toolbox.compile(expr=individual)

    @staticmethod
    def safe_evaluate(func, outputs, labels):
        try:
            loss = func(outputs, labels)

            if loss is None:
                logging.error(f"Loss function returned None: {func}")
                return torch.tensor(float("inf"), device=outputs.device)

            if not torch.is_tensor(loss):
                logging.error(f"Loss function didn't return a tensor: {type(loss)}")
                return torch.tensor(float("inf"), device=outputs.device)

            if not torch.isfinite(loss).all():
                logging.warning(f"Non-finite loss detected: {loss}")
                return torch.tensor(float("inf"), device=outputs.device)

            if loss.ndim > 0:
                loss = loss.mean()

            return loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            # logging.error(traceback.format_exc())
            return torch.tensor(float("inf"), device=outputs.device)

    def train(self, model_and_loss, train_loader):
        set_seed(self.seed)
        model, loss_function = model_and_loss
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == 2:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            targets_one_hot = torch.nn.functional.one_hot(
                targets, num_classes=10
            ).float()
            loss = self.safe_evaluate(loss_function, outputs, targets_one_hot)
            loss.backward()
            optimizer.step()

    def evaluate(self, model_and_loss, val_loader):
        set_seed(self.seed)
        model, _ = model_and_loss
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if idx > 10:
                    break
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

    def create_baseline_model(self):
        return (
            BaselineNN(input_size=28 * 28, hidden_size=128, output_size=10).to(
                self.device
            ),
            torch.nn.MSELoss(),
        )


class ActivationMinerPool(ActivationMiner, BaseMiningPoolMiner):
    pass


class ActivationMinerHF(ActivationMiner, BaseHuggingFaceMiner):
    pass


class ParallelActivationMiner(ActivationMiner, IslandMiner):
    pass


class ParallelActivationMinerPool(ParallelActivationMiner, BaseMiningPoolMiner):
    pass


class ParallelActivationMinerHF(ParallelActivationMiner, BaseHuggingFaceMiner):
    pass


class ParallelLossMiner(LossMiner, IslandMiner):
    pass


class LossMinerPool(LossMiner, BaseMiningPoolMiner):
    pass


class LossMinerHF(LossMiner, BaseHuggingFaceMiner):
    pass


class ParallelLossMinerPool(ParallelLossMiner, BaseMiningPoolMiner):
    pass


class ParallelLossMinerHF(ParallelLossMiner, BaseHuggingFaceMiner):
    pass


class MinerFactory:
    @staticmethod
    def get_miner(config):
        miner_type = config.Miner.miner_type
        platform = config.Miner.push_platform
        core_count = config.Miner.num_processes
        # mp.set_start_method('spawn')

        if platform == "pool":
            if miner_type == "activation":
                if core_count == 1:
                    return ActivationMinerPool(config)
                else:
                    return ParallelActivationMinerPool(config)
            elif miner_type == "loss":
                if core_count == 1:
                    return LossMinerPool(config)
                else:
                    return ParallelActivationMinerPool(config)
        elif platform == "hf":
            if miner_type == "activation":
                if core_count == 1:
                    return ActivationMinerHF(config)
                else:
                    return ParallelActivationMinerHF(config)
            elif miner_type == "loss":
                if core_count == 1:
                    return LossMinerHF(config)
                else:
                    return ParallelLossMinerHF(config)

        raise ValueError(f"Unknown miner type: {miner_type} or platform: {platform}")
