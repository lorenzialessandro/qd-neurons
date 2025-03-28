import numpy as np
import torch
import os
import random
import argparse
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import logging
from datetime import datetime

import gymnasium as gym
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter

from network import NCHL, Neuron
from utils import *

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
path = f"tests/{timestamp}"
os.makedirs(path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f'{path}/cartpole.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def evaluate_team(network, n_episodes=50):
    env = gym.make('CartPole-v1')
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            input = torch.tensor(state).double()
            output = network.forward(input)
            action = np.argmax(output.tolist())

            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            network.update_weights()

        rewards.append(episode_reward)

    # Mean reward across all episodes
    mean_reward = np.mean(rewards)
    env.close()
    
    result = {
        'percentile_70': np.percentile(rewards, 70),
        'mean_reward': mean_reward,
    }
    
    return result

def evaluate_team_parallel(network_config):
    """Wrapper for parallel evaluation"""
    network, n_episodes = network_config
    return network, evaluate_team(network, n_episodes)

def initialize_archives(pop, config):
    archives = {neuron.neuron_id: GridArchive(
        solution_dim=5,
        dims=[10, 10],
        ranges=[(2.5, 4), (0, 1)],
        seed=config["seed"]
    ) for neuron in pop}

    return archives

def initialize_emitters(pop, config, archives):
    emitters = {neuron.neuron_id: EvolutionStrategyEmitter(
        archive=archives[neuron.neuron_id],
        x0=np.append(np.random.uniform(-1, 1, 4), 0.001),
        sigma0=0.2,
        batch_size=1,
        seed=config["seed"]
    ) for neuron in pop}

    return emitters

def initialize_schedulers(pop, emitters, archives):
    schedulers = {neuron.neuron_id: Scheduler(
        emitters=[emitters[neuron.neuron_id]],
        archive=archives[neuron.neuron_id]
    ) for neuron in pop}

    return schedulers

def create_nets(pop, config, n_teams=10):
    nets = []
    teams = []
    # Create random teams of neurons
    for _ in range(n_teams):
        p = pop.copy()
        random.shuffle(p)
        teams.append(p)

    # Create networks for each team with corresponding population
    for team in teams:
        net = NCHL(nodes=config["nodes"], population=team)
        nets.append(net)

    return nets

def selection_fitness_proportional(archive):
    """Select a solution from the archive using fitness-proportional selection"""
    data = archive.data()
    fitnesses = np.array(data["objective"])
    x = 5 # Selection pressure
    probabilities = (fitnesses ** x) / np.sum(fitnesses ** x)
    index = np.random.choice(len(probabilities), p=probabilities)
    params = data["solution"][index]
    return params

def main():
    parser = argparse.ArgumentParser(description='Cartpole Task')
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='Path to config file')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    args = parser.parse_args()
    config = load_config(args.config)
    
    os.makedirs('log', exist_ok=True)
    
    logger.info("Starting Cartpole Task")
    logger.info(f"Configuration: {config}")

    # Set random seed for reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    n_nodes = sum(config["nodes"])

    # Create initial population
    pop = [Neuron(neuron_id=i) for i in range(n_nodes)]

    # Initialize one archive per neuron as required
    archives = initialize_archives(pop, config)
    emitters = initialize_emitters(pop, config, archives)
    schedulers = initialize_schedulers(pop, emitters, archives)
    
    # Set up multiprocessing for network evaluation
    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)
    
    nets = []
    best_fitness_per_iteration = []
    avg_fitness_per_iteration = []
    median_fitness_per_iteration = []

    for i in tqdm(range(config["iterations"])):
        iteration_fitness = []

        # 1. Ask params for each neuron
        for neuron in pop:
            sol = schedulers[neuron.neuron_id].ask()[0]
            if not archives[neuron.neuron_id].empty:
                # Select a solution using fitness-proportional selection (overwrites sol because need ask() before tell())
                sol = selection_fitness_proportional(archives[neuron.neuron_id]) 
            
            neuron.set_params(sol)

        # 2. Create networks from teams
        nets = create_nets(pop, config)

        # 3. Parallel evaluation of networks
        network_configs = [(net, config["episodes"]) for net in nets]
        eval_results = pool.map(evaluate_team_parallel, network_configs)

        # 4. Collect descriptors and objectives
        objectives = defaultdict(list)
        descriptors = defaultdict(list)

        for net, result in eval_results:
            fitness = result['mean_reward']
            iteration_fitness.append(fitness)
            
            # Collect descriptors and objectives for each neuron
            for neuron in net.all_neurons:
                entropy, weight_change = neuron.compute_descriptors()
                
                descriptors[neuron.neuron_id].append([entropy, weight_change])
                objectives[neuron.neuron_id].append(fitness)

        # 5. Tell the scheduler the fitness and descriptors of each neuron
        for neuron in pop:
            # Aggregate the descriptors and objectives
            avg_obj = np.mean(objectives[neuron.neuron_id])
            avg_entropy = np.mean([d[0] for d in descriptors[neuron.neuron_id]])
            avg_weight_change = np.mean([d[1] for d in descriptors[neuron.neuron_id]])
    
            # Update the scheduler
            schedulers[neuron.neuron_id].tell(
                objective=[avg_obj],
                measures=[(avg_entropy, avg_weight_change)]
            )

        # Save the best and average fitness for this iteration
        best_fitness = max(iteration_fitness)
        avg_fitness = np.mean(iteration_fitness)
        median_fitness = np.median(iteration_fitness)
            
        best_fitness_per_iteration.append(best_fitness)
        avg_fitness_per_iteration.append(avg_fitness)
        median_fitness_per_iteration.append(median_fitness)

        if i % 10 == 0:
            print(f"Iteration: {i}, Best Fitness: {best_fitness}, Avg Fitness: {avg_fitness}")
            logger.info(f"Iteration: {i}, Best Fitness: {best_fitness}, Avg Fitness: {avg_fitness}")
        # Check if task is solved
        if best_fitness >= config["threshold"]:
            print(f"Task solved at iteration {i}")
            # break

    # Cleanup
    pool.close()
    pool.join()
    
    logger.info("Cartpole Task completed")
    
    # --- Plotting ---
    plot_fitness_trends(best_fitness_per_iteration, avg_fitness_per_iteration, median_fitness_per_iteration, path, config["threshold"])
    plot_heatmaps(pop, archives, path)
    plot_pcas(pop, archives, path)
    plot_pca_best_rules(pop, archives, path)
    

    
if __name__ == '__main__':
    main()