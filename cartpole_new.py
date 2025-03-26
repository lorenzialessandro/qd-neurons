import numpy as np
import torch
import os
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import gymnasium as gym
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter

from network import NCHL, Neuron
from utils import *

def evaluate_network(network, seed=None, n_episodes=50):
    """Evaluate a single network"""
    if seed is not None:
        # Set different seeds for different processes
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    env = gym.make('CartPole-v1')
    total_reward = 0
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

        total_reward += episode_reward
        rewards.append(episode_reward)

    env.close()

    # Return statistics
    return {
        'percentile_70': np.percentile(rewards, 70),
        'cumulative_mean_reward': total_reward / n_episodes,
    }

def evaluate_networks_parallel(networks, n_episodes, n_workers=None):
    """Evaluate multiple networks in parallel"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(networks))

    # Create a pool of workers
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a partial function with fixed n_episodes
        eval_fn = partial(evaluate_network, n_episodes=n_episodes)

        # Add different seeds to avoid identical random sequences in parallel
        seeds = [random.randint(0, 10000) for _ in range(len(networks))]
        jobs = [executor.submit(eval_fn, net, seed)
                for net, seed in zip(networks, seeds)]

        # Gather results as they complete
        results = []
        for job in as_completed(jobs):
            results.append(job.result())

    return results

def initialize_archives(pop, config):
    archives = {neuron.neuron_id: GridArchive(
        solution_dim=5,
        dims=[10, 10],
        ranges=[(1.5, 3), (0, 1)],
        seed=config["seed"]
    ) for neuron in pop}

    return archives

def initialize_emitters(pop, config, archives):
    emitters = {neuron.neuron_id: EvolutionStrategyEmitter(
        archive=archives[neuron.neuron_id],
        x0=np.random.uniform(-1, 1, 5),
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

def main():
    parser = argparse.ArgumentParser(description='Cartpole Task')
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='Path to config file')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    args = parser.parse_args()
    config = load_config(args.config)

    # Set number of workers
    n_workers = args.workers if args.workers is not None else mp.cpu_count()
    print(f"Using {n_workers} parallel workers")

    # Set random seed for reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    n_nodes = sum(config["nodes"])

    # Create initial population
    pop = [Neuron(neuron_id=i) for i in range(n_nodes)]

    archives = initialize_archives(pop, config)
    emitters = initialize_emitters(pop, config, archives)
    schedulers = initialize_schedulers(pop, emitters, archives)
    
    nets = []

    for i in tqdm(range(config["iterations"])):

        # 1. Ask params for each neuron
        for neuron in pop:
            sol = schedulers[neuron.neuron_id].ask()[0]
            neuron.set_params(sol)

        # 2. Create networks from teams
        nets = create_nets(pop, config)

        # 3. Evaluate networks in parallel
        team_results = evaluate_networks_parallel(nets,
                                                  n_episodes=config["episodes"],
                                                  n_workers=n_workers)

        # Track fitness results
        history_fitness = [result['cumulative_mean_reward']
                           for result in team_results]

        # 4. Collect descriptors and objectives
        objectives = defaultdict(list)
        descriptors = defaultdict(list)

        for net, result in zip(nets, team_results):
            fitness = result['cumulative_mean_reward']

            # Collect descriptors and objectives for each neuron
            for neuron in net.all_neurons:
                entropy, weight_change = neuron.compute_descriptors()
                descriptors[neuron.neuron_id].append([entropy, weight_change])
                objectives[neuron.neuron_id].append(fitness)

        # 5. Tell the scheduler the fitness and descriptors of each neuron
        for neuron in pop:
            # Aggregate the descriptors and objectives
            avg_obj = np.mean(objectives[neuron.neuron_id])
            avg_entropy = np.mean([d[0]
                                  for d in descriptors[neuron.neuron_id]])
            avg_weight_change = np.mean(
                [d[1] for d in descriptors[neuron.neuron_id]])

            # Update the scheduler
            schedulers[neuron.neuron_id].tell(
                objective=[avg_obj],
                measures=[(avg_entropy, avg_weight_change)]
            )

        if i % 10 == 0:
            print(
                f"Iteration: {i}, Mean Fitness: {np.mean(history_fitness):.2f}")

    plot_nets_heatmaps(nets, archives)


if __name__ == '__main__':
    # Required for parallel processing on Windows
    mp.set_start_method('spawn', force=True)
    main()
