import numpy as np
import torch
import os
import random
import argparse
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import gymnasium as gym

from qd import *
from network import NCHL, Neuron
from utils import *
from analysis import *

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"results/qd_mountaincar_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/qd_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_team(network, n_episodes=10):
    """Evaluate a network on MountainCar-v0"""
    env = gym.make("MountainCar-v0")
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

            # Update weights using Hebbian learning
            network.update_weights()

        rewards.append(episode_reward)

    # Calculate statistics
    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    std_reward = np.std(rewards)
    
    return {
        'mean_reward': mean_reward, 
        'max_reward': max_reward,
        'std_reward': std_reward
    }

def evaluate_team_parallel(network_config):
    """Wrapper for parallel evaluation"""
    network, n_episodes = network_config
    return network, evaluate_team(network, n_episodes)

def initialize_archives(archives, pop, config, n_samples=10):
    """Initialize archives with random rules instead of starting empty"""
    logger.info("Initializing archives with random rules...")
    
    # Set up multiprocessing
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)
    
    # Generate multiple samples of random rules
    for _ in range(n_samples):
        # Set random rules for each neuron
        for neuron in pop:
            archive = archives[neuron.neuron_id]
            random_rule = archive.get_random_rule()
            neuron.set_params(random_rule)
        
        # Create networks and evaluate them
        networks = create_nets(pop, config, n_teams=config.get("n_teams", 5))
        network_configs = [(net, 5) for net in networks]
        
        # Evaluate in parallel
        evaluation_results = pool.map(evaluate_team_parallel, network_configs)
        
        # Process results and update archives
        objectives = defaultdict(list)
        descriptors = defaultdict(list)
        
        for net, result in evaluation_results:
            fitness = result['mean_reward']
            
            # Collect data for each neuron
            for neuron in net.all_neurons:
                behavior, complexity = neuron.compute_new_descriptor()
                descriptors[neuron.neuron_id].append([behavior, complexity])
                objectives[neuron.neuron_id].append(fitness)
        
        # Update archives with aggregated metrics
        for neuron in pop:
            neuron_fitness = objectives[neuron.neuron_id]
            if not neuron_fitness:  # Skip if no data for this neuron
                continue
                
            # Use mean fitness for initialization
            mean_fitness = np.mean(neuron_fitness)
            
            # Average descriptors
            avg_behavior = np.mean([d[0] for d in descriptors[neuron.neuron_id]])
            avg_complexity = np.mean([d[1] for d in descriptors[neuron.neuron_id]])
            
            # Update archive
            archives[neuron.neuron_id].tell(
                behavior=[avg_behavior, avg_complexity], 
                fitness=mean_fitness, 
                rule=neuron.get_rule()
            )
    
    # Clean up
    pool.close()
    pool.join()
    
    # Log coverage after initialization

    plot_heatmaps(pop, archives, output_dir, True)
    logger.info("Archives initialized with random rules.")    
    return archives

def create_balanced_teams(pop, config, elites=None, n_teams=10):
    """Create teams with a mix of elite and exploratory neurons"""
    nets = []
    
    # Sort neurons by their performance in previous teams (if available)
    neuron_scores = getattr(create_balanced_teams, 'neuron_scores', {}) # Track neuron scores across calls to this function
    
    # Use some elite networks directly
    if elites and len(elites) > 0:
        elite_ratio = 0.3  # 30% of teams come from elites
        elite_count = max(1, int(n_teams * elite_ratio))
        nets.extend(elites[:elite_count])
        
        # Update neuron scores from elite networks
        for elite_net in elites[:elite_count]:
            for neuron in elite_net.all_neurons:
                neuron_id = neuron.neuron_id
                # Increase score for neurons in elite networks
                neuron_scores[neuron_id] = neuron_scores.get(neuron_id, 0) + 1
    
    # Create remaining networks
    remaining = n_teams - len(nets)
    
    # # New option: create teams with random neurons
    # for _ in range(remaining):
    #     team_neurons = pop.copy()
    #     random.shuffle(team_neurons)
    #     team_neurons = team_neurons[:len(pop)]  # Keep same number of neurons
    #     net = NCHL(nodes=config["nodes"], population=team_neurons)
    #     nets.append(net)
        
    # # Store the neuron scores for next call
    # create_balanced_teams.neuron_scores = neuron_scores
    
    # return nets
    
    for _ in range(remaining):
        # For each network, mix some neurons from high-scoring ones and some random ones
        team_neurons = []
        nodes = config["nodes"]
        
        # Build team layer by layer
        start_idx = 0
        for layer_idx, node_count in enumerate(nodes):
            layer_neurons = pop[start_idx:start_idx+node_count]
            
            # Sort layer neurons by score (higher is better)
            sorted_neurons = sorted(
                layer_neurons, 
                key=lambda n: neuron_scores.get(n.neuron_id, 0), 
                reverse=True
            )
            
            # Take some top neurons and some random ones
            if len(sorted_neurons) > 2:
                # Take 50% top neurons
                top_count = max(1, node_count // 2)
                top_neurons = sorted_neurons[:top_count]
                
                # Take 50% random neurons
                remaining_neurons = sorted_neurons[top_count:]
                random.shuffle(remaining_neurons)
                random_neurons = remaining_neurons[:node_count-top_count]
                
                # Combine and shuffle
                combined = top_neurons + random_neurons
                random.shuffle(combined)
                team_neurons.extend(combined)
            else:
                # For very small layers, just use all neurons
                team_neurons.extend(sorted_neurons)
            
            start_idx += node_count
        
        # Create network
        net = NCHL(nodes=nodes, population=team_neurons)
        nets.append(net)
    
    # Store the neuron scores for next call
    create_balanced_teams.neuron_scores = neuron_scores
    
    return nets

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

def run_qd_with_tweaks(config):
    """Run QD algorithm"""
    # Set random seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    # Network topology
    network_topology = config["nodes"]
    n_nodes = sum(network_topology)
    
    # Create initial neuron population
    pop = [Neuron(neuron_id=i) for i in range(n_nodes)]
    
    # Create archives for each neuron
    archives = {
        neuron.neuron_id: NeuronArchive(
            dims=config["dims"],   
            ranges = config["ranges"],
            sigma=config["sigma"],
            seed=config["seed"] + neuron.neuron_id 
        ) for neuron in pop
    }
    
    # Initialize archives with random rules
    archives = initialize_archives(archives, pop, config, n_samples=5)
    
    # Set up multiprocessing
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)
    
    # Tracking metrics
    best_fitness_history = []
    avg_fitness_history = []
    elite_teams = []
    
    # Main loop
    for iteration in tqdm(range(config["iterations"])):
        # exploration_rate = max(0.2, 0.9 - (iteration / config["iterations"]) * 0.7) 
        # Set exploration rate high at the beginning and decrease it slowly
        
        # slow_rate = 0.5 # how fast to decrease exploration rate : if is 0.5, it will be 0.9 at the beginning and 0.4 at the end
        # exploration_rate = max(0.2, 0.9 - (iteration / config["iterations"]) * slow_rate)
        
        # exploration_rate = 0.2 + 0.7 * np.exp(-iteration / (config["iterations"] * 0.3))
        
        exploration_rate = config["exploration_rate"] 
                
        # 1. Update neuron parameters
        for neuron in pop:
            archive = archives[neuron.neuron_id]
            
            if random.random() < exploration_rate or archive.empty:
                # Exploration: use emitter and take random solution
                solution = archive.ask()
            else:
                # Exploitation: select good solution from archive
                # Uses fitness-proportional selection to choose the best neurons with increasing selection pressure
                archive_data = archive.data()
                fitnesses = archive_data["fitness"]
                
                # Adaptive power scaling that increases pressure over time
                # First shift all fitnesses to be positive
                min_fitness = np.min(fitnesses)
                shifted_fitnesses = fitnesses - min_fitness + 1e-10  # Small epsilon to avoid zeros
                                
                # Apply power scaling
                power = 1 + 3 * (iteration / config["iterations"])
                probabilities = np.power(shifted_fitnesses, power)
                probabilities = probabilities / np.sum(probabilities)
                
                selected_idx = np.random.choice(len(fitnesses), p=probabilities)
                solution = archive_data["rule"][selected_idx]
            
            # Apply solution parameters to neuron
            neuron.set_params(solution)
        
        # 2. Create networks using balanced team formation
        n_teams = config.get("n_teams", 10) # Number of teams to create 
        networks = create_balanced_teams(pop, config, elites=elite_teams, n_teams=n_teams)
        # networks = create_nets(pop, config, n_teams=n_teams)
        
        # 3. Evaluate networks in parallel
        network_configs = [(net, config.get("episodes", 5)) for net in networks]
        evaluation_results = pool.map(evaluate_team_parallel, network_configs)
        
        # 4. Process results
        objectives = defaultdict(list)
        descriptors = defaultdict(list)
        
        all_fitness = []
        for net, result in evaluation_results:
            fitness = result['mean_reward']
            all_fitness.append(fitness)
            
            # Collect data for each neuron
            for neuron in net.all_neurons:
                behavior, complexity = neuron.compute_new_descriptor()
                # behavior, complexity = neuron.compute_descriptors()
                # behavior, complexity = neuron.compute_new_descriptor_2()
                #print(f"Neuron {neuron.neuron_id}: {behavior}, {complexity}")
                
                descriptors[neuron.neuron_id].append([behavior, complexity])
                objectives[neuron.neuron_id].append(fitness)
                # print(f"Neuron {neuron.neuron_id}: {fitness}, {behavior}, {complexity}")
        
        # 5. Update archives with aggregated metrics
        for neuron in pop:
            neuron_fitness = objectives[neuron.neuron_id]
            if not neuron_fitness:  # Skip if no data for this neuron
                continue
            
            # Alternatively, use 70th percentile fitness
            combined_fitness = np.percentile(neuron_fitness, 70)
            
            # Average descriptors
            avg_behavior = np.mean([d[0] for d in descriptors[neuron.neuron_id]])
            avg_complexity = np.mean([d[1] for d in descriptors[neuron.neuron_id]])
            
            # Update archive
            archives[neuron.neuron_id].tell(
                behavior=[avg_behavior, avg_complexity], 
                fitness=combined_fitness, 
                rule=neuron.get_rule() # neuron.params
            )
        
        # 6. Select elite teams for next iteration
        evaluation_results.sort(key=lambda x: x[1]['mean_reward'], reverse=True)
        elite_teams = [net for net, _ in evaluation_results[:max(2, n_teams//10)]]
        
        # Record metrics for this iteration
        iteration_best = max(all_fitness)
        iteration_avg = np.mean(all_fitness)
        
        best_fitness_history.append(iteration_best)
        avg_fitness_history.append(iteration_avg)
        
        # Log progress
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: Best={iteration_best:.1f}, Avg={iteration_avg:.1f}, Exploration Rate={exploration_rate:.2f}")
        
        # Check for convergence
        if iteration_best >= config["threshold"]:
            logger.info(f"Task solved at iteration {iteration}!")

            
    
    # Clean up
    pool.close()
    pool.join()
    

    # Plotting results
    plot_fitness_trends(best_fitness_history, avg_fitness_history, output_dir, config["threshold"])
    plot_heatmaps(pop, archives, output_dir)
    # plot_analysis(pop, archives, output_dir)
    
    # log each archive stats
    for neuron in pop:
        archive = archives[neuron.neuron_id]
        logger.info(f"Neuron {neuron.neuron_id}: {archive.coverage()}")
        
    # Test the best network
    best_network = elite_teams[0]
    # Save the best network
    save_network(best_network, output_dir)
    # exit()
    
    solving_result = evaluate_team(best_network, n_episodes=100)
    mean_reward = solving_result['mean_reward']
    
    if mean_reward >= config["threshold"]:
        logger.info("Best network solved the task!")
    else:
        logger.info("Best network did not solve the task.")
    
    return {
        'best_fitness': best_fitness_history,
        'avg_fitness': avg_fitness_history
    }

if __name__ == "__main__":
    # Configuration
    config = {
        "exploration_rate": 0, #TESTING
        "seed": 123,
        "nodes": [2, 8, 3],  # Input, hidden, output layers
        "iterations": 200,
        "threshold": -110,
        "episodes": 10,
        "n_teams": 15,
        "dims": [15, 15],   
        "ranges": [(0, 1), (0, 1)],
        "sigma": 1
    }
    
    logger.info("Starting QD MountainCar Training")
    logger.info(f"Configuration: {config}")
    
    # Run QD training
    best_network, history = run_qd_with_tweaks(config)