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
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter

from network import NCHL, Neuron
from utils import *
from analysis import *

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"results/qd_cartpole_{timestamp}"
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
    """Evaluate a network on CartPole"""
    # env = gym.make('CartPole-v1')
    env = gym.make("CartPole-v1")
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

def create_balanced_teams(pop, config, elites=None, n_teams=10):
    """Create teams with a mix of elite and exploratory neurons
    
    The goal is to create a balanced team with a mix of neurons that have performed well in the past and some random neurons.
    - 30% of teams are directly taken from elite networks (elites are top-performing networks from previous iterations)
    - 70% of teams are created by mixing neurons 
        - [exploitation] 50% of neurons are top-performing neurons from previous iterations (for each layer, sorts neurons by their historical performance scores)
        - [exploration]  50% of neurons are randomly selected from the remaining neurons
        
    This creates a good balance between exploration and exploitation, and helps in convergence.
    As the process continues, the top-performing neurons are more likely to be selected, which helps in convergence.
    """
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
    
    # New option: create teams with random neurons
    for _ in range(remaining):
        team_neurons = pop.copy()
        random.shuffle(team_neurons)
        team_neurons = team_neurons[:len(pop)]  # Keep same number of neurons
        net = NCHL(nodes=config["nodes"], population=team_neurons)
        nets.append(net)
        
    # Store the neuron scores for next call
    create_balanced_teams.neuron_scores = neuron_scores
    
    return nets
    
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
    
    # Initialize QD components
    # Create archives for each neuron
    archives = {
        neuron.neuron_id: GridArchive(
            solution_dim=5,  # 5 parameters per neuron
            dims=[10, 10],   # 10x10 grid 
            ranges=[(0, 1), (0, 1)], 
            seed=config["seed"] + neuron.neuron_id  # Different seed per neuron
        ) for neuron in pop
    }
    
    # Create emitters for each neuron
    emitters = {
        neuron.neuron_id: EvolutionStrategyEmitter(
            archive=archives[neuron.neuron_id],
            x0=np.append(np.random.uniform(-0.1, 0.1, 4), 0.005),  # Init values
            sigma0=0.1,  
            batch_size=1,  
            seed=config["seed"] + neuron.neuron_id
        ) for neuron in pop
    }
    
    # Create schedulers for each neuron
    schedulers = {
        neuron.neuron_id: Scheduler(
            emitters=[emitters[neuron.neuron_id]],
            archive=archives[neuron.neuron_id]
        ) for neuron in pop
    }
    
    # Set up multiprocessing
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)
    
    # Tracking metrics
    best_fitness_history = []
    avg_fitness_history = []
    elite_teams = []
    
    # Main loop
    for iteration in tqdm(range(config["iterations"])):
        # Balance between exploration and exploitation
        # Starts with 70% exploration and gradually reduces to 10% as training progresses
        exploration_rate = max(0.1, 0.7 - (iteration / config["iterations"]) * 0.6) # linear decay : 0.7 -> 0.1 
        # example: 0.7 - (0/100) * 0.6 = 0.7, 0.7 - (50/100) * 0.6 = 0.4, 0.7 - (100/100) * 0.6 = 0.1
        
        # 1. Update neuron parameters
        for neuron in pop:
            archive = archives[neuron.neuron_id]
            
            solution = schedulers[neuron.neuron_id].ask()[0]
            if random.random() < exploration_rate or archive.empty:
                # Exploration: use emitter and take random solution
                solution = solution
            else:
                # Exploitation: select good solution from archive
                # Uses fitness-proportional selection to choose the best neurons with increasing selection pressure
                archive_data = archive.data()
                fitnesses = archive_data["objective"]
                
                # Adaptive power scaling that increases pressure over time
                power = 1 + 3 * (iteration / config["iterations"]) # linear scaling from 1 to 4
                probabilities = np.power(fitnesses, power) # fitnesses^power => higher fitnesses get higher probabilities
                probabilities = probabilities / np.sum(probabilities)
                
                selected_idx = np.random.choice(len(fitnesses), p=probabilities) 
                solution = archive_data["solution"][selected_idx]
            
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
                # print(f"Neuron {neuron.neuron_id}: {behavior}, {complexity}")
                
                # behavior, complexity = neuron.compute_descriptors()
                descriptors[neuron.neuron_id].append([behavior, complexity])
                objectives[neuron.neuron_id].append(fitness)
        
        # 5. Update archives with aggregated metrics
        for neuron in pop:
            neuron_fitness = objectives[neuron.neuron_id]
            if not neuron_fitness:  # Skip if no data for this neuron
                continue
                
            # Use a combination of mean and max for more balanced optimization
            weight = min(0.8, iteration / (config["iterations"] * 0.5)) # Gradually shift from mean to max
            mean_fitness = np.mean(neuron_fitness)
            max_fitness = np.max(neuron_fitness)
            
            combined_fitness = (1 - weight) * mean_fitness + weight * max_fitness
            
            # Alternatively, use 70th percentile fitness
            combined_fitness = np.percentile(neuron_fitness, 70)
            
            # Average descriptors
            avg_behavior = np.mean([d[0] for d in descriptors[neuron.neuron_id]])
            avg_complexity = np.mean([d[1] for d in descriptors[neuron.neuron_id]])
            
            # Update archive
            schedulers[neuron.neuron_id].tell(
                objective=[combined_fitness],
                measures=[(avg_behavior, avg_complexity)]
            )
        
        # 6. Select elite teams for next iteration
        evaluation_results.sort(key=lambda x: x[1]['mean_reward'], reverse=True)
        elite_teams = [net for net, _ in evaluation_results[:max(2, n_teams//10)]] # Keep top 20% of teams as elites
        
        # Record metrics for this iteration
        iteration_best = max(all_fitness)
        iteration_avg = np.mean(all_fitness)
        
        best_fitness_history.append(iteration_best)
        avg_fitness_history.append(iteration_avg)
        
        # Log progress
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: Best={iteration_best:.1f}, Avg={iteration_avg:.1f}")
        
        # Check for convergence
        if iteration_best >= config["threshold"]:
            logger.info(f"Task solved at iteration {iteration}!")
    
    # Clean up
    pool.close()
    pool.join()
    
    # ------------ FINAL NETWORK ------------
    # TODO: change this strategy of creating final network
    # Create final network with best parameters from archives
    best_neurons = []
    for neuron in pop:
        archive = archives[neuron.neuron_id]
        if archive.empty:
            # Use current neuron parameters if archive is empty
            best_neuron = neuron
        else:
            # Get best parameters from archive
            best_idx = np.argmax(archive.data()["objective"])
            best_params = archive.data()["solution"][best_idx]
            
            # Create new neuron with these parameters
            best_neuron = Neuron(neuron_id=neuron.neuron_id)
            best_neuron.set_params(best_params)
        
        best_neurons.append(best_neuron)
    
    
    # ------------ TESTING ------------
    #TODO: change this testing strategy
    
    # Create best network
    best_network = NCHL(nodes=network_topology, population=best_neurons)
    # Evaluate final network
    final_result = evaluate_team(best_network, n_episodes=100)
    logger.info(f"Final performance: {final_result}")
    
    # ------------ PLOTTING ------------
    
    # Save results

    plot_fitness_trends(best_fitness_history, avg_fitness_history, output_dir, config["threshold"])
    save_network_params(best_network, output_dir)
    plot_heatmaps(pop, archives, output_dir)
    # plot_pcas(pop, archives, output_dir)
    # plot_k_pcas(pop, archives, output_dir, k=5)
    # plot_pca_best_rules(pop, archives, output_dir)
    plot_analysis(pop, archives, output_dir)
    
    # log each archive stats
    for neuron in pop:
        archive = archives[neuron.neuron_id]
        logger.info(f"Neuron {neuron.neuron_id}: {archive.stats}")
    
    return best_network, {
        'best_fitness': best_fitness_history,
        'avg_fitness': avg_fitness_history
    }

if __name__ == "__main__":
    # Configuration
    config = {
        "seed": 5,
        "nodes": [4, 4, 2],  # Input, hidden, output layers
        "iterations": 100,
        "threshold": 475,
        "episodes": 10,
        "n_teams": 10
    }
    
    logger.info("Starting QD CartPole Training")
    logger.info(f"Configuration: {config}")
    
    # Run QD training
    best_network, history = run_qd_with_tweaks(config)