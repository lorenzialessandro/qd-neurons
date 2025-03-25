import numpy as np
import torch
import os
import random
import argparse
import datetime
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import gymnasium as gym
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter

from network import NCHL, Neuron
from utils import *

def create_teams(initial_pop, n_shuffle=10, team_size=10):
    teams = []
    for i in range(n_shuffle):
        pop = initial_pop.copy()
        random.shuffle(pop)

        for j in range(0, len(pop), team_size):
            teams.append(pop[j:j+team_size])

    return teams

def evaluate_team(network, n_episodes=50):
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
        # 70th percentile of episode rewards
        'percentile_70': np.percentile(rewards, 70),
        'mean_reward': total_reward / n_episodes,     # Standard solving criterion
        'raw_rewards': rewards                        # Raw episode rewards
    }

def main():
    parser = argparse.ArgumentParser(description='Cartpole Task')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)

    # archive : store individual neuron solutions
    archive = GridArchive(
        solution_dim=5,  # 5 parameters for each neuron
        dims=[10, 10],
        ranges=[(1.5, 3), (0, 1)],  # average entropy and weight change
        seed=config["seed"],
    )

    emitter = EvolutionStrategyEmitter(
        archive=archive,
        x0=np.random.uniform(-1, 1, 5),  # initial solution randomly generated
        sigma0=0.2,  # initial standard deviation
        batch_size=10,  
        seed=config["seed"],
    )

    scheduler = Scheduler(emitters=[emitter], archive=archive)


    # Initialize tracking variables
    best_fitness = -float('inf')
    history = []
    history_best = []
    history_mean = []  # Track mean rewards separately
    
    # track all net with its own fitness
    nets = {
        'fitness': [],
        'network': []
    }

    for i in tqdm(range(config["iterations"])):
        # Ask the scheduler for a new solution
        sol = scheduler.ask()
        # Create a population of neurons from the solution and shuffle them into teams
        pop = [Neuron(neuron_id=i, params=sol[i]) for i in range(sum(config["nodes"]))]
        teams = create_teams(pop, n_shuffle=10, team_size=sum(config["nodes"]))
        
        # Track neuron descriptors and objectives
        objectives = defaultdict(list)
        descriptors = defaultdict(list)
        # Track iteration statistics
        iteration_fitnesses = []
        iteration_means = []

        for team in teams:
            net = NCHL(config["nodes"], population=team)
            eval_results = evaluate_team(net, n_episodes=config["episodes"])
            # Use original metric for evolution
            fitness = eval_results['percentile_70']
            mean_reward = eval_results['mean_reward']

            iteration_fitnesses.append(fitness)
            iteration_means.append(mean_reward)
            history.append(fitness)
            
            for neuron in net.all_neurons:
                entropy, weight_change = neuron.compute_descriptors()
                descriptors[neuron.neuron_id].append([entropy, weight_change])
                objectives[neuron.neuron_id].append(fitness)
                
            # Track all nets
            nets['fitness'].append(fitness)
            nets['network'].append(net)
            
        # Track best fitness and mean reward of each iteration
        best_fitness_iteration = max(iteration_fitnesses)
        mean_reward_iteration = np.mean(iteration_means)
        history_best.append(best_fitness_iteration)
        history_mean.append(mean_reward_iteration)

        # Update best fitness overall
        best_fitness = max(best_fitness, max(iteration_fitnesses))

        if i % 10 == 0:
            print(f"\nIteration {i}")
            print(f"Average Fitness This Iteration: {np.mean(iteration_fitnesses):.2f}")
            print(f"Best Fitness This Iteration: {best_fitness_iteration:.2f}")
            print(f"Mean Reward This Iteration: {mean_reward_iteration:.2f}")
            print(f"Average Fitness Overall: {np.mean(history):.2f}")
            print(f"Best Fitness Overall: {best_fitness:.2f}")
            print("\nArchive stats:")
            print(archive.stats)

        # Aggregate descriptors and objectives for each neuron
        aggregate_descriptors = []
        aggregate_objectives = []
        for neuron_id, desc in descriptors.items():
            avg_entropy = np.mean([x[0] for x in desc])
            avg_weight_change = np.mean([x[1] for x in desc])
            aggregate_descriptors.append([avg_entropy, avg_weight_change])
            aggregate_objectives.append(np.mean(objectives[neuron_id]))

        # Tell the scheduler the fitness of the solution
        scheduler.tell(aggregate_objectives, aggregate_descriptors)

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Best fitness achieved: {best_fitness:.2f}")
    print(f"Final average fitness: {np.mean(history):.2f}")
    print("\nArchive stats:")
    print(archive.stats)

    if config["log"]:
        # Log data to file
        output_log = f'{config["output_log"]}/cartpole_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
        os.makedirs(config["output_log"], exist_ok=True)
        log_data(output_log, best_fitness, history, archive, config)
    
    if config["plot"]:
        output_plot = f'{config["output_plot"]}/cartpole_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        os.makedirs(output_plot, exist_ok=True)
        
        plot_archive_heatmap(archive, output_plot)                   # Plot archive heatmap
        plot_fitness_history(history, output_plot)                   # Plot fitness history
        plot_fitness_history(history_best, output_plot, best=True)   # Plot best fitness history

    # Sort the dictionary by fitness
    sorted_nets = sorted(zip(nets['fitness'], nets['network']), key=lambda x: x[0], reverse=True)
    # save the best 5 networks
    for i, (fitness, net) in enumerate(sorted_nets[:5]):
        visualize_network(net, save=True, fitness=fitness, file_name=f'temp/net_{i}.png')
    

if __name__ == '__main__':
    main()