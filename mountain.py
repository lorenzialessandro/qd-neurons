import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import gymnasium as gym
import importlib
import decorator
from ribs.visualize import grid_archive_heatmap
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter

from network import NCHL, Neuron

def create_initial_population(pop_size=100):
    pop = [Neuron(neuron_id = i) for i in range(pop_size)]
    return pop

def create_teams(initial_pop, n_shuffle=10, team_size=10):
    teams = []
    for i in range(n_shuffle):
        pop = initial_pop.copy()
        random.shuffle(pop)
        
        for j in range(0, len(pop), team_size):
            teams.append(pop[j:j+team_size])
            
    return teams

def evaluate_team(network, n_episodes=100):
    env = gym.make('MountainCar-v0')
    total_reward = 0
    rewards = []
    positions = []  # Track maximum positions reached
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        max_position = -1.2  # Starting position
        
        while not (done or truncated):
            input = torch.tensor(state).double()
            output = network.forward(input)
            action = np.argmax(output.tolist())
            
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            max_position = max(max_position, state[0])  # Track highest position
            
            network.update_weights()
        
        # Modify reward based on maximum position reached
        position_reward = 100 * (max_position + 1.2) / (0.6 + 1.2)  # Normalize position progress
        modified_reward = episode_reward + position_reward
        
        total_reward += modified_reward
        rewards.append(modified_reward)
        positions.append(max_position)
    
    env.close()
    return np.mean(rewards)  # average reward
    return np.percentile(rewards, 70)  # 70th percentile of rewards

SEED = 145
NODES = [2, 4, 3]  # MountainCar (2 inputs, 3 outputs)
    
# archive : store individual neuron solutions
archive = GridArchive(
    solution_dim=5,  # 5 parameters for each neuron
    dims = [10, 10],
    ranges = [(0, 3), (0, 3)],  # average entropy and weight change
    seed = SEED,
)  

emitter = EvolutionStrategyEmitter(
    archive = archive,
    x0 = np.random.uniform(-1, 1, 5),  # initial solution randomly generated
    sigma0 = 0.3,  # 
    batch_size = sum(NODES),  # number of neurons
    seed = SEED,
)

scheduler = Scheduler(emitters = [emitter], archive = archive)

best_fitness = -float('inf')
history = []
history_best = []
max_positions = []  # Track best positions reached

for i in tqdm(range(1000)):  
    sol = scheduler.ask()
    pop = [Neuron(neuron_id=i, params=sol[i]) for i in range(sum(NODES))]
    teams = create_teams(pop)
    
    objectives = defaultdict(list)
    descriptors = defaultdict(list)
    
    # Track iteration statistics
    iteration_fitnesses = []
    
    for team in teams:
        net = NCHL(NODES, population=team)
        fitness = evaluate_team(net)
        iteration_fitnesses.append(fitness)

        history.append(fitness)
        
        for neuron in net.all_neurons:
            entropy, weight_change = neuron.compute_descriptors()
            descriptors[neuron.neuron_id].append([entropy, weight_change])
            objectives[neuron.neuron_id].append(fitness)
    
    # Track best fitness of each iteration
    best_fitness_iteration = max(iteration_fitnesses)
    history_best.append(best_fitness_iteration)
    
    # Track best fitness overall
    best_fitness = max(best_fitness, max(iteration_fitnesses))
    
    if i % 10 == 0:
        print(f"\nIteration {i}")
        print(f"Average Fitness This Iteration: {np.mean(iteration_fitnesses):.2f}")
        print(f"Best Fitness This Iteration: {best_fitness_iteration:.2f}")
        print(f"Average Fitness Overall: {np.mean(history):.2f}")
        print(f"Best Fitness Overall: {best_fitness:.2f}")
        print("\nArchive stats:")
        print(archive.stats)
    
    # Aggregate descriptors and objectives for each neuron
    aggregate_descriptors = []
    aggregate_objectives = []
    for neuron_id, desc in descriptors.items():
        # descriptors 
        avg_entropy = np.mean([x[0] for x in desc])
        avg_weight_change = np.mean([x[1] for x in desc])
        aggregate_descriptors.append([avg_entropy, avg_weight_change])
        # objectives
        aggregate_objectives.append(np.mean(objectives[neuron_id]))
    
    scheduler.tell(aggregate_objectives, aggregate_descriptors)

# Print final statistics
print("\nFinal Statistics:")
print(f"Best fitness achieved: {best_fitness:.2f}")
print(f"Final average fitness: {np.mean(history):.2f}")
print("\nArchive stats:")
print(archive.stats)

# Plotting

# Plot 1: Archive Heatmap
grid_archive_heatmap(archive, cmap='Greens')
plt.title('Archive Heatmap')
plt.xlabel('Average Entropy')
plt.ylabel('Average Weight Change')
plt.savefig('results/mountain_car_archive.png')

# Plot 2: Fitness History
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.title('Fitness History (MountainCar)')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('results/mountain_car_fitness_history.png')

# Plot 3: Best Fitness History
plt.figure(figsize=(10, 6))
plt.plot(history_best)
plt.title('Best Fitness History (MountainCar)')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('results/mountain_car_best_fitness_history.png')