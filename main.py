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

def evaluate_team(network, n_episodes=10):
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
    return np.percentile(rewards, 70) # 70th percentile of rewards
    # return total_reward / n_episodes # average reward

SEED = 1234
NODES = [4, 4, 2]
    
# archive : store individual neuron solutions
archive = GridArchive(
    solution_dim=5, # 5 parameters for each neuron
    dims = [10, 10],
    ranges = [(1.5, 3), (0, 0.5)], # average entropy and weight change
    seed = SEED,
)  

emitter = EvolutionStrategyEmitter(
    archive = archive,
    x0 = np.random.uniform(-1, 1, 5), # initial solution randomly generated
    sigma0 = 0.1, # initial standard deviation
    batch_size = sum(NODES), # number of neurons
    seed = SEED,
)

scheduler = Scheduler(emitters = [emitter], archive = archive)


best_fitness = -float('inf')
history = []

for i in tqdm(range(200)):  
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
    
    # Track best fitness
    best_fitness = max(best_fitness, max(iteration_fitnesses))
    
    if i % 10 == 0:
        print(f"\nIteration {i}")
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
        # aggregate_objectives.append(np.percentile(objectives[neuron_id], 70)) # 70th percentile of fitness instead of mean
    
    scheduler.tell(aggregate_objectives, aggregate_descriptors)


# Plotting

# Plot 1: Archive Heatmap
grid_archive_heatmap(archive)
plt.title('Archive Heatmap')
plt.savefig('archive.png')

# Plot 3: Fitness History
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.title('Fitness History')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('fitness_history.png')

# Print final statistics
print("\nFinal Statistics:")
print(f"Best fitness achieved: {best_fitness:.2f}")
print(f"Final average fitness: {np.mean(history):.2f}")
print("\nArchive stats:")
print(archive.stats)