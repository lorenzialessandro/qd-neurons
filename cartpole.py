import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from collections import deque
import gymnasium as gym
from ribs.visualize import grid_archive_heatmap
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter
from multiprocessing import Pool

from network import NCHL, Neuron


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
    return {
        # 70th percentile of episode rewards
        'percentile_70': np.percentile(rewards, 70),
        'mean_reward': total_reward / n_episodes,     # Standard solving criterion
        'raw_rewards': rewards                        # Raw episode rewards
    }


SEED = 2
NODES = [4, 8, 2]

# archive : store individual neuron solutions
archive = GridArchive(
    solution_dim=5,  # 5 parameters for each neuron
    dims=[10, 10],
    ranges=[(1.5, 3), (0, 1)],  # average entropy and weight change
    seed=SEED,
)

emitter = EvolutionStrategyEmitter(
    archive=archive,
    x0=np.random.uniform(-1, 1, 5),  # initial solution randomly generated
    sigma0=0.2,  # initial standard deviation
    batch_size=sum(NODES),  # number of neurons
    seed=SEED,
)

scheduler = Scheduler(emitters=[emitter], archive=archive)


# Initialize tracking variables
best_fitness = -float('inf')
history = []
history_best = []
history_mean = []  # Track mean rewards separately

for i in tqdm(range(500)):
    sol = scheduler.ask()
    pop = [Neuron(neuron_id=i, params=sol[i]) for i in range(sum(NODES))]
    teams = create_teams(pop, n_shuffle=60, team_size=sum(NODES))

    # Track neuron descriptors and objectives
    objectives = defaultdict(list)
    descriptors = defaultdict(list)
    # Track iteration statistics
    iteration_fitnesses = []
    iteration_means = []

    for team in teams:
        net = NCHL(NODES, population=team)
        eval_results = evaluate_team(net)
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
plt.savefig('results/cart_pole_archive.png')

# Plot 2: Fitness History
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.title('Fitness History')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('results/cart_pole_fitness_history.png')

# Plot 3: Best Fitness History
plt.figure(figsize=(10, 6))
plt.plot(history_best)
plt.title('Best Fitness History')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.tight_layout()
plt.savefig('results/cart_pole_best_fitness_history.png')