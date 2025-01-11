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

'''
Steps:
    1. generate a initial population of n (100) neurons with random parameters (5)
    2. generate 10 shuffle of the initial population (10 x [100 x 5]) and, for each shuffle, divide in 10 groups of 10 neurons, creating 10 teams (networks) of 10 neurons
        - so each neuron will be in 10 teams
    3. evaluation: for each team, create the NCHL network with its neurons
        1. evaluate the network with CartPole problem and calculate the fitness (reward)
        2. for each neuron, store its 2 descriptors (entropy and average weight changes)
    4. aggregation: for each neuron, calculate the average of its descriptors among all teams
        - average of entropy
        - average of average weight changes
        - 70th percentile of fitness
    5. based on the information, update the pyribs archive 
'''

def generate_teams(population, n_neurons_per_team=10, n_shuffles=10):
    all_teams = []
    
    for _ in range(n_shuffles):
        shuffled_pop = population.copy() # shuffle the population
        random.shuffle(shuffled_pop)
        # divide the population in teams
        n_teams = len(shuffled_pop) // n_neurons_per_team
        for i in range(n_teams):
            start_idx = i * n_neurons_per_team
            end_idx = start_idx + n_neurons_per_team
            team = shuffled_pop[start_idx:end_idx]
            all_teams.append(team)
    
    return all_teams
        
def evaluate_team(network, n_episodes=5, render=False):
    """ 
    Evaluate the network with CartPole problem and calculate the fitness (reward)
    """
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    test_rewards = []
    
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
            
            network.update_weights() # update the weights
        
        test_rewards.append(episode_reward)
    
    env.close()
    return np.mean(test_rewards)

    
class Optimizer:
    def __init__(self, nodes):
        self.nodes = nodes
        # create the archive
        self.archive = GridArchive(
            solution_dim=5,     # 5 parameters for each neuron
            dims=[20, 20],
            ranges=[(0.0, 3.5), (0.0, 2.0)],
        )
        
        # create the emitter
        self.emitter = EvolutionStrategyEmitter(
            archive=self.archive,
            sigma0=0.2,          
            batch_size=100,
            x0=np.zeros(5)
        )
                
        # create the scheduler
        self.scheduler = Scheduler(emitters=[self.emitter], archive=self.archive)
    
    def run_evolution(self):
        # Ask new solutions from the emitter 
        solutions = self.scheduler.ask()
        
        # generate population
        pop = []
        for i, sol in enumerate(solutions):
            neuron = Neuron(neuron_id=i)
            neuron.set_hebbian_params(sol[0], sol[1], sol[2], sol[3])
            neuron.set_eta(sol[4])
            pop.append(neuron)
            
        # generate teams
        teams = generate_teams(pop)
        
        data = defaultdict(list)
        
        # evaluate teams
        for team in teams:
            network = NCHL(nodes=self.nodes, population=team)
            fitness = evaluate_team(network)
            
            # store descriptors of each neuron in the network
            for neuron in network.all_neurons:
                d1, d2 = neuron.compute_descriptors()
                data[neuron.id].append((d1, d2, fitness))
                
        if DEBUG:
            # print descriptors
            for neuron, descriptors in data.items():
                print(f'Neuron {neuron} - Min and Max Entropy: {min([d[0] for d in descriptors]), max([d[0] for d in descriptors])} - Min and Max Weight Changes: {min([d[1] for d in descriptors]), max([d[1] for d in descriptors])}')
        
        # aggregate data
        aggregated_data = defaultdict(list)
        for neuron, descriptors in data.items():
            avg_entropy = np.mean([d[0] for d in descriptors])
            avg_weight_changes = np.mean([d[1] for d in descriptors])
            avg_fitness = np.mean([d[2] for d in descriptors])
            pct_fitness = np.percentile([d[2] for d in descriptors], 70) # 70th percentile of fitness
                        
            aggregated_data[neuron] = (avg_entropy, avg_weight_changes, avg_fitness)
    
        # update the archive
        objectives = []
        descriptors = []
        
        for idx, sol in enumerate(solutions):
            fit = aggregated_data[idx][2] # fitness
            desc = [aggregated_data[idx][0], aggregated_data[idx][1]] # descriptors
            objectives.append(fit)
            descriptors.append(desc)
            
        self.scheduler.tell(objectives, descriptors)
        
        return self.archive.stats.obj_max
        
        
DEBUG = True
        
if __name__ == '__main__':
    optimizer = Optimizer([4, 4, 2])
    history = []
    for i in tqdm(range(10)):
        obj_max = optimizer.run_evolution()
        if DEBUG: print(f'Itr {i} - Max fitness: {obj_max}')
        history.append(obj_max)
        
    print(history)
        
    grid_archive_heatmap(optimizer.archive) # plot the archive heatmap
    plt.show()
    
    # plot the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('Training History')
    plt.xlabel('Iteration')
    plt.ylabel('Max Fitness')
    plt.show()