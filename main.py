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

def generate_teams(population, n_neurons_per_team=10, n_teams=10):
    """Generate teams ensuring each neuron appears in exactly n_teams teams"""
    n_total = len(population)
    teams = [[] for _ in range(n_teams * (n_total // n_neurons_per_team))]
    
    for neuron_idx, neuron in enumerate(population):
        # Calculate which teams this neuron should be in
        team_indices = [(neuron_idx + i) % len(teams) for i in range(n_teams)]
        for team_idx in team_indices:
            teams[team_idx].append(neuron)
            
    # Filter out incomplete teams
    return [team for team in teams if len(team) == n_neurons_per_team]
        
        
def evaluate_team(agent, n_steps=1):
    """ Evaluate the network with CartPole problem and calculate the fitness (reward) """
    env = gym.make('CartPole-v1')
    
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    for _ in range(n_steps):
        while not (done or truncated):
            input = torch.tensor(state).double()
            
            output = agent.forward(input)
            
            action = np.clip(np.argmax(output.tolist()), 0, env.action_space.n - 1)
            state, rew, done, truncated,  _ = env.step(action)
            
            total_reward += rew
            
            agent.update_weights()
                
    return total_reward

    
class Optimizer:
    def __init__(self, nodes):
        self.nodes = nodes
        # create the archive
        self.archive = GridArchive(
            solution_dim=5,     # 5 parameters for each neuron
            dims=[20, 20],
            ranges=[(-3, 3),    
                (-1, 1)] 
        )
        
        # create the emitter
        self.emitter = GaussianEmitter(
            archive=self.archive,
            sigma=0.3,
            x0=np.zeros(5),
            batch_size=sum(self.nodes) # as population size
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
            fitness = evaluate_team(network, n_steps=20)
            
            # store descriptors of each neuron in the network
            for neuron in network.all_neurons:
                d1, d2 = neuron.compute_descriptors()
                data[neuron.id].append((d1, d2, fitness))
        
        # aggregate data
        aggregated_data = defaultdict(list)
        for neuron, descriptors in data.items():
            avg_entropy = np.mean([d[0] for d in descriptors])
            avg_weight_changes = np.mean([d[1] for d in descriptors])
            avg_fitness = np.mean([d[2] for d in descriptors])
            fit_percentile = np.percentile([d[2] for d in descriptors], 70) # 70th percentile of fitness
                        
            aggregated_data[neuron] = (avg_entropy, avg_weight_changes, fit_percentile)
    
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
        
if __name__ == '__main__':
    optimizer = Optimizer([4, 4, 2])
    history = []
    for i in tqdm(range(100)):
        obj_max = optimizer.run_evolution()
        # print(f'Itr {i} - Max fitness: {obj_max}')
        history.append(obj_max)
        
    print(history)
        
    # Plot heatmap for archive
    plt.figure(figsize=(10, 6))
    grid_archive_heatmap(optimizer.archive, cmap="viridis")
    plt.title("Archive Heatmap")
    plt.show()
    
    # plot the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('Training History')
    plt.xlabel('Iteration')
    plt.ylabel('Max Fitness')
    plt.show()
    
    