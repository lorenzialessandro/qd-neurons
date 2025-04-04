import numpy as np
import torch
import random
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import gymnasium as gym

from network import NCHL, Neuron

class NeuronArchive:
    def __init__(self, dims, sigma=0.1, seed=None):
        self.dims = dims                # Dimensions of the archive
        self.n_dims = len(dims)         # Number of dimensions
        self.solution_dim = 5           # Hebbian rule : (A, B, C, D, eta)
        self.sigma = sigma              # Standard deviation for mutation
        self.ranges = [(0, 1), (-1, 1)]  # Ranges for each dimension
        
        random.seed(seed) 
        np.random.seed(seed)
        
        self.archive = {}       
    
    def empty(self):
        return len(self.archive) == 0
    
    def get_random_rule(self):
        # Generate a random Hebbian rule
        rule = [random.uniform(-1, 1) for _ in range(self.solution_dim)]
        return rule
    
    def _mutate_rule(self, rule):
        # Mutate the rule using a Gaussian distribution
        mutated_rule = []
        for x in rule:
            # Apply Gaussian mutation
            mutated_value = x + random.gauss(0, self.sigma)
            # Ensure the mutated value is within the range [-1, 1]
            mutated_value = max(-1, min(mutated_value, 1))
            mutated_rule.append(mutated_value)
            
        return mutated_rule 
        
    def _valid_solution(self, solution):
        if len(solution) != 2:
            return False
        if len(solution["rule"]) != self.solution_dim:
            return False
        return True
    
    def _valid_pos(self, pos):
        if len(pos) != self.n_dims:
            return False
        for i in range(self.n_dims):
            if pos[i] < 0 or pos[i] >= self.dims[i]:
                return False
        return True
    
    def _add_solution(self, pos, solution):
        if not self._valid_solution(solution):
            raise ValueError("Solution dimension mismatch")
        if not self._valid_pos(pos):
            raise ValueError("Invalid position")
        
        if pos not in self.archive:
            # If the cell is empty, add the solution
            self.archive[pos] = solution
        else:
            if self.archive[pos]["fitness"] < solution["fitness"]:
                # If the cell is occupied, replace the solution if the new one is better
                self.archive[pos] = solution
            
    def ask_rule(self):
        # Select a random solution from the archive
        if self.empty():
            return self.get_random_rule()
        
        # Select a random position from the archive
        pos = random.choice(list(self.archive.keys()))
        rule = self.archive[pos]["rule"]
        # Mutate the solution
        offspring = self._mutate_rule(rule)
        
        return offspring
    
    def _discretize_behavior(self, behavior):
        """
        Convert continuous behavior values to discrete grid indices
        based on the defined ranges for each dimension.
        """
        if len(behavior) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} behavior values, got {len(behavior)}")
        
        discretized = []
        
        for i, value in enumerate(behavior):
            min_val, max_val = self.ranges[i]
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            # Clamp to [0, 1]
            normalized = max(0, min(normalized, 1))
            # Convert to grid index [0, dims[i]-1]
            index = min(int(normalized * self.dims[i]), self.dims[i] - 1)
            discretized.append(index)
        
        return tuple(discretized)  # Convert to tuple to be hashable
    
    def tell(self, behavior, fitness, rule):
        self._add_solution(
            pos=self._discretize_behavior(behavior), 
            solution={
                "rule": rule,
                "fitness": fitness
            }
        )

# -----------------------------------------------------
# -----------------------------------------------------    

def create_nets(pop, config):
    nets = []
    for _ in range(config["n_teams"]):
        team_neurons = pop.copy()
        random.shuffle(team_neurons)
        team_neurons = team_neurons[:len(pop)]  # Keep same number of neurons
        net = NCHL(nodes=config["nodes"], population=team_neurons)
        nets.append(net)
    return nets

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

def main():
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)
    config = {
        "seed": 5,
        "nodes": [4, 4, 2],  # Input, hidden, output layers
        "iterations": 100,
        "threshold": 475,
        "episodes": 10,
        "n_teams": 10
    }
    n_nodes = sum(config["nodes"])
    
    # Create initial population of neurons (randomly initialized)
    pop = [Neuron(neuron_id=i) for i in range(n_nodes)]

    # Create archives for each neuron
    archives = {
        neuron.neuron_id : NeuronArchive(
            dims=[10, 10], 
            sigma=0.1,
            seed=config["seed"] + neuron.neuron_id
        ) for i, neuron in enumerate(pop)
    }
    
    # Loop
    for iteration in range(5):
        for neuron in pop:
            archive = archives[neuron.neuron_id]
            # Ask
            rule = archive.ask_rule() 
            neuron.set_params(rule) 
            
        # Create networks teams from the population
        networks = create_nets(pop, config) 
        # Evaluate networks in parallel
        network_configs = [(net, config.get("episodes", 5)) for net in networks]
        evaluation_results = pool.map(evaluate_team_parallel, network_configs)
        
        # Process results
        objectives = defaultdict(list)
        descriptors = defaultdict(list)
        
        all_fitness = []
        for net, result in evaluation_results:
            fitness = result['mean_reward']
            all_fitness.append(fitness)
            
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
                
            # Use 70th percentile fitness
            combined_fitness = np.percentile(neuron_fitness, 70)
            
            # Average descriptors
            avg_behavior = np.mean([d[0] for d in descriptors[neuron.neuron_id]])
            avg_complexity = np.mean([d[1] for d in descriptors[neuron.neuron_id]])
            
            # Update archive
            archives[neuron.neuron_id].tell(
                behavior=[avg_behavior, avg_complexity], 
                fitness=combined_fitness, 
                rule=neuron.params #TODO: add get_params() function in network.py
            )

        print(f"Iteration {iteration + 1} completed.")
        
        
if __name__ == "__main__":
    main()