import numpy as np
import functools
from random import Random
from cma import CMAEvolutionStrategy as cmaes
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm

# Import the new network module
from network import NCHL, Neuron

# Utility wrapper for the CMAES optimizer
def generator(random, args):
    return np.asarray([random.uniform(args["pop_init_range"][0],
                                      args["pop_init_range"][1])
                       for _ in range(args["num_vars"])])

def generator_wrapper(func):
    @functools.wraps(func)
    def _generator(random, args):
        return np.asarray(func(random, args))
    return _generator

class Optimizer():
    def __init__(self, num_vars, seed=1, pop_init_range=[-1,1], lmbda=20, mu=10, sigma=1.):
        args = {
            "num_vars": num_vars,
            "pop_init_range": pop_init_range,
        }
        self.cmaes = cmaes(generator(Random(seed), args),
                           sigma,
                           {'popsize': lmbda,
                            'seed': seed,
                            'CMA_mu': mu})
        self.pop = []
        self.best = None

    def tell(self, fitness):
        self.cmaes.tell(self.pop, fitness)
        self.best = self.cmaes.best

    def ask(self):
        self.pop = self.cmaes.ask()
        return self.pop[:]

# Updated evaluation function to use the new NCHL implementation
def eval(params):
    task = gym.make("CartPole-v1")
    nodes = [4, 10, 2]
    agent = NCHL(nodes, device="cpu")
    agent.set_params(params)
    
    rews = []
    for i in range(10):  # Run 100 episodes
        obs, info = task.reset()  # Reset for each episode
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            input_tensor = torch.tensor(obs, dtype=torch.float64)
            output = agent.forward(input_tensor)
            agent.update_weights()
            
            action = np.argmax(output.cpu().detach().numpy()[0])
            obs, rew, done, truncated, info = task.step(action)
            episode_reward += rew
            
        rews.append(episode_reward)
            
    return -np.mean(rews)

# Main optimization and training
if __name__ == "__main__":
    # Create a temporary NCHL instance to determine the number of parameters
    temp_network = NCHL([4, 10, 2])
    number_of_parameters = temp_network.nparams
    print(f"Number of parameters to optimize: {number_of_parameters}")

    # Initialize optimizer
    opt = Optimizer(number_of_parameters)

    best_sol_fitness = []

    # Iterate for some generations
    num_generations = 20
    for i in tqdm(range(num_generations)):
        # Get candidate solutions
        inds = opt.ask()
        # Evaluate each candidate
        fits = [eval(ind) for ind in inds]
        # Update optimizer with fitness values
        opt.tell(fits)
        # Track best solution's fitness
        best_sol_fitness.append(-np.min(fits))  # Convert back to positive reward
        
        # Print progress information
        if i % 5 == 0 or i == num_generations - 1:
            print(f"Generation {i}, Best fitness: {-np.min(fits)}")

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(best_sol_fitness)
    plt.title('Learning Curve')
    plt.xlabel('Generation')
    plt.ylabel('Best Reward')
    plt.grid(True)
    plt.show()

    # Analyze the best solution
    best_params = opt.best.x
    print(f"Best solution fitness: {-opt.best.f}")
    
    # Test the best solution
    test_agent = NCHL([4, 10, 2])
    test_agent.set_params(best_params)
    
    # Run a visualization episode
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        input_tensor = torch.tensor(obs, dtype=torch.float64)
        output = test_agent.forward(input_tensor)
        test_agent.update_weights()
        
        action = np.argmax(output.cpu().detach().numpy()[0])
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Test episode reward: {total_reward}")
    env.close()
    
    # Optional: Analyze neuron behavior
    print("\nNeuron Analysis:")
    for layer_idx, layer in enumerate(test_agent.neurons):
        print(f"Layer {layer_idx}:")
        for neuron_idx, neuron in enumerate(layer):
            entropy, weight_change = neuron.compute_descriptors()
            print(f"  Neuron {neuron_idx}: Activation Entropy={entropy:.4f}, Avg Weight Change={weight_change:.6f}")