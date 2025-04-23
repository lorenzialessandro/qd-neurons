import numpy as np
import torch
import random
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import gymnasium as gym

import matplotlib.pyplot as plt

from network import NCHL, Neuron

class NeuronArchive:
    """Archive for storing and managing Hebbian rules."""
    def __init__(self, dims, ranges, sigma = 0.1, seed=None):
        """
        Initialize the Neuron Archive.
        
        :param dims: List of dimensions for the archive.
        :param ranges: List of ranges for each dimension.
        :param sigma: Standard deviation for Gaussian mutation.
        :param seed: Random seed for reproducibility.
        """
        
        self.dims = dims                # Dimensions of the archive
        self.n_dims = len(dims)         # Number of dimensions
        self.solution_dim = 5           # Hebbian rule : (A, B, C, D, eta)
        self.sigma = sigma              # Standard deviation for mutation
        self.ranges = ranges            # Ranges for each dimension
        
        random.seed(seed) 
        np.random.seed(seed)
        
        self.archive = {}     
    
    def get_random_rule(self):
        """Generate a random Hebbian rule."""
        rule = [random.uniform(-1, 1) for _ in range(self.solution_dim - 1)] # A, B, C, D in [-1, 1]
        rule.append(random.uniform(0, 1))  # eta is in [0, 1]
        return rule
    
    def _mutate_rule(self, rule):
        """Mutate a given rule using Gaussian mutation."""
        mutated_rule = []
        
        # The first 4 values are Hebbian parameters
        for i in range(len(rule) - 1):
            # Apply Gaussian mutation
            mutated_value = rule[i] + random.gauss(0, self.sigma)
            # Ensure the mutated value is within the range [-1, 1]
            mutated_value = max(-1, min(mutated_value, 1))
            mutated_rule.append(mutated_value)
        
        # The last value is the learning rate (eta)
        eta = rule[-1] + random.gauss(0, self.sigma * 0.5)  # Use smaller sigma for eta
        eta = max(0, min(eta, 1))  # Different bounds for learning rate
        mutated_rule.append(eta)
                
        return mutated_rule
        
    def _valid_solution(self, solution):
        """Check if the solution is valid."""
        if not isinstance(solution, dict):
            return False
        if "rule" not in solution or "fitness" not in solution:
            return False
        if len(solution["rule"]) != self.solution_dim:
            return False
        return True
    
    def _valid_pos(self, pos):
        """Check if the position is valid."""
        if len(pos) != self.n_dims:
            return False
        for i in range(self.n_dims):
            if pos[i] < 0 or pos[i] >= self.dims[i]:
                return False
        return True
    
    def _add_solution(self, pos, solution):
        """Add a solution to the archive."""
        
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
            
    def ask(self):
        """
        Ask the archive for a new solution:
        - Randomly selects a solution from the archive and mutates it.
        - Returns the mutated solution.
        """
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
        """Discretize continuous behavior values to discrete grid indices based on the defined ranges for each dimension."""
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
        """
        Store and update the archive with a new solution:
        - Discretizes the behavior to find the corresponding grid cell.
        - If the cell is empty, adds the solution.
        - If the cell is occupied, replaces the solution if the new one is better.
        """
        self._add_solution(
            pos=self._discretize_behavior(behavior), 
            solution={
                "rule": rule,
                "fitness": fitness,
                "behavior": behavior
            }
        )
        
    def empty(self):
        """Check if the archive is empty."""
        return len(self.archive) == 0
    
    def data(self):
        """Retrieve the data from the archive."""
        return self.archive
    
    def coverage(self):
        """Calculate the coverage of the archive."""
        tot_cells = np.prod(self.dims)
        covered_cells = len(self.archive)
        return covered_cells / tot_cells 
        
    def visualize(self, ax=None, cmap='magma', transpose_measures=False, 
              aspect='auto', vmin=None, vmax=None, cbar=True, 
              cbar_kwargs=None, rasterized=False, pcm_kwargs=None):
        
        """Visualize the archive using a heatmap."""
        
        if ax is None:
            ax = plt.gca()
        
        # Retrieve data from archive
        lower_bounds = [self.ranges[0][0], self.ranges[1][0]]
        upper_bounds = [self.ranges[0][1], self.ranges[1][1]]
        x_dim, y_dim = self.dims
        
        # Create boundary arrays
        x_bounds = np.linspace(lower_bounds[0], upper_bounds[0], x_dim + 1)
        y_bounds = np.linspace(lower_bounds[1], upper_bounds[1], y_dim + 1)
        
        # Color for each cell in the heatmap
        colors = np.full((y_dim, x_dim), np.nan)
        
        # Extract objective values and positions
        if not self.empty():
            positions = list(self.archive.keys())
            objectives = [self.archive[pos]["fitness"] for pos in positions]
            
            # Fill the colors array
            for pos, obj in zip(positions, objectives):
                x_idx, y_idx = pos
                colors[y_idx, x_idx] = obj
        
        if transpose_measures:
            # Transpose by swapping the x and y boundaries and flipping the bounds
            x_bounds, y_bounds = y_bounds, x_bounds
            lower_bounds = np.flip(lower_bounds)
            upper_bounds = np.flip(upper_bounds)
            colors = colors.T
        
        # Set axis limits
        ax.set_xlim(lower_bounds[0], upper_bounds[0])
        ax.set_ylim(lower_bounds[1], upper_bounds[1])
        ax.set_aspect(aspect)
        
        # Set labels
        ax.set_xlabel(f"Std Activations [{lower_bounds[0]}, {upper_bounds[0]}]")
        ax.set_ylabel(f"Learning Rate [{lower_bounds[1]}, {upper_bounds[1]}]")
        
        # Create the plot
        if pcm_kwargs is None:
            pcm_kwargs = {}
        
        if vmin is None and not np.isnan(colors).all():
            vmin = np.nanmin(colors)
        
        if vmax is None and not np.isnan(colors).all():
            vmax = np.nanmax(colors)
        
        t = ax.pcolormesh(x_bounds, y_bounds, colors, cmap=cmap, 
                        vmin=vmin, vmax=vmax, rasterized=rasterized, 
                        **pcm_kwargs)
        
        # Create color bar
        if cbar:
            if cbar_kwargs is None:
                cbar_kwargs = {}
            cbar = plt.colorbar(t, ax=ax, **cbar_kwargs)
            # cbar.set_label('Fitness')
        
        return ax
  