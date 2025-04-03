import numpy as np
import torch
import torch.nn as nn

class Neuron:
    def __init__(self, neuron_id: int, params=None, device="cpu"):
        self.neuron_id = neuron_id
        self.device = device     
    
        # Hebbian rule parameters initialized to zero
        self.pre_factor = torch.tensor(0.0, device=device)
        self.post_factor = torch.tensor(0.0, device=device)
        self.correlation = torch.tensor(0.0, device=device)
        self.decorrelation = torch.tensor(0.0, device=device)
        self.eta = torch.tensor(0.0, device=device) 
        self.params = []
    
        # Current activation value of the neuron
        self.activation = torch.tensor(0.0, device=device)
        
        # Store activations and weight changes for the neuron for descriptors
        self.activations = []
        self.weight_changes = []
        
        if params is not None:
            self.set_params(params)

    def add_activation(self, activation):
        """Add an activation to the list of activations."""
        self.activations.append(activation.item())
        
    def add_weight_change(self, weight_change):
        """Add a weight change to the list of weight changes."""
        self.weight_changes.append(weight_change)
    
    def set_params(self, params: list):
        """Set the Hebbian learning parameters and learning rate for this neuron."""
        self.params = params
        self.set_hebbian_params(params[0], params[1], params[2], params[3])
        self.set_eta(params[4])
    
    def set_hebbian_params(self, pre: float, post: float, corr: float, decorr: float):
        """Set the Hebbian learning parameters for this neuron."""
        self.pre_factor = torch.tensor(pre, device=self.device)
        self.post_factor = torch.tensor(post, device=self.device)
        self.correlation = torch.tensor(corr, device=self.device)
        self.decorrelation = torch.tensor(decorr, device=self.device)

    def set_eta(self, eta: float):
        """Set the learning rate for this neuron."""
        self.eta = torch.tensor(eta, device=self.device)

    def set_activation(self, activation):
        """Set the current activation value of the neuron."""
        self.activation = activation.to(self.device)
        self.add_activation(activation)

    # def get_hebbian_terms(self):
    #     """Get the Hebbian terms for weight updates."""
    #     return (
    #         (self.pre_factor * self.activation).to(self.device),
    #         (self.post_factor * self.activation).to(self.device),
    #         1. if self.correlation == 1. else (self.correlation * self.activation).to(self.device),
    #         self.decorrelation.to(self.device)
    #     )
    
    def get_hebbian_terms(self):
        """Get the Hebbian terms for weight updates."""
        return (
            (self.pre_factor * self.activation).to(self.device),
            (self.post_factor * self.activation).to(self.device),
            torch.tensor(1.0, device=self.device) if self.correlation == 1. else (self.correlation * self.activation).to(self.device),
            self.decorrelation.to(self.device)
        )
    
    def compute_behavioral_variability(self):
        """
        Measures how much variation exists in a series of activations over different time segments.
        
        This function measures how activation patterns change over time by analyzing statistical variation between chunks of activation data
        - Higher values indicate the neuron behaves differently across different chunks of data
        - Lower values indicate the neuron behaves similarly across different chunks of data : consistent behavior
        """
       
        if not self.activations:
            return 0.0
        if len(self.activations) < 10: # not enough activations to compute variability
            return 0.5

        # Option 1: Compute standard deviation of activations
        activations_tensor = torch.tensor(self.activations, device=self.device)
        std = torch.std(activations_tensor).item()
        # return std

        # Option 2: 
        n_chunks = min(10, len(self.activations) // 10) 
        chunks = torch.chunk(activations_tensor, n_chunks) 
        # mean of each chunk
        means = torch.stack([torch.mean(chunk) for chunk in chunks])
        # variance of means
        chunk_var = torch.var(means).item()
        # normalize between 0 and 1
        norm_chunk_var = min(1.0, chunk_var / 0.5)
        return norm_chunk_var
    
        
    def compute_new_complexity(self):
        """
        Measures the complexity of the activation sequence using a frequency-based analysis (Fourier Transform).
        
        This function analyzes the frequency components of activation patterns using Fast Fourier Transform (FFT) to measure how complex the activation signal is.
        - Higher values indicate a more complex activation pattern with many frequency components
        - Lower values suggest a simpler pattern dominated by fewer frequencies
        """
        if not self.activations:
            return 0.0
        if len(self.activations) < 10:
            return 0.5

        # Compute the complexity of the neuron
        activations_tensor = torch.tensor(self.activations, device=self.device)
        
        # Apply windowing to reduce spectral leakage
        # Hann window smooths the signal before applying FFT, reducing spectral leakage (artifacts due to non-periodic signals)
        window = torch.hann_window(len(activations_tensor), device=self.device)
        windowed_activations = activations_tensor * window
        
        # Apply Fast Fourier Transform (FFT) and take absolute values to get magnitude of frequency components
        fft_components = torch.abs(torch.fft.fft(windowed_activations))
        tot_power = torch.sum(fft_components) # Total power of the FFT components
        
        if tot_power == 0:
            return 0.0
        
        # Take only the first half of FFT (positive frequencies) and normalize by total power (only positive values because of symmetry)
        half_fft = fft_components[:len(fft_components)//2]
        half_power = torch.sum(half_fft)
        
        if half_power == 0:
            return 0.0
        
        norm_fft = half_fft / half_power # Normalize by total power
        cum_power = torch.cumsum(norm_fft, dim=0) # Cumulative sum of normalized FFT components
        
        # Count how many frequency components are needed to reach 80% of the total power
        power_threshold = 0.80
        n_components = torch.sum(cum_power < power_threshold).item()
        
        # Adjust normalization to create more differentiation
        max_components = len(half_fft)
        norm_n_components = n_components / max_components
        tensor = torch.tensor([norm_n_components], device=self.device)
        
        # Apply non-linear scaling to spread out values
        # This creates more differentiation between medium and high complexity
        scaled_complexity = torch.tanh(tensor * 3).item()
        
        return scaled_complexity
    
    def compute_params_complexity(self):
        """Compute the complexity of the neuron based on its parameters
        
        This function measures the complexity of the neuron based on its Hebbian learning parameters.
        - Higher values indicate a more complex neuron with more diverse learning rules
        - Lower values suggest a simpler neuron with fewer learning rules
        """
        params = torch.tensor([
            abs(self.pre_factor.item()),
            abs(self.post_factor.item()),
            abs(self.correlation.item()),
            abs(self.decorrelation.item()),
        ])
        #TODO: also include eta in params?
        
        # Compute complexity based on the parameters
        magnitude = torch.sum(params).item() # magnitude is defined as the sum of absolute values of parameters
        
        # Normalize between 0 and 1
        norm_magnitude = min(1.0, magnitude) #TODO: check if normalization should be min(1.0, magnitude / 0.4) where 0.4 is the max value of params
        return norm_magnitude
    
    def compute_complexity(self):
        # If not enough data, return middle complexity
        if not self.weight_changes or len(self.weight_changes) < 5:
            return 0.5
        
        weight_changes = torch.tensor(self.weight_changes, device=self.device)
        weight_diversity = torch.std(weight_changes).item()
        # Normalize between 0 and 1
        norm_diversity = min(1.0, weight_diversity / 0.5)
        return norm_diversity
        
    def compute_new_descriptor(self):
        if not self.activations:
            return 0.0, 0.0
        
        # 1. Temporal stability: Whether a neuron's behavior changes over time (behavioral variability)
        # 2. Signal complexity: Whether a neuron has simple or complex activation patterns (complexity)

        # Compute behavioral variability
        behavioral_variability = self.compute_behavioral_variability()
        # Compute complexity
        complexity = self.compute_params_complexity()
        
        return behavioral_variability, complexity
        

    def compute_descriptors(self):
        """Compute the descriptors for the neuron."""
        if not self.activations:
            return 0.0, 0.0
            
        # Compute on specified device
        activations_tensor = torch.tensor(self.activations, device=self.device)
        
        # Descriptor 1: Average entropy of the activations
        hist = torch.histc(activations_tensor, bins=20, min=-1.0, max=1.0) 
        prob = hist / torch.sum(hist) + 1e-6  # Keep the small epsilon for numerical stability
        avg_entropy = -torch.sum(prob * torch.log2(prob)).item()
        
        # Descriptor 2: Average absolute weight change
        if self.weight_changes:
            weight_changes_tensor = torch.tensor(self.weight_changes, device=self.device)
            avg_weight_change = torch.mean(torch.abs(weight_changes_tensor)).item()
        else:
            avg_weight_change = 0.0
        
        return avg_entropy, avg_weight_change

class NCHL(nn.Module):
    def __init__(self, nodes: list, params=None, population=None, grad=False, device="cpu", init=None):
        super(NCHL, self).__init__()
        
        self.device = device  
        self.grad = grad
        self.nodes = torch.tensor(nodes, device=device)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in range(len(self.nodes) - 1)])
        
        # Initialize Neurons
        self.all_neurons = []
        self.neurons = self._initialize_neurons(nodes, population, device)
        
        # Create network layers
        self.network = self._initialize_network(nodes, init)
        
        self.double()
        self.to(device) 
        
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]
        
        if params is not None:
            self.set_params(params)

    def _initialize_neurons(self, nodes, population, device):
        neurons = []
        neuron_id = 0
        
        if population is not None:
            assert len(population) == sum(nodes), (
                f"Population size does not match number of neurons. "
                f"Expected: {sum(nodes)}, Got: {len(population)}"
            )
            i = 0
            for n_neurons in nodes:
                layer_neurons = []
                for _ in range(n_neurons):
                    layer_neurons.append(population[i])
                    self.all_neurons.append(population[i])
                    i += 1
                neurons.append(layer_neurons)
        else:
            for n_neurons in nodes:
                layer_neurons = []
                for _ in range(n_neurons):
                    neuron = Neuron(neuron_id, device=device)
                    layer_neurons.append(neuron)
                    self.all_neurons.append(neuron)
                    neuron_id += 1
                neurons.append(layer_neurons)
        
        return neurons

    def _initialize_network(self, nodes, init):
        network = []
        for i in range(len(nodes) - 1):
            layer = nn.Linear(nodes[i], nodes[i + 1], bias=False)
            layer.double()
            
            if init is None:
                nn.init.xavier_uniform_(layer.weight.data)
            else:
                self._initialize_weights(layer, init)
                
            layer.weight.data = layer.weight.data.double()
            layer.to(self.device)  
            network.append(layer)
        return network

    def _initialize_weights(self, layer, init):
        if init == 'xa_uni':
            nn.init.xavier_uniform_(layer.weight.data, 0.3)
        elif init == 'sparse':
            nn.init.sparse_(layer.weight.data, 0.8)
        elif init == 'uni':
            nn.init.uniform_(layer.weight.data, -0.1, 0.1)
        elif init == 'normal':
            nn.init.normal_(layer.weight.data, 0, 0.024)
        elif init == 'ka_uni':
            nn.init.kaiming_uniform_(layer.weight.data, 3)
        elif init == 'uni_big':
            nn.init.uniform_(layer.weight.data, -1, 1)
        elif init == 'xa_uni_big':
            nn.init.xavier_uniform_(layer.weight.data)

    def forward(self, inputs):
        with torch.no_grad():
            x = inputs.to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Set input layer activations (using first item in batch)
            for i, neuron in enumerate(self.neurons[0]):
                neuron.set_activation(x[0, i])
            
            # Forward pass
            for layer_idx, layer in enumerate(self.network):
                x = torch.tanh(layer(x))
                # Set activations for neurons in current layer
                for i, neuron in enumerate(self.neurons[layer_idx + 1]):
                    neuron.set_activation(x[0, i])
            return x

    def update_weights(self):
        weights = self.get_weights()
        
        for layer_idx in range(len(weights)):
            pre_neurons = self.neurons[layer_idx]
            post_neurons = self.neurons[layer_idx + 1]
            
            # Pre-compute Hebbian terms for the layer
            pre_terms = torch.stack([torch.stack(n.get_hebbian_terms()).to(self.device) for n in pre_neurons])
            post_terms = torch.stack([torch.stack(n.get_hebbian_terms()).to(self.device) for n in post_neurons])
            
            # Prepare activations 
            pre_activations = torch.stack([n.activation.to(self.device) for n in pre_neurons])
            post_activations = torch.stack([n.activation.to(self.device) for n in post_neurons])
            
            # Create weight update matrix
            pre_contribution = pre_terms[:, 0].unsqueeze(0).expand(len(post_neurons), -1).to(self.device)
            post_contribution = post_terms[:, 1].unsqueeze(1).expand(-1, len(pre_neurons)).to(self.device)
            
            # Correlation terms 
            corr_mask = ((pre_terms[:, 2] != 1.).unsqueeze(0) & (post_terms[:, 2] != 1.).unsqueeze(1)).to(self.device)
            corr_contrib = torch.where(
                corr_mask,
                pre_terms[:, 2].unsqueeze(0).to(self.device) * post_terms[:, 2].unsqueeze(1).to(self.device),
                torch.zeros_like(pre_contribution, device=self.device)
            )
            
            # Decorrelation terms 
            decorr_mask = ((pre_terms[:, 3] != 1.).unsqueeze(0) & (post_terms[:, 3] != 1.).unsqueeze(1)).to(self.device)
            decorr_contrib = torch.where(
                decorr_mask,
                pre_terms[:, 3].unsqueeze(0).to(self.device) * post_terms[:, 3].unsqueeze(1).to(self.device),
                torch.zeros_like(pre_contribution, device=self.device)
            )
            
            # Combine all contributions 
            dw = (pre_contribution + post_contribution + corr_contrib + decorr_contrib).to(self.device)
            
            # Learning rates 
            pre_etas = torch.stack([n.eta.to(self.device) for n in pre_neurons])
            post_etas = torch.stack([n.eta.to(self.device) for n in post_neurons])
            eta_matrix = ((pre_etas.unsqueeze(0) + post_etas.unsqueeze(1)) / 2).to(self.device)
            
            # Final weight update
            weight_change = (eta_matrix * dw).to(self.device)
            
            # Store weight changes and update weights
            for i, post_neuron in enumerate(post_neurons):
                for j, pre_neuron in enumerate(pre_neurons):
                    change = weight_change[i, j].item()
                    pre_neuron.add_weight_change(change)
                    post_neuron.add_weight_change(change)
            
            # Update weights
            weights[layer_idx] = (weights[layer_idx].to(self.device) + weight_change).to(self.device)

        self.set_weights(weights)

    def get_weights(self):
        return [l.weight.data for l in self.network]

    def set_weights(self, weights):
        if isinstance(weights[0], torch.Tensor):
            for i, weight in enumerate(weights):
                self.network[i].weight = nn.Parameter(weight.to(self.device), requires_grad=self.grad)
        else:
            tmp = self.get_weights()
            start = 0
            for i, l in enumerate(tmp):
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size], device=self.device)
                start = size
                self.network[i].weight = nn.Parameter(
                    torch.reshape(params, (l.size()[0], l.size()[1])),
                    requires_grad=self.grad
                )

    def set_params(self, params: list):
        """Set learning rates (etas) and Hebbian rules for all neurons."""
        etas = params[:sum(self.nodes)]
        hrules = params[sum(self.nodes):]
        
        # Set learning rates for each neuron
        start = 0
        for layer in self.neurons:
            for neuron in layer:
                neuron.set_eta(etas[start])
                start += 1
        
        # Set Hebbian rules
        start = 0
        # Input layer (3 parameters per neuron)
        for neuron in self.neurons[0]:
            rules = hrules[start:start + 3]
            neuron.set_hebbian_params(
                pre=rules[0],
                post=0.0,  # Input layer has no post factor
                corr=rules[1],
                decorr=rules[2]
            )
            start += 3
        
        # Hidden layers (4 parameters per neuron)
        for layer in self.neurons[1:-1]:
            for neuron in layer:
                rules = hrules[start:start + 4]
                neuron.set_hebbian_params(
                    pre=rules[0],
                    post=rules[1],
                    corr=rules[2],
                    decorr=rules[3]
                )
                start += 4
        
        # Output layer (3 parameters per neuron)
        for neuron in self.neurons[-1]:
            rules = hrules[start:start + 3]
            neuron.set_hebbian_params(
                pre=0.0,  # Output layer has no pre factor
                post=rules[0],
                corr=rules[1],
                decorr=rules[2]
            )
            start += 3