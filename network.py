import numpy as np
import torch
import torch.nn as nn

class Neuron:
    def __init__(self, neuron_id: int, device="cpu"):
        self.id = neuron_id
        self.device = device  
        
         Hebbian rule parameters initialized to zero
        self.pre_factor = torch.tensor(0.0, device=device)
        self.post_factor = torch.tensor(0.0, device=device)
        self.correlation = torch.tensor(0.0, device=device)
        self.decorrelation = torch.tensor(0.0, device=device)
        self.eta = torch.tensor(0.0, device=device)
        
        # Current activation value of the neuron
        self.activation = torch.tensor(0.0, device=device)
        
        # Store activations and weight changes for the neuron for descriptors
        self.activations = []
        self.weight_changes = []

    def add_activation(self, activation):
        """Add an activation to the list of activations."""
        self.activations.append(activation.item())
        
    def add_weight_change(self, weight_change):
        """Add a weight change to the list of weight changes."""
        self.weight_changes.append(weight_change)
        
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

    def get_hebbian_terms(self):
        """Get the Hebbian terms for weight updates."""
        return (
            (self.pre_factor * self.activation).to(self.device),
            (self.post_factor * self.activation).to(self.device),
            1. if self.correlation == 1. else (self.correlation * self.activation).to(self.device),
            self.decorrelation.to(self.device)
        )

    def compute_descriptors(self):
        """Compute the descriptors for the neuron."""
        if not self.activations:
            return 0.0, 0.0
            
        # Compute on specified device
        activations_tensor = torch.tensor(self.activations, device=self.device)
        
        # Descriptor 1: Average entropy of the activations
        hist = torch.histc(activations_tensor, bins=10, min=-1.0, max=1.0) # [-1, 1] because of tanh
        # Normalize histogram to get probability distribution
        prob = hist / torch.sum(hist) + 1e-6 
        # Compute entropy
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
                    neuron = Neuron(neuron_id, device)
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