import numpy as np
import torch
import torch.nn as nn

class Neuron:
    def __init__(self, neuron_id: int, device="cpu"):
        """
        Initialize a neuron with its unique ID and parameters.

        Args:
            neuron_id (int): Unique identifier for the neuron
            device (str): Computing device (CPU/GPU)
        """
        self.id = neuron_id
        self.device = torch.device(device)

        # Hebbian rule parameters
        # Set with random values - TODO: check if this is the best way to initialize
        self.pre_factor = np.random.uniform(-1.0, 1.0)
        self.post_factor = np.random.uniform(-1.0, 1.0)
        self.correlation = np.random.uniform(-1.0, 1.0)
        self.decorrelation = np.random.uniform(-1.0, 1.0)
        self.eta = np.random.uniform(0.001, 0.3)  # Broader learning rate range
        
        # Current activation value
        self.activation = 0.0
        
        # Store activations and weight changes for the neuron
        self.activations = [] # neuron's activation over time (need for descriptor 1)
        self.weight_changes = [] # neuron's weight changes over time (need for descriptor 1)

    def add_activation(self, activation):
        """Add an activation to the list of activations."""
        self.activations.append(activation.item())
        
    def add_weight_change(self, weight_change):
        """Add a weight change to the list of weight changes."""
        self.weight_changes.append(weight_change)
        
    def set_hebbian_params(self, pre: float, post: float, corr: float, decorr: float):
        """Set the Hebbian learning parameters for this neuron."""
        self.pre_factor = torch.tensor(pre).to(self.device)
        self.post_factor = torch.tensor(post).to(self.device)
        self.correlation = torch.tensor(corr).to(self.device)
        self.decorrelation = torch.tensor(decorr).to(self.device)

    def set_eta(self, eta: float):
        """Set the learning rate for this neuron."""
        self.eta = torch.tensor(eta).to(self.device)

    def set_activation(self, activation):
        """Set the current activation value of the neuron."""
        self.activation = activation.to(self.device)
        self.add_activation(activation) # add the activation to the list of activations

    def get_hebbian_terms(self):
        """Get the Hebbian terms for weight updates."""
        pre = self.pre_factor * self.activation
        post = self.post_factor * self.activation
        corr = 1. if self.correlation == 1. else self.correlation * self.activation
        decorr = self.decorrelation
        return pre, post, corr, decorr
    
    def compute_descriptors(self):
        """Compute the descriptors for the neuron."""
        # Descriptor 1: Average entropy of the activations
        if len(self.activations) > 0:
            hist, _ = np.histogram(self.activations, bins=10, density=True)  # Normalize histogram
            prob = hist + 1e-8 
            avg_entropy = -np.sum(prob * np.log(prob))
        else:
            avg_entropy = 0  # Default if no activations available

        # Descriptor 2: Average weight change
        avg_weight_change = np.mean(self.weight_changes) if len(self.weight_changes) > 0 else 0

        return avg_entropy, avg_weight_change

    def __repr__(self):
        return f"\n     Neuron ({self.id})\
            \n          Activation: {self.activation}\
            \n          Hebbian parameters: {self.pre_factor}, {self.post_factor}, {self.correlation}, {self.decorrelation}\
            \n          Learning rate: {self.eta}\
            \n"
        
class NCHL(nn.Module):
    def __init__(self, nodes: list, params=None, population=None,  grad=False, device="cpu", init=None):
        super(NCHL, self).__init__()
        self.device = torch.device(device)
        self.grad = grad
        self.nodes = torch.tensor(nodes).to(self.device)
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1]
                            for i in range(len(self.nodes) - 1)])

        # Create neurons for each layer
        self.all_neurons = []
        self.neurons = []
        if population is not None:
            # Check if population is valid
            assert len(population) == sum(nodes), ("Population size does not match number of neurons. Expected: {}, Got: {}".format(
                sum(nodes), len(population)))
            # Use provided population
            i = 0
            for layer, n_neurons in enumerate(nodes):
                layer_neurons = []
                for _ in range(n_neurons):
                    layer_neurons.append(population[i])
                    self.all_neurons.append(population[i])
                    i += 1
                self.neurons.append(layer_neurons)

        else:
            # Create new population
            neuron_id = 0
            for layer, n_neurons in enumerate(nodes):
                layer_neurons = []
                for _ in range(n_neurons):
                    neuron = Neuron(neuron_id, device)
                    layer_neurons.append(neuron)
                    self.all_neurons.append(neuron)
                    neuron_id += 1
                self.neurons.append(layer_neurons)

        # Create linear layers
        self.network = []
        for i in range(len(nodes) - 1):
            layer = nn.Linear(nodes[i], nodes[i + 1], bias=False)
            layer.double()
            self.network.append(layer)

        # Initialize weights
        if init is None:
            # Default initialization: Xavier uniform
            for layer in self.network:
                nn.init.xavier_uniform_(layer.weight.data)
                layer.weight.data = layer.weight.data.double()
        else:
            self._initialize_weights(init)

        self.double()
        self.nparams = sum(self.nodes) * 5 - self.nodes[0] - self.nodes[-1]

        # Set parameters if provided
        if params is not None:
            self.set_params(params)

    def _initialize_weights(self, init):
        for l in self.network:
            if init == 'xa_uni':
                torch.nn.init.xavier_uniform_(l.weight.data, 0.3)
            elif init == 'sparse':
                torch.nn.init.sparse_(l.weight.data, 0.8)
            elif init == 'uni':
                torch.nn.init.uniform_(l.weight.data, -0.1, 0.1)
            elif init == 'normal':
                torch.nn.init.normal_(l.weight.data, 0, 0.024)
            elif init == 'ka_uni':
                torch.nn.init.kaiming_uniform_(l.weight.data, 3)
            elif init == 'uni_big':
                torch.nn.init.uniform_(l.weight.data, -1, 1)
            elif init == 'xa_uni_big':
                torch.nn.init.xavier_uniform_(l.weight.data)
            # Ensure weights are in double precision
            l.weight.data = l.weight.data.double()

    def forward(self, inputs):
        with torch.no_grad():
            x = inputs.to(self.device)
            # Ensure input is 2D
            if x.dim() == 1:
                x = x.unsqueeze(0)

            # Set input layer activations (using first item in batch)
            for i, neuron in enumerate(self.neurons[0]):
                neuron.set_activation(x[0, i])

            # Forward pass through the network
            for layer_idx in range(len(self.network)):
                x = self.network[layer_idx](x)
                x = torch.tanh(x)
                # Set activations for neurons in the current layer (using first item in batch)
                for i, neuron in enumerate(self.neurons[layer_idx + 1]):
                    neuron.set_activation(x[0, i])
            return x

    def get_weights(self):
        return [l.weight.data for l in self.network]

    def set_weights(self, weights):
        if type(weights) == list and type(weights[0]) == torch.Tensor:
            for i in range(len(self.network)):
                self.network[i].weight = nn.Parameter(
                    weights[i], requires_grad=self.grad)
        elif len(weights) == self.nweights:
            tmp = self.get_weights()
            start = 0
            i = 0
            for l in tmp:
                size = l.size()[0] * l.size()[1] + start
                params = torch.tensor(weights[start:size])
                start = size
                self.network[i].weight = nn.Parameter(
                    torch.reshape(params, (l.size()[0], l.size()[1])).to(
                        self.device),
                    requires_grad=self.grad)
                i += 1

    def set_params(self, params: list):
        etas = params[:sum(self.nodes)]
        hrules = params[sum(self.nodes):]
        self._set_neuron_params(etas, hrules)

    def _set_neuron_params(self, etas: list, hrules: list):
        start = 0
        eta_idx = 0

        # Input layer
        for neuron in self.neurons[0]:
            hrule = hrules[start:start + 3]
            neuron.set_hebbian_params(hrule[0], 0., hrule[1], hrule[2])
            neuron.set_eta(etas[eta_idx])
            start += 3
            eta_idx += 1

        # Hidden layers
        for layer in self.neurons[1:-1]:
            for neuron in layer:
                hrule = hrules[start:start + 4]
                neuron.set_hebbian_params(*hrule)
                neuron.set_eta(etas[eta_idx])
                start += 4
                eta_idx += 1

        # Output layer
        for neuron in self.neurons[-1]:
            hrule = hrules[start:start + 3]
            neuron.set_hebbian_params(0., hrule[0], hrule[1], hrule[2])
            neuron.set_eta(etas[eta_idx])
            start += 3
            eta_idx += 1

    def update_weights(self):
        weights = self.get_weights()
        num_layers = len(weights)

        for layer_idx in range(num_layers):
            pre_neurons = self.neurons[layer_idx]
            post_neurons = self.neurons[layer_idx + 1]

            # Calculate weight updates using neuron parameters
            for i, post_neuron in enumerate(post_neurons):
                for j, pre_neuron in enumerate(pre_neurons):
                    # Get Hebbian terms from both neurons
                    pre_terms = pre_neuron.get_hebbian_terms()
                    post_terms = post_neuron.get_hebbian_terms()

                    # Calculate weight update
                    dw = (pre_terms[0] + post_terms[1] +
                          (0 if (pre_terms[2] == 1. and post_terms[2] == 1.)
                           else pre_terms[2] * post_terms[2]) +
                          (0 if (pre_terms[3] == 1. and post_terms[3] == 1.)
                           else pre_terms[3] * post_terms[3]))

                    # Apply learning rate and update weight
                    eta = (pre_neuron.eta + post_neuron.eta) / 2
                    delta_w = eta * dw # weight change
                    
                    # Store weight change for neuron
                    # store on both neurons since the weight change is the same for both
                    pre_neuron.add_weight_change(delta_w.item()) 
                    post_neuron.add_weight_change(delta_w.item()) 
                    
                    # Update weight
                    weights[layer_idx][i, j] += delta_w

        self.set_weights(weights)


# print structure of the network
fka = NCHL([4, 4, 2])
for layer in fka.neurons:
    print(f"[Layer {fka.neurons.index(layer)}]")
    for neuron in layer:
        print(neuron)