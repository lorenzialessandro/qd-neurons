from matplotlib.cm import get_cmap
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap
from ribs.archives import GridArchive
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the configuration file
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def save_network_params(network, output_dir):
    """Save the parameters of a network"""
    params = []
    for neuron in network.all_neurons:
        params.append({
            'neuron_id': neuron.neuron_id,
            'params': neuron.params
        })
    
    np.save(f"{output_dir}/best_network_params.npy", params)

# -----------------------------

def plot_archive_heatmap(archive: GridArchive, output_dir):
    grid_archive_heatmap(archive, cmap='Greens')
    plt.title('Archive Heatmap')
    plt.xlabel('Average Entropy')
    plt.ylabel('Average Weight Change')

    # plt.savefig(f'{output_dir}/archive_heatmap.png')
    plt.show()

def plot_params(pop, path):
    """For each neuron plot final params value as column"""
    labels = ['pre', 'post', 'corr', 'dec', 'eta']
    colors = ['b', 'g', 'r', 'c', 'm']
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    
    for neuron in pop:
        plt.sca(axs[neuron.neuron_id])
        params = neuron.get_rule()
    
        bars = plt.bar(range(len(params)), params, color=colors) # Plot params as bars
    
        plt.xticks(range(len(params)), labels)
        plt.title(f"Neuron {neuron.neuron_id}")
        plt.ylim([-1.2, 1.2])
            
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)    
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{path}/all_neurons_params.png")

def plot_heatmaps(pop, archives, path, init=False):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    init = "_init" if init else ""

    for neuron in pop:
        archive = archives[neuron.neuron_id]
        plt.sca(axs[neuron.neuron_id])  # Set the current axis
        # grid_archive_heatmap(archive, cmap='Greens')
        archive.visualize(ax=axs[neuron.neuron_id], cmap='Greens')
        plt.xlabel('Temporal stability')
        #plt.ylabel('Signal complexity')
        plt.ylabel('Eta')
        plt.title(f'Neuron {neuron.neuron_id}')

    plt.tight_layout()
    plt.savefig(f"{path}/all_neurons_heatmap{init}.png")
    # plt.show()

def plot_k_pcas(pop, archives, path, k=5):
    """k are the best rules"""
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    for neuron in pop:
        archive = archives[neuron.neuron_id]
        plt.sca(axs[neuron.neuron_id])
        
        # Perform and plot PCA
        params = []
        fitnesses = []
        data = archive.data()
        # Create a list of (solution, objective) pairs for sorting
        solution_objective_pairs = list(zip(data["solution"], data["objective"]))
        # Sort by objective value
        sorted_pairs = sorted(solution_objective_pairs, key=lambda pair: pair[1])
        
        k = min(k, len(sorted_pairs))
        
        # Get the k best solutions
        k_solutions = [pair[0] for pair in sorted_pairs[:k]]
        # Get the fitnesses of the k best solutions
        k_fitnesses = [pair[1] for pair in sorted_pairs[:k]]
        
        scaler = StandardScaler()
        params_scaled = scaler.fit_transform(k_solutions)
        pca = PCA()  # apply PCA
        pca_result = pca.fit_transform(params_scaled)
        scatter = plt.scatter(
            pca_result[:, 0], pca_result[:, 1], c=k_fitnesses, cmap='viridis')
        
        plt.xlabel(f"PC1")
        plt.ylabel("PC2")
        plt.title(f'Neuron {neuron.neuron_id}')
        plt.colorbar(scatter, label='Fitness')
  
    plt.tight_layout()
    plt.savefig(f"{path}/k_best_rules_pca.png")
    # plt.show()
            

def plot_pcas(pop, archives, path):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    for neuron in pop:
        archive = archives[neuron.neuron_id]
        plt.sca(axs[neuron.neuron_id])  # Set the current axis

        # Perform and plot PCA

        params = archive.data()["solution"]
        fitness = archive.data()["objective"]
        scaler = StandardScaler()
        params_scaled = scaler.fit_transform(params)
        pca = PCA()  # apply PCA
        pca_result = pca.fit_transform(params_scaled)
        scatter = plt.scatter(
            pca_result[:, 0], pca_result[:, 1], c=fitness, cmap='viridis')
        # print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        # loadings = pca.components_
        # print("PC1 loadings:", loadings[0])
        # print("PC2 loadings:", loadings[1])

        # Plot the PCA axes (principal components)
        # origin = [0, 0]  # The origin of the axes (usually the center of the plot)
        # for i in range(pca.components_.shape[0]):
        #     plt.quiver(*origin, *pca.components_[i, :2], angles='xy', scale_units='xy', scale=1, color='red', linewidth=2)

        plt.xlabel(f"PC1")
        plt.ylabel("PC2")
        plt.title(f'Neuron {neuron.neuron_id}')
        plt.colorbar(scatter, label='Fitness')

    plt.tight_layout()
    plt.savefig(f"{path}/all_neurons_pca.png")
    # plt.show()


def plot_fitness_trends(best_fitness_per_iteration, avg_fitness_per_iteration, path, threshold):
    plt.figure(figsize=(10, 6))
    generations = np.arange(len(best_fitness_per_iteration))

    # Plot best fitness
    plt.plot(generations, best_fitness_per_iteration,
             'b-', label="Best Fitness")
    # Plot average fitness
    plt.plot(generations, avg_fitness_per_iteration,
             'r-', label="Average Fitness")
    # Plot threshold
    plt.axhline(y=threshold, color='r',
                linestyle='--', label="Threshold")

    plt.title("Fitness Trends")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/fitness_trends.png')
    # plt.show()


def plot_pca_best_rules(pop, archives, path):
    params = []
    fitnesses = []
    neuron_ids = []
    for neuron in pop:
        best_rule = archives[neuron.neuron_id].best_elite
        params.append(best_rule["solution"])
        fitnesses.append(best_rule["objective"])
        neuron_ids.append(neuron.neuron_id)

    params = np.array(params)
    fitnesses = np.array(fitnesses)
    neuron_ids = np.array(neuron_ids)

    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params)
    pca = PCA()  # apply PCA
    pca_result = pca.fit_transform(params_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                          s=100, c=fitnesses, cmap='viridis')
    # add neuron id as text
    for i, txt in enumerate(neuron_ids):
        plt.annotate(f"N {txt}", (pca_result[i, 0], pca_result[i, 1]))
    plt.colorbar(scatter, label='Fitness')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Best Rules')
    plt.savefig(f'{path}/pca_best_rules.png')

# -----------------------------
# --- Network Visualization ---

def visualize_network(net, threshold=0.1, figsize=(12, 8), node_size=500,
                      show_weights=True, fitness=0, cmap_name='coolwarm', save=False, file_name='network_structure.png'):
    """
    Create a visualization of the neural network structure and connections.

    Parameters:
    - net: NCHL network instance
    - threshold: Only show connections with absolute weight values above this threshold
    - figsize: Size of the figure (width, height)
    - node_size: Size of the nodes
    - show_weights: Whether to show weight values on edges
    - cmap_name: Colormap for edge colors based on weight values
    """
    G = nx.DiGraph()

    # Get all neurons and weights
    weights = net.get_weights()

    # Create a mapping for neuron positions
    pos = {}
    neuron_to_layer = {}

    # Add nodes for each neuron
    for layer_idx, layer in enumerate(net.neurons):
        layer_size = len(layer)

        for neuron_idx, neuron in enumerate(layer):
            # Add node with neuron ID
            node_id = f"N{neuron.neuron_id}"
            G.add_node(node_id)

            # Position neurons in layers evenly spaced
            y_position = neuron_idx - (layer_size - 1) / 2
            pos[node_id] = (layer_idx * 2, y_position)

            # Store neuron to layer mapping for later
            neuron_to_layer[node_id] = layer_idx

            # Add node attributes
            G.nodes[node_id]['activation'] = neuron.activation.item()
            G.nodes[node_id]['layer'] = layer_idx

    # Add edges between neurons in adjacent layers
    edge_weights = []

    for layer_idx in range(len(net.neurons) - 1):
        weight_matrix = weights[layer_idx].cpu().numpy()

        # Loop through each connection
        for i in range(weight_matrix.shape[0]):  # Post-neurons
            post_neuron_id = net.neurons[layer_idx + 1][i].neuron_id

            for j in range(weight_matrix.shape[1]):  # Pre-neurons
                pre_neuron_id = net.neurons[layer_idx][j].neuron_id
                weight = weight_matrix[i, j]

                # Only add edges with weights above threshold
                if abs(weight) > threshold:
                    G.add_edge(f"N{pre_neuron_id}",
                               f"N{post_neuron_id}", weight=weight)
                    edge_weights.append(weight)

    # Create figure
    plt.figure(figsize=figsize)

    # Get colormap and normalize weights for coloring
    cmap = get_cmap(cmap_name)
    if edge_weights:
        max_weight = max(abs(min(edge_weights)), abs(max(edge_weights)))
        edge_colors = [cmap(0.5 + weight / (2 * max_weight))
                       for weight in edge_weights]
    else:
        edge_colors = []

    # Draw the nodes by layer with different colors
    for layer_idx in range(len(net.neurons)):
        layer_nodes = [node for node,
                       layer in neuron_to_layer.items() if layer == layer_idx]
        label_dict = {node: node.replace('N', '') for node in layer_nodes}

        nx.draw_networkx_nodes(G, pos,
                               nodelist=layer_nodes,
                               node_color=f'C{layer_idx}',
                               node_size=node_size,
                               alpha=0.8)

        nx.draw_networkx_labels(G, pos, labels=label_dict)

    # Draw edges with colors based on weights
    edges = list(G.edges())
    if edges:
        nx.draw_networkx_edges(G, pos,
                               edgelist=edges,
                               width=2,
                               alpha=0.6,
                               edge_color=edge_colors,
                               #   connectionstyle='arc3,rad=0.1' # Curved edges for better visualization
                               )

    # Add edge labels if needed
    if show_weights and edges:
        # Create a dictionary of edge labels with formatted weights
        edge_labels = {
            (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        # Draw edge labels with better positioning to avoid overlap
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='black',
            font_family='sans-serif',
            font_weight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
            label_pos=0.8,  # Position labels closer to source nodes to reduce overlap
            rotate=True
        )

    # Add layer labels
    for layer_idx in range(len(net.neurons)):
        plt.text(layer_idx * 2,
                 max(pos.values(), key=lambda x: x[1])[1] + 1,
                 f'Layer {layer_idx}',
                 horizontalalignment='center',
                 fontsize=12)

    plt.axis('off')
    plt.tight_layout()

    # # Add a colorbar to show weight scale
    # if edge_weights:
    #     sm = plt.cm.ScalarMappable(cmap=cmap,
    #                               norm=plt.Normalize(-max_weight, max_weight))
    #     sm.set_array([])
    #     cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal',
    #                       pad=0.1, fraction=0.05, shrink=0.8)
    #     cbar.set_label('Connection Weight')

    plt.title('Fitness: {:.2f}'.format(fitness))
    if save:
        plt.savefig(file_name)
    plt.show()

    # Return the graph and positions for further manipulation if needed
    return G, pos


def print_network_structure_and_connections(net, threshold=0.1):
    """
    Print the structure of the network including neurons and their connections.

    Parameters:
    - net: NCHL network instance
    - threshold: Only print connections with absolute weight values above this threshold
                 (set to 0 to print all connections)
    """
    print("\nNetwork Structure and Connections:")
    weights = net.get_weights()

    for layer_idx, layer in enumerate(net.neurons):
        print(f"[Layer {layer_idx}]")
        for neuron_idx, neuron in enumerate(layer):
            print(
                f"    Neuron {neuron.neuron_id} (activation: {neuron.activation.item():.4f})")

            # Print outgoing connections for all layers except the output layer
            if layer_idx < len(net.neurons) - 1:
                # Get the weights for connections from this neuron to the next layer
                outgoing_weights = weights[layer_idx][:, neuron_idx]

                connections = []
                for next_idx, weight in enumerate(outgoing_weights):
                    weight_val = weight.item()
                    if abs(weight_val) > threshold:
                        next_neuron_id = net.neurons[layer_idx +
                                                     1][next_idx].neuron_id
                        connections.append(
                            f"Neuron {next_neuron_id} (weight: {weight_val:.4f})")

                if connections:
                    print(f"        Outgoing connections to:")
                    for conn in connections:
                        print(f"            → {conn}")
                else:
                    print(
                        f"        No significant outgoing connections (threshold: {threshold})")

            # Print incoming connections for all layers except the input layer
            if layer_idx > 0:
                # Get the weights for connections to this neuron from the previous layer
                incoming_weights = weights[layer_idx - 1][neuron_idx]

                connections = []
                for prev_idx, weight in enumerate(incoming_weights):
                    weight_val = weight.item()
                    if abs(weight_val) > threshold:
                        prev_neuron_id = net.neurons[layer_idx -
                                                     1][prev_idx].neuron_id
                        connections.append(
                            f"Neuron {prev_neuron_id} (weight: {weight_val:.4f})")

                if connections:
                    print(f"        Incoming connections from:")
                    for conn in connections:
                        print(f"            ← {conn}")
                else:
                    print(
                        f"        No significant incoming connections (threshold: {threshold})")

    print("*" * 50)
