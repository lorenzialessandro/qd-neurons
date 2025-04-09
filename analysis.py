import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

from network import NCHL, Neuron

def get_archive_data(archive, neuron_id):
    data = archive.data()
    
    solutions = []
    objectives = []
    measures = []

    for key, value in data.items():
        solutions.append(value['rule'])
        objectives.append(value['fitness'])
        measures.append(value['behavior'])
    
    all_data = []
    
    for i in range(len(solutions)):
        all_data.append({
            "neuron_id": neuron_id,
            "solution": solutions[i],
            "pre": solutions[i][0],
            "post": solutions[i][1],
            "corr": solutions[i][2],
            "dec": solutions[i][3],
            "eta": solutions[i][4],
            "fit": objectives[i],
            "variability": measures[i][0],
            "complexity": measures[i][1]
        })
        
    df = pd.DataFrame(all_data)
    return df 

def get_all_archive_data(pop, archives):
    all_data = []
    
    for neuron in pop:
        archive = archives[neuron.neuron_id]
        df = get_archive_data(archive, neuron.neuron_id)
        all_data.append(df)
    
    all_data = pd.concat(all_data)
    return all_data
                   
def plot_prams_correlation(pop, data, path):
    params = ['pre', 'post', 'corr', 'dec', 'eta']
    
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for neuron in pop:
        df = data[data['neuron_id'] == neuron.neuron_id]
        
        plt.sca(axs[neuron.neuron_id])
        corr_data = df[params + ['fit']].corr()
        
        sns.heatmap(corr_data, annot=True, vmin=-1, vmax=1)
        plt.title(f"Neuron {neuron.neuron_id}")
    
    plt.suptitle("Correlation between parameters and fitness")
    plt.tight_layout()
    plt.savefig(f"{path}/correlation_all.png")
      
def plot_desc_correlation(pop, data, path):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for i, neuron in enumerate(pop):
        df = data[data['neuron_id'] == neuron.neuron_id]
        
        # Variability-Complexity-Fitness relationship
        scatter = axs[i].scatter(df['variability'], df['complexity'], c=df['fit'], cmap='viridis')
        fig.colorbar(scatter, ax=axs[i], label='Fitness')
        axs[i].set_xlabel('Variability')
        axs[i].set_ylabel('Complexity')
        axs[i].set_title(f"Neuron {neuron.neuron_id}")
    
    plt.suptitle("Variability-Complexity-Fitness relationship")
    plt.tight_layout()
    plt.savefig(f"{path}/desc_all.png")
    
def plot_pca_params(pop, data, path):
    params = ['pre', 'post', 'corr', 'dec', 'eta']
    
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    for i, neuron in enumerate(pop):
        df = data[data['neuron_id'] == neuron.neuron_id]
        param_data = df[params]
        
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(param_data)
        
        scatter = axs[i].scatter(pca_data[:, 0], pca_data[:, 1], c=df['fit'], cmap='viridis')
        fig.colorbar(scatter, ax=axs[i], label='Fitness')
        axs[i].set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2f})')
        axs[i].set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2f})')
        axs[i].set_title(f"Neuron {neuron.neuron_id}")
    
    plt.suptitle("PCA of parameters")
    plt.tight_layout()
    plt.savefig(f"{path}/pca_params.png")
    
def plot_pca_top_k_rules(pop, data, path, k=5):
    params = ['pre', 'post', 'corr', 'dec', 'eta']
    
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    
    for i, neuron in enumerate(pop):
        df = data[data['neuron_id'] == neuron.neuron_id]
        # Select top k rules based on fitness
        df_sorted = df.sort_values(by='fit', ascending=False).head(k)
        param_data = df_sorted[params]
        
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(param_data)
        
        scatter = axs[i].scatter(pca_data[:, 0], pca_data[:, 1], c=df_sorted['fit'], cmap='viridis')
        fig.colorbar(scatter, ax=axs[i], label='Fitness')
        axs[i].set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2f})')
        axs[i].set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2f})')
        axs[i].set_title(f"Neuron {neuron.neuron_id}")
    
    plt.suptitle("PCA of top k rules")
    plt.tight_layout()
    plt.savefig(f"{path}/pca_top_k_rules.png")
    
# --- Main function ---

def plot_analysis(pop, archives, path):
    data = get_all_archive_data(pop, archives)

    plot_params(pop, data, path)            # 0. Parameters vs Fitness
    exit()
    plot_prams_correlation(pop, data, path)     # 1. Correlation between parameters and fitness
    plot_desc_correlation(pop, data, path)      # 2. Descriptors-Fitness relationship
    plot_pca_params(pop, data, path)            # 3. PCA of parameters
    plot_pca_top_k_rules(pop, data, path, 3)    # 4. PCA of top k rules
    
    # 5. Clustering of rules ???