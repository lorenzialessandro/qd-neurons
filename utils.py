import numpy as np
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap
from ribs.archives import GridArchive

def log_data(filename, best_fitness, history, archive, hparams):
    with open(filename, 'w') as f:
        # log final statistics
        f.write("Final Statistics:\n")
        f.write(f"Best fitness achieved: {best_fitness:.2f}\n")
        f.write(f"Final average fitness: {np.mean(history):.2f}\n")
        f.write("\nArchive stats:\n")
        f.write(str(archive.stats))
        # log hyperparameters
        f.write("\n\nHyperparameters:\n")
        for param in vars(hparams):
            f.write(f'{param}: {getattr(hparams, param)}\n')


def plot_archive_heatmap(archive: GridArchive, output_dir):
    grid_archive_heatmap(archive, cmap='Greens')
    plt.title('Archive Heatmap')
    plt.xlabel('Average Entropy')
    plt.ylabel('Average Weight Change')
    
    plt.savefig(f'{output_dir}/archive_heatmap.png')
    plt.show()
    
def plot_fitness_history(history, output_dir, best=False):
    best = 'Best' if best else ''
    iterations = np.arange(len(history))
    trend = np.poly1d(np.polyfit(iterations, history, 1))
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Fitness")
    plt.plot(iterations, trend(iterations), linestyle="dashed", color="red", label="Trend Line")  # trend line
    plt.title(f'{best} Fitness History')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()  # Show legend
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cart_pole_{best}_fitness_history.png')
    plt.show()