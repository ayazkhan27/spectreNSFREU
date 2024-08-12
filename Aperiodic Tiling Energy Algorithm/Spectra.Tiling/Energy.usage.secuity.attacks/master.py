import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon, Circle
from spectre import buildSpectreBase, transPt, buildSupertiles, SPECTRE_POINTS
from shapely.geometry import LineString

# Parameters
SPECTRE_ITERATIONS = 3
INITIAL_ENERGY = 1000000.0  # 1,000,000 joules
CLUSTER_HEAD_ENERGY_COST = 1.0  # 1 joule per round for cluster heads
TRANSMISSION_TO_CLUSTER_HEAD_COST = 0.0001  # 0.0001 joules per unit distance
ENERGY_CONSUMPTION_RATE = 0.1  # 0.1 joules per unit distance for base station
COMMUNICATION_RADIUS = 20  # Base station communication radius
DPSA_ITERATIONS = 100  # Number of iterations for DPSA
MAIN_ROUNDS = 5  # Number of main rounds

def generate_spectre_tiles():
    print("Generating Spectre tiles...")
    tiles = buildSpectreBase()
    for i in range(SPECTRE_ITERATIONS - 1):
        print(f"Building supertiles iteration {i+1}")
        tiles = buildSupertiles(tiles)
    return tiles

def place_sensors_and_identify_clusters(tiles):
    print("Placing sensors and identifying clusters...")
    clusters = []

    def add_sensor_group(transformation, label):
        tile_points = [transPt(transformation, pt) for pt in SPECTRE_POINTS]
        clusters.append(np.array(tile_points))

    tiles["Delta"].forEachTile(add_sensor_group)
    print(f"Number of clusters: {len(clusters)}")
    return clusters

def select_cluster_heads(clusters, energy_levels):
    print("Selecting cluster heads...")
    cluster_heads = []
    start_index = 0
    for cluster in clusters:
        cluster_size = len(cluster)
        cluster_energy = energy_levels[start_index:start_index + cluster_size]
        head_index = np.argmax(cluster_energy)
        cluster_heads.append(cluster[head_index])
        start_index += cluster_size
    print(f"Number of cluster heads selected: {len(cluster_heads)}")
    return np.array(cluster_heads)

class DPSA:
    def __init__(self, cluster_heads, communication_radius):
        self.cluster_heads = cluster_heads
        self.communication_radius = communication_radius
        self.dimension = len(cluster_heads)
        self.population_size = 50
        self.population = [self.generate_random_path() for _ in range(self.population_size)]
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness(self.best_solution)

    def generate_random_path(self):
        path = list(range(self.dimension))
        np.random.shuffle(path)
        return path + [path[0]]  # Ensure the path ends where it starts

    def fitness(self, path):
        total_distance = sum(np.linalg.norm(self.cluster_heads[path[i]] - self.cluster_heads[path[i-1]]) for i in range(1, len(path)))
        if self.is_path_self_intersecting([self.cluster_heads[i] for i in path]):
            return 0  # Penalize self-intersecting paths
        return 1 / (total_distance + 1e-6)  # Minimize distance

    def is_path_self_intersecting(self, path):
        line = LineString(path)
        return not line.is_simple

    def optimize(self):
        for _ in range(DPSA_ITERATIONS):
            for i in range(self.population_size):
                new_solution = self.mutate(self.population[i])
                new_fitness = self.fitness(new_solution)
                if new_fitness > self.fitness(self.population[i]):
                    self.population[i] = new_solution
                    if new_fitness > self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = new_fitness

        return [self.cluster_heads[i] for i in self.best_solution]

    def mutate(self, path):
        new_path = path[:-1]  # Remove the last element (which is the same as the first)
        i, j = np.random.choice(len(new_path), 2, replace=False)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path + [new_path[0]]  # Add the first element to the end

def simulate_data_transmission(clusters, cluster_heads):
    data_at_cluster_heads = {tuple(head): 0 for head in cluster_heads}
    for cluster, head in zip(clusters, cluster_heads):
        data_at_cluster_heads[tuple(head)] = len(cluster) - 1  # -1 because the head doesn't send to itself
    return data_at_cluster_heads

def simulate_network(clusters, energy_levels):
    all_paths = []
    all_metrics = []

    for main_round in range(MAIN_ROUNDS):
        print(f"\nMain Round {main_round + 1}")
        cluster_heads = select_cluster_heads(clusters, energy_levels)
        
        data_at_cluster_heads = simulate_data_transmission(clusters, cluster_heads)
        
        dpsa = DPSA(cluster_heads, COMMUNICATION_RADIUS)
        optimized_path = dpsa.optimize()
        
        data_collected = sum(data_at_cluster_heads[tuple(head)] for head in optimized_path)
        path_length = sum(np.linalg.norm(optimized_path[i] - optimized_path[i-1]) for i in range(1, len(optimized_path)))
        
        metrics = {
            'Path Length': path_length,
            'Data Collected': data_collected,
        }
        
        all_paths.append(optimized_path)
        all_metrics.append(metrics)
        
        energy_levels = update_energy_levels(energy_levels, clusters, cluster_heads, optimized_path)
        
        print(f"  Path Length: {path_length:.2f}, Data Collected: {data_collected}")
        print(f"  Average remaining energy: {np.mean(energy_levels):.2f}")
        
        plot_final_path(clusters, cluster_heads, optimized_path, main_round+1, metrics)
    
    return all_paths, energy_levels, all_metrics

def update_energy_levels(energy_levels, clusters, cluster_heads, path):
    start_index = 0
    for cluster, head in zip(clusters, cluster_heads):
        cluster_size = len(cluster)
        cluster_energy = energy_levels[start_index:start_index + cluster_size]
        head_index = np.where((cluster == head).all(axis=1))[0][0]
        
        for i in range(cluster_size):
            if i != head_index:
                distance = np.linalg.norm(cluster[i] - head)
                cluster_energy[i] -= TRANSMISSION_TO_CLUSTER_HEAD_COST * distance
        
        cluster_energy[head_index] -= CLUSTER_HEAD_ENERGY_COST
        
        energy_levels[start_index:start_index + cluster_size] = cluster_energy
        start_index += cluster_size
    
    for i in range(len(path) - 1):
        current_pos = path[i]
        next_pos = path[i+1]
        distance = np.linalg.norm(next_pos - current_pos)
        energy_levels -= distance * ENERGY_CONSUMPTION_RATE / len(energy_levels)
    
    return np.maximum(energy_levels, 0.0)

def plot_final_path(clusters, cluster_heads, path, round_number, metrics):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all sensors
    for cluster in clusters:
        ax.scatter(cluster[:, 0], cluster[:, 1], c='gray', s=20, alpha=0.5)
    
    # Plot cluster heads
    cluster_head_coords = np.array(cluster_heads)
    ax.scatter(cluster_head_coords[:, 0], cluster_head_coords[:, 1], c='red', s=100, marker='^', label='Cluster Heads')
    
    # Plot the path of the base station
    path_coords = np.array(path)
    ax.plot(path_coords[:, 0], path_coords[:, 1], 'o-', color='green', markersize=8, linewidth=2, label='Base Station Path')
    ax.plot(path_coords[0, 0], path_coords[0, 1], 'go', markersize=12, label='Start/End Point')
    
    ax.set_title(f'Base Station Path (Cluster Heads Only) - Round {round_number}')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True)
    
    # Add metrics to the plot
    metrics_text = f"Path Length: {metrics['Path Length']:.2f}\nData Collected: {metrics['Data Collected']}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"base_station_path_round_{round_number}.png")
    plt.close()

def plot_metrics_over_rounds(all_metrics):
    metrics_to_plot = ['Path Length', 'Data Collected']
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 8))
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in all_metrics]
        axes[i].plot(range(1, len(values)+1), values, marker='o')
        axes[i].set_title(metric)
        axes[i].set_xlabel('Round')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig("metrics_over_rounds.png")
    plt.close()

if __name__ == "__main__":
    print("Starting simulation...")
    tiles = generate_spectre_tiles()
    clusters = place_sensors_and_identify_clusters(tiles)
    energy_levels = np.full(sum(len(cluster) for cluster in clusters), INITIAL_ENERGY, dtype=float)
    all_paths, remaining_energy, all_metrics = simulate_network(clusters, energy_levels)
    plot_metrics_over_rounds(all_metrics)
    print("Simulation completed.")