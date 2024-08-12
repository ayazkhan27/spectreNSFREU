# spectre_partitioning_with_mobile_base_station.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon, Circle
from matplotlib.colors import ListedColormap
from scipy.spatial import KDTree
from spectre import buildSpectreBase, transPt, buildSupertiles, SPECTRE_POINTS

# Parameters
GRID_RESOLUTION = 1
ENERGY_CONSUMPTION_RATE = 1
MAX_ITERATIONS = 2
SENSING_RADIUS = 1  # Example value for sensing radius
COMMUNICATION_RADIUS = 2 * SENSING_RADIUS  # Communication radius is twice the sensing radius

def calculate_sensor_radius(tile_points):
    longest_distance = max(np.linalg.norm(pt - np.mean(tile_points, axis=0)) for pt in tile_points)
    return longest_distance

def generate_spectre_tiles():
    tiles = buildSpectreBase()
    iterations = 0
    sensor_counts = []
    
    while iterations < MAX_ITERATIONS:
        tiles = buildSupertiles(tiles)
        sensor_positions = place_sensors_inscribed(tiles)
        sensor_counts.append(len(sensor_positions))
        iterations += 1
        print(f"Iteration {iterations} completed. Number of sensors: {len(sensor_positions)}")
    
    return tiles, sensor_counts

def place_sensors_inscribed(tiles):
    sensor_positions = []
    
    def add_sensor_points(transformation, label):
        nonlocal sensor_positions
        tile_points = [transPt(transformation, pt) for pt in SPECTRE_POINTS]
        centroid = np.mean(tile_points, axis=0)
        sensor_positions.append(centroid)
    
    tiles["Delta"].forEachTile(add_sensor_points)
    return sensor_positions

def calculate_overall_centroid(sensor_positions):
    return np.mean(sensor_positions, axis=0)

def identify_clusters(sensor_positions):
    num_sensors = len(sensor_positions)
    num_clusters = (num_sensors + 8) // 9  # Round up to the nearest multiple of 9
    clusters = []
    for i in range(num_clusters):
        clusters.append(sensor_positions[i * 9:min((i + 1) * 9, num_sensors)])
    return clusters

def select_cluster_heads(clusters):
    cluster_heads = []
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        closest_sensor = min(cluster, key=lambda pos: np.linalg.norm(pos - centroid))
        cluster_heads.append(closest_sensor)
    return cluster_heads

def calculate_cluster_centroids(clusters):
    cluster_centroids = []
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        cluster_centroids.append(centroid)
    return cluster_centroids

def simulate_mobile_base_station(cluster_centroids, cluster_heads):
    path = []
    for centroid in cluster_centroids:
        path.append(centroid)
        print(f"Base station moves to: {centroid}")
        for head in cluster_heads:
            if np.linalg.norm(centroid - head) <= COMMUNICATION_RADIUS:
                print(f"Cluster head at {head} is within communication range.")
            else:
                print(f"Cluster head at {head} is NOT within communication range.")
    return path

def plot_spectre_tiles_with_clusters(tiles, sensor_positions, sensor_radius, cluster_heads, base_station_path):
    fig, ax = plt.subplots(figsize=(15, 15))
    all_points = []

    def draw_tile(transformation, label):
        points = [transPt(transformation, pt) for pt in SPECTRE_POINTS]
        all_points.extend(points)
        polygon = mplPolygon(points, closed=True, fill=None, edgecolor='b')
        ax.add_patch(polygon)

    tiles["Delta"].forEachTile(draw_tile)

    for sensor_pos in sensor_positions:
        circle = Circle(sensor_pos, sensor_radius, color='r', fill=False, linestyle='dotted')
        ax.add_patch(circle)
        ax.plot(sensor_pos[0], sensor_pos[1], 'ko', markersize=2)
    
    # Plot the cluster heads
    for cluster_head in cluster_heads:
        ax.plot(cluster_head[0], cluster_head[1], 'bo', markersize=6, label='Cluster Head')

    # Plot the base station path
    for i, pos in enumerate(base_station_path):
        ax.plot(pos[0], pos[1], 'go', markersize=10, label=f'Base Station Position {i+1}')
        if i > 0:
            ax.plot([base_station_path[i-1][0], pos[0]], [base_station_path[i-1][1], pos[1]], 'g--')

    if all_points:
        all_points = np.array(all_points)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        plot_width = x_max - x_min + 20
        plot_height = y_max - y_min + 20

        ax.set_xlim(x_center - plot_width / 2, x_center + plot_width / 2)
        ax.set_ylim(y_center - plot_height / 2, y_center + plot_height / 2)

    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.title("Spectre Tile with Cluster Heads and Mobile Base Station Path")
    plt.savefig("spectre_with_mobile_base_station.png")
    plt.show()

# Generate spectre tiles and count sensors per iteration
tiles, sensor_counts = generate_spectre_tiles()

# Place sensors inscribed within each tile
sensor_positions = place_sensors_inscribed(tiles)
sensor_positions = np.array(sensor_positions)

# Calculate sensor radius
example_tile_points = [transPt(np.eye(3), pt) for pt in SPECTRE_POINTS]
SENSOR_RADIUS = calculate_sensor_radius(example_tile_points)

# Identify clusters and select cluster heads
clusters = identify_clusters(sensor_positions)
cluster_heads = select_cluster_heads(clusters)

# Calculate cluster centroids for the mobile base station path
cluster_centroids = calculate_cluster_centroids(clusters)

# Simulate the movement of the mobile base station
base_station_path = simulate_mobile_base_station(cluster_centroids, cluster_heads)

# Plot the spectre tiles with cluster heads and the mobile base station path
plot_spectre_tiles_with_clusters(tiles, sensor_positions, SENSOR_RADIUS, cluster_heads, base_station_path)
