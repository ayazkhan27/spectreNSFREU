# File: clone_attack_detection_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import random
import pandas as pd
from network_generation import generate_aperiodic_network, generate_hexagonal_network, generate_triangular_network, generate_square_network

plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})

# Parameters
SENSOR_RADIUS = 10
COMMUNICATION_RANGE = SENSOR_RADIUS * 2
CLONE_PERCENTAGE = 0.01
DETECTION_THRESHOLD = 0.1  # Probability threshold for detecting a cloned node

def generate_networks(sensor_radius, num_sensors):
    aperiodic_network = generate_aperiodic_network(sensor_radius, num_sensors, 3)
    hexagonal_network = generate_hexagonal_network(num_sensors, sensor_radius)
    triangular_network = generate_triangular_network(num_sensors, sensor_radius)
    square_network = generate_square_network(num_sensors, sensor_radius)
    return {
        'Aperiodic': aperiodic_network,
        'Hexagonal': hexagonal_network,
        'Triangular': triangular_network,
        'Square': square_network
    }

def get_cloned_positions(network, seed, clone_percentage):
    random.seed(seed)
    num_clones = int(len(network) * clone_percentage)
    clone_indices = np.random.choice(len(network), num_clones, replace=False)
    return [tuple(network[idx]) for idx in clone_indices]

def simulate_clone_attack(network, clone_positions, base_station_position):
    detections = 0
    paths = []
    time_steps = 0
    total_hops = 0
    detected_clones = set()
    compromised_nodes = set(clone_positions)
    active_clones = set(clone_positions)

    while active_clones:
        new_active_clones = set()
        for clone_position in active_clones:
            if tuple(clone_position) in detected_clones or has_reached_base_station(clone_position, base_station_position):
                continue
            
            path = [clone_position]
            visited_nodes = set()
            while not has_reached_base_station(clone_position, base_station_position):
                visited_nodes.add(tuple(clone_position))
                next_position, step_time, pattern_found = smart_random_walk(network, clone_position, visited_nodes)
                if next_position is None:
                    break
                if np.linalg.norm(np.array(next_position) - np.array(clone_position)) > COMMUNICATION_RANGE:
                    break
                clone_position = next_position
                compromised_nodes.add(tuple(clone_position))
                path.append(clone_position)
                time_steps += step_time
                total_hops += 1
                if random.random() < DETECTION_THRESHOLD:
                    detected_clones.add(tuple(clone_position))
                    detections += 1
                    break
                if has_reached_base_station(clone_position, base_station_position):
                    break
            paths.append(path)
            if not has_reached_base_station(clone_position, base_station_position) and tuple(clone_position) not in detected_clones:
                new_active_clones.add(tuple(clone_position))
        
        active_clones = new_active_clones
    
    return detections, paths, time_steps, total_hops, detected_clones, len(compromised_nodes)

def smart_random_walk(network, intruder_position, visited_nodes):
    distances = np.linalg.norm(np.array(network) - np.array(intruder_position), axis=1)
    sorted_indices = np.argsort(distances)
    for idx in sorted_indices:
        nearest_node = network[idx]
        if tuple(nearest_node) not in visited_nodes and np.linalg.norm(nearest_node - intruder_position) <= COMMUNICATION_RANGE:
            step_time = calculate_time_step(nearest_node, intruder_position, network)
            pattern_found = detect_pattern(nearest_node, network)
            return nearest_node, step_time, pattern_found
    return None, 0, False

def calculate_time_step(nearest_node, current_node, network):
    distance = np.linalg.norm(np.array(nearest_node) - np.array(current_node))
    unique_angles, unique_distances = get_unique_angles_distances(current_node, network)
    complexity_factor = len(unique_angles) + len(unique_distances)
    return distance / SENSOR_RADIUS * complexity_factor

def get_unique_angles_distances(current_node, network):
    current_node = np.array(current_node)
    unique_angles = set()
    unique_distances = set()
    
    for node in network:
        node = np.array(node)
        vector = node - current_node
        distance = np.linalg.norm(vector)
        unique_distances.add(distance)
        
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        if angle < 0:
            angle += 360
        unique_angles.add(angle)
    
    return unique_angles, unique_distances

def detect_pattern(current_node, network):
    unique_angles, unique_distances = get_unique_angles_distances(current_node, network)
    
    max_angles = 6
    max_distances = 2
    
    return len(unique_angles) <= max_angles and len(unique_distances) <= max_distances

def has_reached_base_station(position, base_station_position):
    return np.linalg.norm(np.array(position) - np.array(base_station_position)) <= SENSOR_RADIUS

def plot_network_with_paths(network, paths, clone_positions, detected_clones, base_station_position, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    network = np.array(network)
    
    # Plot sensors and their ranges
    for node in network:
        sensor_circle = plt.Circle(node, SENSOR_RADIUS, color='blue', alpha=0.2)
        ax.add_artist(sensor_circle)
        plt.plot(node[0], node[1], 'bo', markersize=2, label='Uncompromised Nodes' if 'Uncompromised Nodes' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Plot base station
    plt.plot(base_station_position[0], base_station_position[1], 'go', markersize=10, label='Base Station')
    
    # Plot cloned nodes
    for pos in clone_positions:
        plt.plot(pos[0], pos[1], 'ro', markersize=5, label='Cloned Nodes' if 'Cloned Nodes' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Plot detected cloned nodes
    for pos in detected_clones:
        plt.plot(pos[0], pos[1], 'yo', markersize=5, label='Detected Cloned Nodes' if 'Detected Cloned Nodes' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Plot paths
    for path in paths:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1, alpha=0.5)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def run_simulation(num_sensors=559, num_iterations=1, num_rounds=10):
    sensor_radius = SENSOR_RADIUS
    networks = generate_networks(sensor_radius, num_sensors)
    results = {network_type: [] for network_type in networks.keys()}
    
    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}")
        round_seed = round_idx  # Use round index as the seed for consistency
        clone_positions_per_network = {network_type: get_cloned_positions(network, round_seed, CLONE_PERCENTAGE) for network_type, network in networks.items()}
        
        for network_type, network in networks.items():
            clone_positions = clone_positions_per_network[network_type]
            for iteration in range(num_iterations):
                detections, paths, time_steps, total_hops, detected_clones, compromised_nodes = simulate_clone_attack(network, clone_positions, tuple(np.mean(network, axis=0)))
                results[network_type].append({
                    'round': round_idx,
                    'detections': detections,
                    'paths': paths,
                    'time_steps': time_steps,
                    'total_hops': total_hops,
                    'base_station_reached': has_reached_base_station(paths[-1][-1], tuple(np.mean(network, axis=0))),
                    'detected_clones': len(detected_clones),
                    'compromised_nodes': compromised_nodes
                })
            if round_idx == 0 and iteration == 0:
                plot_network_with_paths(network, results[network_type][-1]['paths'], clone_positions, detected_clones, tuple(np.mean(network, axis=0)), f'{network_type} Network')
    
    plot_metrics(results, num_rounds)

def plot_metrics(results, num_rounds):
    metrics = ['time_steps', 'total_hops', 'base_station_reached', 'compromised_nodes', 'detections']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for network_type, network_results in results.items():
            y_values = [result[metric] for result in network_results]
            plt.plot(range(num_rounds), y_values, marker='o', linestyle='-', label=network_type)
        plt.title(f'{metric.replace("_", " ").title()} for Each Topology Over {num_rounds} Rounds', fontsize=16, fontweight='bold')
        plt.xlabel('Round', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        plt.legend(title='Network Types', title_fontsize='12', fontsize='10', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_simulation(num_iterations=1, num_rounds=10)
