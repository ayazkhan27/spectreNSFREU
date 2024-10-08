# intruder_attack_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from network_generation import generate_aperiodic_network, generate_hexagonal_network, generate_triangular_network, generate_square_network
import random

# Parameters
SENSOR_RADIUS = 10
HOP_DISTANCE = SENSOR_RADIUS

def simulate_intruder_attack(network, intruder_position, base_station_position, network_type):
    path = [intruder_position]
    time_steps = 0
    visited_nodes = set()
    is_aperiodic = network_type == 'aperiodic'
    while not has_reached_base_station(intruder_position, base_station_position):
        visited_nodes.add(tuple(intruder_position))
        intruder_position, step_time = smart_random_walk(network, intruder_position, visited_nodes, is_aperiodic)
        path.append(intruder_position)
        time_steps += step_time
    return path, time_steps

def smart_random_walk(network, intruder_position, visited_nodes, is_aperiodic):
    distances = np.linalg.norm(np.array(network) - np.array(intruder_position), axis=1)
    sorted_indices = np.argsort(distances)
    for idx in sorted_indices:
        nearest_node = network[idx]
        if tuple(nearest_node) not in visited_nodes:
            step_time = calculate_time_step(nearest_node, intruder_position, network, is_aperiodic)
            return nearest_node, step_time
    return intruder_position, 0

def calculate_time_step(nearest_node, current_node, network, is_aperiodic):
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

def has_reached_base_station(position, base_station_position):
    return np.linalg.norm(np.array(position) - np.array(base_station_position)) <= SENSOR_RADIUS

def run_simulation(num_iterations=10):
    random.seed()  # Ensure randomness in each simulation run

    global SENSOR_RADIUS
    num_sensors = 559  # Example value
    sensor_radius = SENSOR_RADIUS

    # Generate networks
    aperiodic_network = generate_aperiodic_network(sensor_radius, num_sensors, 3)
    hexagonal_network = generate_hexagonal_network(num_sensors, sensor_radius)
    triangular_network = generate_triangular_network(num_sensors, sensor_radius)
    square_network = generate_square_network(num_sensors, sensor_radius)

    # Base station positions (center of the network)
    aperiodic_base_station = tuple(np.mean(aperiodic_network, axis=0))
    hexagonal_base_station = tuple(np.mean(hexagonal_network, axis=0))
    triangular_base_station = tuple(np.mean(triangular_network, axis=0))
    square_base_station = tuple(np.mean(square_network, axis=0))

    network_types = ['Aperiodic', 'Hexagonal', 'Triangular', 'Square']
    networks = [aperiodic_network, hexagonal_network, triangular_network, square_network]
    base_stations = [aperiodic_base_station, hexagonal_base_station, triangular_base_station, square_base_station]
    
    results = {network_type: [] for network_type in network_types}
    
    for i in range(num_iterations):
        for network_type, network, base_station in zip(network_types, networks, base_stations):
            intruder_initial_position = (random.uniform(-200, 200), random.uniform(-200, 200))
            path, time_steps = simulate_intruder_attack(network, intruder_initial_position, base_station, network_type)
            results[network_type].append(time_steps)
        print(f"Iteration {i + 1} completed.")

    # Calculate average time steps
    avg_time_steps = {network_type: np.mean(results[network_type]) for network_type in network_types}

    # Plot results
    plot_results(avg_time_steps)

def plot_results(avg_time_steps):
    fig, ax = plt.subplots()
    network_types = list(avg_time_steps.keys())
    avg_times = list(avg_time_steps.values())
    
    ax.bar(network_types, avg_times, color=['red', 'green', 'blue', 'purple'])
    
    ax.set_xlabel('Network Topology')
    ax.set_ylabel('Average Time Steps')
    ax.set_title('Average Time Steps for Different Network Topologies')
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
