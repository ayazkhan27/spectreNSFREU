import numpy as np
from Energy_consume import run_simulation, generate_networks, get_cloned_positions, simulate_clone_attack, has_reached_base_station
from network_generation import generate_aperiodic_network, generate_hexagonal_network, generate_triangular_network, generate_square_network

# Energy consumption parameters
ENERGY_IDLE = 0.1  # Energy consumed per round when idle
ENERGY_TRANSMIT = 1.0  # Energy consumed when transmitting
ENERGY_RECEIVE = 0.5  # Energy consumed when receiving
ENERGY_PROCESS = 0.2  # Energy consumed when processing (e.g., for intrusion detection)

def calculate_energy_consumption(network, paths, detected_clones, compromised_nodes):
    total_energy = 0
    
    # Energy for idle nodes
    total_energy += len(network) * ENERGY_IDLE
    
    # Energy for transmitting and receiving along paths
    for path in paths:
        total_energy += (len(path) - 1) * (ENERGY_TRANSMIT + ENERGY_RECEIVE)
    
    # Energy for processing at each compromised node
    total_energy += len(compromised_nodes) * ENERGY_PROCESS
    
    # Extra energy for intrusion detection at detected clone positions
    total_energy += len(detected_clones) * ENERGY_PROCESS
    
    return total_energy

def analyze_energy_consumption(num_sensors=71, num_iterations=1, num_rounds=10000):
    sensor_radius = 10  # This value is from the original code
    networks = generate_networks(sensor_radius, num_sensors)
    results = {network_type: {'total_energy': 0, 'rounds': num_rounds} for network_type in networks.keys()}
    
    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}")
        round_seed = round_idx
        clone_positions_per_network = {network_type: get_cloned_positions(network, round_seed, 0.1) for network_type, network in networks.items()}
        
        for network_type, network in networks.items():
            clone_positions = clone_positions_per_network[network_type]
            for _ in range(num_iterations):
                detections, paths, _, _, detected_clones, compromised_nodes = simulate_clone_attack(network, clone_positions, tuple(np.mean(network, axis=0)))
                
                # Calculate energy consumption for this round
                energy_consumed = calculate_energy_consumption(network, paths, set(detected_clones), set(compromised_nodes))
                results[network_type]['total_energy'] += energy_consumed
    
    # Calculate average energy consumption per round for each network type
    for network_type in results:
        avg_energy = results[network_type]['total_energy'] / results[network_type]['rounds']
        results[network_type]['avg_energy_per_round'] = avg_energy
        print(f"{network_type} Network - Average Energy Consumed per Round: {avg_energy:.2f}")
    
    return results

def compare_energy_efficiency(energy_results, num_sensors):
    for network_type, result in energy_results.items():
        coverage_area = num_sensors  # Assuming coverage is proportional to number of sensors
        efficiency = coverage_area / result['avg_energy_per_round']
        print(f"{network_type} Network - Energy Efficiency: {efficiency:.2f}")

if __name__ == "__main__":
    num_sensors = 71
    energy_results = analyze_energy_consumption(num_sensors=num_sensors, num_rounds=1000)
    compare_energy_efficiency(energy_results, num_sensors)