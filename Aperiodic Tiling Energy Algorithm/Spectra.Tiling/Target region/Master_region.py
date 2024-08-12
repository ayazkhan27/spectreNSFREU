import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.patches import Circle, Polygon
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from spectre import buildSpectreBase, transPt, buildSupertiles, SPECTRE_POINTS
from DE_sensor_coverage import run_de
from PSO_sensor_coverage import run_pso

plt.style.use(['science', 'ieee'])

# Global constants
SENSOR_RADIUS = 10
GRID_RESOLUTION = 1
INITIAL_ENERGY = 100000  # 100000 watts
ENERGY_CONSUMPTION_ON = 1  # 1 watt when on
ENERGY_CONSUMPTION_OFF = 0.5  # 1/2 watt when off
FIGURE_DPI = 1000

def generate_spectre_tiles(n_iterations):
    print(f"Generating spectre tiles with {n_iterations} iterations...")
    tiles = buildSpectreBase()
    for _ in range(n_iterations):
        tiles = buildSupertiles(tiles)
    print("Spectre tiles generated.")
    return tiles

def place_sensors_inscribed(tiles):
    print("Placing sensors on the spectre tiles...")
    sensor_positions = []
    def add_sensor_points(transformation, label):
        nonlocal sensor_positions
        tile_points = [transPt(transformation, pt) for pt in SPECTRE_POINTS]
        centroid = np.mean(tile_points, axis=0)
        sensor_positions.append(centroid)
    tiles["Delta"].forEachTile(add_sensor_points)
    print(f"Total sensors placed: {len(sensor_positions)}")
    return np.array(sensor_positions)

def get_sensor_tiling_bounds(sensor_positions):
    min_x = min(sensor[0] for sensor in sensor_positions)
    max_x = max(sensor[0] for sensor in sensor_positions)
    min_y = min(sensor[1] for sensor in sensor_positions)
    max_y = max(sensor[1] for sensor in sensor_positions)
    return min_x, min_y, max_x, max_y

def generate_valid_target_area(tiles, sensor_positions):
    min_x, min_y, max_x, max_y = get_sensor_tiling_bounds(sensor_positions)
    
    while True:
        num_vertices = np.random.randint(4, 8)  # 4 to 7 vertices
        points = []
        for _ in range(num_vertices):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            points.append([x, y])
        
        hull = ConvexHull(points)
        target_area = [points[i] for i in hull.vertices]
        
        if is_target_area_valid(target_area, sensor_positions):
            area = calculate_polygon_area(target_area)
            if 3600 <= area <= 6400:  # 20x20 to 40x40
                return target_area

def is_target_area_valid(target_area, sensor_positions):
    # Check if all vertices can be covered
    if not all(any(can_sensor_cover_point(sensor, point) for sensor in sensor_positions) for point in target_area):
        return False
    
    # Check if the entire area can be covered
    if not can_cover_entire_shape(target_area, sensor_positions):
        return False
    
    # Check for line intersections
    if do_lines_intersect(target_area):
        return False
    
    return True

def can_sensor_cover_point(sensor, point):
    return np.sum((np.array(sensor) - np.array(point))**2) <= SENSOR_RADIUS**2

def can_cover_entire_shape(polygon, sensor_positions):
    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    
    for x in np.arange(min_x, max_x, 0.5):
        for y in np.arange(min_y, max_y, 0.5):
            if point_in_polygon((x, y), polygon):
                if not any(can_sensor_cover_point(sensor, (x, y)) for sensor in sensor_positions):
                    return False
    return True

def do_lines_intersect(polygon):
    n = len(polygon)
    for i in range(n):
        for j in range(i+2, n-1 if i == 0 else n):
            if line_intersection(polygon[i], polygon[(i+1)%n], polygon[j], polygon[(j+1)%n]):
                return True
    return False

def line_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def calculate_polygon_area(polygon):
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = abs(area) / 2.0
    return area

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class DPSA:
    def __init__(self, tiles, sensor_positions, target_area, pop_size, max_iter, w, c1, c2, F, CR, threshold):
        print("Initializing DPSA...")
        self.tiles = tiles
        self.sensor_positions = sensor_positions
        self.target_area = target_area
        self.dim = len(sensor_positions)
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # PSO and DE parameters
        self.w, self.c1, self.c2 = w, c1, c2
        self.F, self.CR = F, CR
        self.threshold = threshold
        
        self.energy_levels = np.full(self.dim, INITIAL_ENERGY, dtype=float)
        self.energy_history = []
        
        self.update_relevant_sensors()
        
        self.population = np.random.rand(pop_size, self.dim) < 0.5
        self.population[:, ~self.relevant_mask] = 0
        self.velocities = np.zeros((pop_size, self.dim))
        
        self.fitness = np.array([self.fitness_func(ind) for ind in self.population])
        
        self.pbest = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest = self.pbest[np.argmax(self.pbest_fitness)]
        self.gbest_fitness = np.max(self.pbest_fitness)
        
        self.fitness_history = [self.gbest_fitness]
        self.coverage_history = [self.calculate_coverage(self.gbest)]
        
        print(f"DPSA initialized. Initial best fitness: {self.gbest_fitness:.5f}")

    def update_relevant_sensors(self):
        self.relevant_sensors = [
            i for i, sensor in enumerate(self.sensor_positions)
            if self.can_sensor_cover_target(sensor)
        ]
        self.relevant_mask = np.zeros(self.dim, dtype=bool)
        self.relevant_mask[self.relevant_sensors] = True

    def can_sensor_cover_target(self, sensor):
        return  point_in_polygon(sensor, self.target_area)

    def fitness_func(self, individual):
        coverage = self.calculate_coverage(individual)
        active_relevant_nodes = np.sum(individual[self.relevant_mask])
        total_relevant_nodes = np.sum(self.relevant_mask)
        
        if total_relevant_nodes == 0:
            return 0  # No relevant sensors, worst fitness
        
        coverage_score = coverage
        efficiency_score = 1 - (active_relevant_nodes / total_relevant_nodes)
        
        return coverage_score * 0.85 + efficiency_score * 0.15

    def calculate_coverage(self, individual):
        covered_points = set()
        total_points = 0
        
        min_x = min(p[0] for p in self.target_area)
        max_x = max(p[0] for p in self.target_area)
        min_y = min(p[1] for p in self.target_area)
        max_y = max(p[1] for p in self.target_area)
        
        for x in np.arange(min_x, max_x, 0.5):
            for y in np.arange(min_y, max_y, 0.5):
                if point_in_polygon((x, y), self.target_area):
                    total_points += 1
                    for i in self.relevant_sensors:
                        if individual[i]:
                            sensor_x, sensor_y = self.sensor_positions[i]
                            if (x - sensor_x)**2 + (y - sensor_y)**2 <= SENSOR_RADIUS**2:
                                covered_points.add((x, y))
                                break
        
        return len(covered_points) / total_points if total_points > 0 else 0

    def update_particle(self, i):
        if self.fitness[i] < self.threshold * self.gbest_fitness:
            # Use PSO update
            r1, r2 = np.random.rand(2)
            self.velocities[i] = (self.w * self.velocities[i] +
                                self.c1 * r1 * np.logical_xor(self.pbest[i].astype(int), self.population[i].astype(int)).astype(float) +
                                self.c2 * r2 * np.logical_xor(self.gbest.astype(int), self.population[i].astype(int)).astype(float))
            prob = 1 / (1 + np.exp(-self.velocities[i]))
            new_state = (np.random.rand(self.dim) < prob).astype(int)
        else:
            # Use DE update
            a, b, c = self.population[np.random.choice(self.pop_size, 3, replace=False)]
            mutant = a.astype(float) + self.F * np.logical_xor(b.astype(int), c.astype(int)).astype(float)
            cross_points = np.random.rand(self.dim) < self.CR
            trial = np.where(cross_points, mutant, self.population[i].astype(float))
            new_state = (trial > 0.5).astype(int)
            # Ensure only relevant sensors can be activated
            self.population[i] = new_state * self.relevant_mask

    def update_energy_levels(self, individual):
        energy_consumption = np.where(individual == 1, ENERGY_CONSUMPTION_ON, ENERGY_CONSUMPTION_OFF)
        self.energy_levels -= energy_consumption
        self.energy_levels = np.maximum(self.energy_levels, 0)  # Ensure non-negative energy

    def optimize(self):
        print("Starting optimization...")
        same_fitness_count = 0
        last_best_fitness = float('-inf')
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                self.update_particle(i)
                
                new_fitness = self.fitness_func(self.population[i])
                
                if new_fitness > self.pbest_fitness[i]:
                    self.pbest[i] = self.population[i].copy()
                    self.pbest_fitness[i] = new_fitness
                
                if new_fitness > self.gbest_fitness:
                    self.gbest = self.population[i].copy()
                    self.gbest_fitness = new_fitness
                    same_fitness_count = 0
                else:
                    same_fitness_count += 1
                
                self.fitness[i] = new_fitness
            
            self.update_energy_levels(self.gbest)
            self.energy_history.append(np.sum(self.energy_levels))
            self.coverage_history.append(self.calculate_coverage(self.gbest))
            self.fitness_history.append(self.gbest_fitness)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best fitness: {self.gbest_fitness:.5f}, Network Energy: {self.energy_history[-1]:.2f}")
            
            if iteration % 100 == 0:
                self.plot_tiling_and_active_sensors(iteration)
            
            if iteration >= 150 and same_fitness_count >= 150:
                print(f"Stopping early at iteration {iteration} due to no improvement in fitness.")
                break
        
        self.plot_tiling_and_active_sensors(iteration)  # Plot final state
        print("Optimization completed.")
        return self.gbest, self.gbest_fitness

    def plot_tiling_and_active_sensors(self, iteration):
        print(f"Plotting tiling area and active sensors at iteration {iteration}...")
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Plot tiles
        def draw_tile(transformation, label):
            points = [transPt(transformation, pt) for pt in SPECTRE_POINTS]
            ax.add_patch(Polygon(points, closed=True, fill=None, edgecolor='k', alpha=0.3))
        self.tiles["Delta"].forEachTile(draw_tile)
        
        # Plot sensor positions and sensing radius
        for i, (sensor, is_active) in enumerate(zip(self.sensor_positions, self.gbest)):
            if is_active:
                ax.add_patch(Circle(sensor, SENSOR_RADIUS, fill=False, edgecolor='r', linestyle='-', linewidth=2, alpha=0.8))
                ax.plot(sensor[0], sensor[1], 'ro', markersize=5, alpha=0.8)
            else:
                ax.add_patch(Circle(sensor, SENSOR_RADIUS, fill=False, edgecolor='b', linestyle=':', linewidth=1, alpha=0.5))
                ax.plot(sensor[0], sensor[1], 'bo', markersize=3, alpha=0.5)
        
        # Plot target area
        target_polygon = Polygon(self.target_area, fill=False, edgecolor='g', linewidth=3, alpha=1.0)
        ax.add_patch(target_polygon)
        
        ax.set_title(f"DPSA at {iteration} iterations", fontsize=20)
        ax.set_xlabel("Width", fontsize=16)
        ax.set_ylabel("Height", fontsize=16)
        ax.set_aspect('equal', adjustable='box')
        
        # Add a legend
        active_sensor = plt.Line2D([], [], color='r', marker='o', linestyle='-', markersize=5, label='Active Sensor')
        inactive_sensor = plt.Line2D([], [], color='b', marker='o', linestyle=':', markersize=3, label='Inactive Sensor')
        target_area = plt.Line2D([], [], color='g', linestyle='-', linewidth=3, label='Target Area')
        ax.legend(handles=[active_sensor, inactive_sensor, target_area], loc='upper right', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'DPSA_at_{iteration}_iterations.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"DPSA plot saved as 'DPSA_at_{iteration}_iterations.png'")

def plot_fitness_comparison(dpsa_fitness, de_fitness, pso_fitness):
    plt.figure(figsize=(12, 8))
    plt.plot(dpsa_fitness, label='DPSA')
    plt.plot(de_fitness, label='DE')
    plt.plot(pso_fitness, label='PSO')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('Fitness Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_comparison.png', dpi=FIGURE_DPI)
    plt.close()

def plot_coverage_comparison(dpsa_coverage, de_coverage, pso_coverage):
    plt.figure(figsize=(12, 8))
    plt.plot(dpsa_coverage, label='DPSA')
    plt.plot(de_coverage, label='DE')
    plt.plot(pso_coverage, label='PSO')
    plt.xlabel('Iterations')
    plt.ylabel('Coverage')
    plt.title('Coverage Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('coverage_comparison.png', dpi=FIGURE_DPI)
    plt.close()

def main():
    print("Starting main function...")
    N_ITERATIONS = 3
    tiles = generate_spectre_tiles(N_ITERATIONS)
    sensor_positions = place_sensors_inscribed(tiles)
    target_area = generate_valid_target_area(tiles, sensor_positions)
    
    print(f"Number of sensors: {len(sensor_positions)}")
    
    try:
        print("Running DPSA...")
        dpsa = DPSA(tiles, sensor_positions, target_area, pop_size=300, max_iter=75,
                    w=0.6, c1=2.0, c2=2.0, F=0.7, CR=0.9, threshold=0.8)
        dpsa_solution, dpsa_fitness = dpsa.optimize()
        print(f"DPSA completed. Best fitness: {dpsa_fitness:.5f}")
    except Exception as e:
        print(f"Error running DPSA: {str(e)}")

    # Run DE
    try:
        print("Running DE...")
        de_solution, de_fitness, de_fitness_history, de_coverage_history = run_de(
            tiles, sensor_positions, target_area, pop_size=300, max_iter=75,
            F=0.7, CR=0.9, sensor_radius=SENSOR_RADIUS, initial_energy=INITIAL_ENERGY,
            energy_consumption_on=ENERGY_CONSUMPTION_ON, energy_consumption_off=ENERGY_CONSUMPTION_OFF
        )
        print(f"DE completed. Best fitness: {de_fitness:.5f}")
    except Exception as e:
        print(f"Error running DE: {str(e)}")

    # Run PSO
    try:
        print("Running PSO...")
        pso_solution, pso_fitness, pso_fitness_history, pso_coverage_history = run_pso(
            tiles, sensor_positions, target_area, pop_size=300, max_iter=75,
            w=0.6, c1=2.0, c2=2.0, sensor_radius=SENSOR_RADIUS, initial_energy=INITIAL_ENERGY,
            energy_consumption_on=ENERGY_CONSUMPTION_ON, energy_consumption_off=ENERGY_CONSUMPTION_OFF
        )
        print(f"PSO completed. Best fitness: {pso_fitness:.5f}")
    except Exception as e:
        print(f"Error running PSO: {str(e)}")

    # Plot comparisons only if all algorithms ran successfully
    if 'dpsa_fitness' in locals() and 'de_fitness' in locals() and 'pso_fitness' in locals():
        print("Plotting comparisons...")
        plot_fitness_comparison(dpsa.fitness_history, de_fitness_history, pso_fitness_history)
        plot_coverage_comparison(dpsa.coverage_history, de_coverage_history, pso_coverage_history)
    else:
        print("Skipping comparison plots due to incomplete results.")

    print("Main function completed.")

if __name__ == "__main__":
    main()