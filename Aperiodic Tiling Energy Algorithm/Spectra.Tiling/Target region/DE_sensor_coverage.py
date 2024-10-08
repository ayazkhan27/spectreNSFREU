# DE_sensor_coverage.py

import numpy as np
from matplotlib.patches import Circle, Polygon
from scipy.spatial import ConvexHull

class DE:
    def __init__(self, tiles, sensor_positions, target_area, pop_size, max_iter, F, CR, sensor_radius, initial_energy, energy_consumption_on, energy_consumption_off):
        self.tiles = tiles
        self.sensor_positions = sensor_positions
        self.target_area = target_area
        self.dim = len(sensor_positions)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.SENSOR_RADIUS = sensor_radius
        self.INITIAL_ENERGY = initial_energy
        self.ENERGY_CONSUMPTION_ON = energy_consumption_on
        self.ENERGY_CONSUMPTION_OFF = energy_consumption_off
        
        self.update_relevant_sensors()
        
        self.population = np.random.rand(pop_size, self.dim) < 0.5
        self.fitness = np.array([self.fitness_func(ind) for ind in self.population])
        
        self.best = self.population[np.argmax(self.fitness)]
        self.best_fitness = np.max(self.fitness)
        
        self.fitness_history = [self.best_fitness]
        self.coverage_history = [self.calculate_coverage(self.best)]

    def update_relevant_sensors(self):
        self.relevant_sensors = [
            i for i, sensor in enumerate(self.sensor_positions)
            if self.can_sensor_cover_target(sensor)
        ]
        self.relevant_mask = np.zeros(self.dim, dtype=bool)
        self.relevant_mask[self.relevant_sensors] = True

    def can_sensor_cover_target(self, sensor):
        return any(self.can_sensor_cover_point(sensor, point) for point in self.target_area)

    def can_sensor_cover_point(self, sensor, point):
        return np.sum((np.array(sensor) - np.array(point))**2) <= self.SENSOR_RADIUS**2

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
                if self.point_in_polygon((x, y), self.target_area):
                    total_points += 1
                    for i in self.relevant_sensors:
                        if individual[i]:
                            sensor_x, sensor_y = self.sensor_positions[i]
                            if (x - sensor_x)**2 + (y - sensor_y)**2 <= self.SENSOR_RADIUS**2:
                                covered_points.add((x, y))
                                break
        
        return len(covered_points) / total_points if total_points > 0 else 0

    @staticmethod
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

    def crossover(self, target, donor):
        trial = np.where(np.random.rand(self.dim) < self.CR, donor, target)
        return trial * self.relevant_mask

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                a, b, c = self.population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = a ^ np.logical_xor(b, c)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = self.fitness_func(trial)
                
                if trial_fitness > self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    
                    if trial_fitness > self.best_fitness:
                        self.best = trial
                        self.best_fitness = trial_fitness
            
            self.fitness_history.append(self.best_fitness)
            self.coverage_history.append(self.calculate_coverage(self.best))
            
            if iteration % 10 == 0:
                print(f"DE Iteration {iteration}, Best fitness: {self.best_fitness:.5f}")
        
        return self.best, self.best_fitness

def run_de(tiles, sensor_positions, target_area, pop_size, max_iter, F, CR, sensor_radius, initial_energy, energy_consumption_on, energy_consumption_off):
    de = DE(tiles, sensor_positions, target_area, pop_size, max_iter, F, CR, sensor_radius, initial_energy, energy_consumption_on, energy_consumption_off)
    best_solution, best_fitness = de.optimize()
    return best_solution, best_fitness, de.fitness_history, de.coverage_history