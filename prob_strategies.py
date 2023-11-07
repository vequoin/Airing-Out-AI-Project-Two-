import random
import math
from collections import deque

class Strategy3Bot:
    def __init__(self,ship, ship_size, alpha, initial_position):
        self.ship = ship
        self.ship_size = ship_size
        self.alpha = alpha
        self.position = initial_position  # This is now a tuple
        self.probability_grid = [[1/(len(ship.open_cells)) for _ in range(ship.get_length())] for _ in range(self.ship.get_length())]
        # Setting the initial position's probability to 0
        x, y = initial_position
        self.probability_grid[x][y] = 0
        self.actions = 0

    def update_position(self, new_position):
        # new_position is a tuple
        self.position = new_position
        self.update_probabilities()

    def update_probabilities(self):
        # Resetting the probability grid
        self.probability_grid = [[1/len(self.ship.get_open_cells()) for _ in range(self.ship_size)] for _ in range(self.ship_size)]
        x, y = self.position
        self.probability_grid[x][y] = 0
        # Update the probabilities based on the new position
        for i in range(self.ship_size):
            for j in range(self.ship_size):
                self.probability_grid[i][j] = self.calculate_probability((i, j))

    def calculate_probability(self, target):
        # cell is a tuple
        distance = self.bfs(self.position, target)
        # Apply the formula given in the specifications
        if distance == 1:
            return 1  # If the bot is immediately next to the leak, the probability of receiving a beep is 1
        else:
            return math.exp(-self.alpha * (distance - 1))

    def get_distance(self, start, target):
        # Both start and target are tuples
        start_x, start_y = start
        target_x, target_y = target
        return abs(start_x - target_x) + abs(start_y - target_y)

    def choose_next_move(self):
        # A method to decide the next move based on the probabilities
        # For example, you might choose the cell with the highest probability
        # This is just a placeholder for however you want to implement the decision logic
        max_probability = max(max(row) for row in self.probability_grid)
        candidate_cells = [
            (i, j) for i in range(self.ship_size) for j in range(self.ship_size)
            if self.probability_grid[i][j] == max_probability
        ]
        distance_to_candidates = []
        cell_to_visit = None
        min_distance = math.inf
        for candidate in candidate_cells:
            distance = self.bfs(self.position, candidate)
            if distance < min_distance:
                min_distance = distance
                cell_to_visit = candidate
        return cell_to_visit
            

    def print_probabilities(self):
        # Helper method to print the probability grid
        for row in self.probability_grid:
            # Formatting each probability as a float with 2 decimal places
            print(' '.join('{:.2f}'.format(cell) for cell in row))
            


    def bfs(self, start_position, target):
        queue = deque([(start_position, 0)])  # Queue holds tuples of (position, distance)
        visited = set([start_position])

        while queue:
            cell, curr_distance = queue.popleft()

            if cell == target:
                return curr_distance

            # Add unvisited neighbors to the queue
            for neighbor in self.ship.get_open_neighbors(cell):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, curr_distance + 1))

        return 0

