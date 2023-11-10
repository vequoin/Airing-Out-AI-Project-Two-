import math
import random
from collections import deque
from Ship import Ship
from bot import Bot

class GameManager:
    def __init__(self, ship_length, bot):
        self.probability_matrix = {}  # Hash Map
        self.initialize_probability_matrix(ship_length)
        self.bot_position = random.choice(self.ship.open_cells)
        self.ship_length = ship_length
        self.ship = Ship(self.ship_length)
        self.bot = Bot(self.ship, 8,)
        self.alpha = 0.5;
        self.leaks = []
        self.initialize_probability_matrix(self.ship_length)
         # Store the selected bot strategy.
        #self.bot_strategy = bot_strategy  
        
        self.initialize_leaks()
        self.play_game_eight()

    def initialize_probability_matrix(self, ship_length):
        num_cells = ship_length * ship_length
        total_pairs = num_cells * (num_cells - 1) // 2

        for i in range(ship_length):
            for j in range(ship_length):
                for k in range(ship_length):
                    for l in range(ship_length):
                        if (i, j) <= (k, l):
                            self.probability_matrix[((i, j), (k, l))] = 1 / total_pairs

    def initialize_leaks(self):
        open_cells = self.ship.open_cells.copy()
        
        open_cells.remove(self.bot_position) 

        num_leaks = 2
        for _ in range(num_leaks):
            leak_position = random.choice(open_cells)
            self.leaks.append(leak_position)
        return True
    
    def update_probabilities(self, beep, bot_position):
        for (cell1, cell2), probability in self.probability_matrix.items():
            if beep:
                beep_probabilities = self.calculate_beep_probability(cell1, cell2, bot_position)
                self.probability_matrix[(cell1, cell2)] = beep_probabilities
            else:
                no_beep_probabilities = 1 - self.calculate_beep_probability(cell1, cell2, bot_position)
                self.probability_matrix[(cell1, cell2)] = no_beep_probabilities

        self.normalize_probabilities()

    def calculate_beep_probability(self, cell1, cell2, bot_position):
        # Calculate individual beep probabilities for each cell
        probability_cell1 = self.calculate_individual_beep_probability(cell1, bot_position)
        probability_cell2 = self.calculate_individual_beep_probability(cell2, bot_position)

        # Calculate the probability of no beep for each cell
        no_beep_probability_cell1 = 1 - probability_cell1
        no_beep_probability_cell2 = 1 - probability_cell2

        # Calculate the probability of no beep for both cells
        no_beep_probability_both_cells = no_beep_probability_cell1 * no_beep_probability_cell2

        # Calculate the probability of beep for both cells
        beep_probability_both_cells = 1 - no_beep_probability_both_cells

        return beep_probability_both_cells

    
    def normalize_probabilities(self):
        total_probability = sum(self.probability_matrix.values())
        for key in self.probability_matrix:
            self.probability_matrix[key] /= total_probability

    def calculate_individual_beep_probability(self, cell, bot_position):
        distance = self.bfs(bot_position, cell)
    
        # If there's no path, return a default value (e.g., 0)
        if distance is None:
            return 0
        
        # Apply the formula given in the specifications
        if distance == 1:
            return 1  # If the bot is immediately next to the leak, the probability of receiving a beep is 1
        else:
            return math.exp(-self.alpha * (distance-1))
    
    def bfs(self,start_position, target):
        queue = deque([(start_position, 0)])  # Queue holds tuples of (position, distance)
        visited = set([start_position])
        print(start_position)
        print(target)
        while queue:
            cell, curr_distance = queue.popleft()

            if cell == target:
                print(f"target reached")
                return curr_distance

            # Add unvisited neighbors to the queue
            for neighbor in self.ship.get_open_neighbors(cell):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, curr_distance + 1))

        return None
    
    def play_game_eight(self):
        actions = 0
        while self.leaks:
                