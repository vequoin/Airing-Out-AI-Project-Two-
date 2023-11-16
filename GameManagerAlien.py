from bot import Bot
from Ship import Ship
import itertools
import random
from collections import deque


class GameManagerAlien():
    
    def __init__(self, ship_size, bot_strategy, k) -> None:
        self.ship = Ship(ship_size)  # Create a ship object with the given size.
        
        self.ship_length = ship_size  # Store the ship size.
        
        self.k = k
        
        self.alpha = k
        
        # Initialize positions for the bot, fire, and button.
        self.bot_position = random.choice(self.ship.open_cells)
        
        self.covered_grid = []

         # Store the selected bot strategy.
        self.bot_strategy = bot_strategy  

        # Create a Bot instance initialized with the ship, its strategy,
        self.bot = Bot(self.ship, bot_strategy, self.bot_position, k)

        # Initialize a set to keep track of nodes (positions) that have been visited during pathfinding
        self.visited_nodes = set()
        
        self.move_pattern = {}
        
        self.intruders = []
        
        self.probability_grid = [['#' if cell == 1 else 1/len(self.ship.open_cells) for cell in row] for row in self.ship.ship]
        
        #self.knowledge_grid = [['UNKNOWN' for _ in range(ship_size)] for _ in range(ship_size)]
        self.knowledge_grid = [['#' if cell == 1 else 'UNKNOWN' for cell in row] for row in self.ship.ship]
        
        self.initialize_intruders()
        
        
    def sense(self):
      sensed_grid = self.generate_sense_grid()
      if self.intruders[0] in sensed_grid:
          for cell in sensed_grid: 
            self.knowledge_grid[cell[0]][cell[1]] = "MIGHT_HAVE_INTRUDER"
          return sensed_grid
      else:
          for cell in sensed_grid:
              self.knowledge_grid[cell[0]][cell[1]] = "NOT DETECTED"
          return None    
        
    def initialize_intruders(self):
        open_cells = self.ship.open_cells.copy()
        
        open_cells.remove(self.bot_position) 

        num_leaks = 2 if self.bot_strategy in range(5, 10) else 1
        for _ in range(num_leaks):
            leak_position = self.get_intruders_position()
            if not leak_position:
                return False
            self.intruders.append(leak_position)
        return True
            
    def get_intruders_position(self):
    # Extract bot's x and y coordinates
        x, y = self.bot.position

        # Determine the detection square bounds
        left_bound = x - self.k
        right_bound = x + self.k
        upper_bound = y - self.k
        lower_bound = y + self.k

        # Get all open cells from the ship
        all_open_cells = set(self.ship.get_open_cells())
    

        # Collect cells that are inside the detection square
        detection_square_cells = {
            (x, y) for x in range(left_bound, right_bound + 1) 
                   for y in range(upper_bound, lower_bound + 1) 
                   if (x, y) in all_open_cells
        }
        
        potential_leak_positions = all_open_cells - detection_square_cells
        if potential_leak_positions:
        # Randomly select a leak position
            leak_position = random.choice(list(potential_leak_positions))
            return leak_position
        else:
            # Handle the case where there are no potential leak positions
            # For example, return None or raise a custom exception
            return None
        
        
        
    def generate_sense_grid(self):
        x, y = self.bot.position

        # Determine the detection square bounds
        left_bound = x - self.k
        right_bound = x + self.k
        upper_bound = y - self.k
        lower_bound = y + self.k

        # Get all open cells from the ship
        all_open_cells = set(self.ship.get_open_cells())

        # Collect cells that are inside the detection square
        return {
            (x, y) for x in range(left_bound, right_bound + 1) 
                   for y in range(upper_bound, lower_bound + 1) 
                   if (x, y) in all_open_cells
        }
        
    def move_intruder(self):
        return random.choice(self.ship.get_open_neighbors(self.intruders[0]))
    
    
    def strategy_explore(self):
        return random.choice(self.ship.open_cells)
    
    def find_path_to_edge(self, start_position, edge_cell):
        queue = deque([(start_position, [])])  # Each item is a tuple (position, path_so_far)
        visited = set([start_position])

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == edge_cell:
                return path  # Return the path once we've reached the edge cell

            # Add unvisited neighbors to the queue
            for neighbor in self.ship.get_open_neighbors((x, y)):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)

        return []  # Return an empty list if no path is found
    
    
    def analyze_sensed_movement(self, previous_grid, current_grid):
        movement_pattern = {}
        for x in range(self.ship_length):
            for y in range(self.ship_length):
                if current_grid[x][y] == "MIGHT_HAVE_INTRUDER" and previous_grid[x][y] != "MIGHT_HAVE_INTRUDER":
                    movement_pattern[(x, y)] = "new_sighting"
                elif current_grid[x][y] == "MIGHT_HAVE_INTRUDER" and previous_grid[x][y] == "MIGHT_HAVE_INTRUDER":
                    movement_pattern[(x, y)] = "consistent_sighting"
        return movement_pattern
        
            
        
    def strategy_bot_two(self):
        Moves = 0
        curr_path = []
        isExploring = True
        isHunting = False
        while self.bot.position != self.intruders[0]:
            num_not_detect = sum([cell for cell in self.knowledge_grid if cell == "NOT DETECTED"])
            num_U = sum([cell for cell in self.knowledge_grid if cell == "UNKNOWN"])
            print(f"Moves is: {Moves}")
            self.intruders[0] = self.move_intruder()
            if self.bot.position == self.intruders[0]:
                print("Intruder caught")
                return Moves
            if isExploring:
                if not curr_path or Moves % 5 == 0:
                    sensed = self.sense()
                    Moves += 1
                    if sensed:
                        isHunting = True
                        isExploring = False
                    else:
                        if num_not_detect > num_U*2:
                            curr_path = self.find_path_to_edge(self.bot.position, random.choice(self.ship.open_cells))
                        else:
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")
                        
            if isHunting:
                if Moves % 2 == 0:
                    previous_grid = [row[:] for row in self.knowledge_grid]
                    sensed = self.sense()
                    Moves += 1
                    movement_pattern = self.analyze_sensed_movement(previous_grid, self.knowledge_grid)
                    if movement_pattern:
                        self.move_pattern = movement_pattern.copy()
                    # Update path based on movement_pattern
                    next_target = self.get_next_target_based_on_movement_pattern(movement_pattern)
                    if not next_target:
                        next_target = self.get_next_target_based_on_sensed_data()
                    if not next_target:
                        next_target = self.get_next_target_based_on_movement_pattern(self.move_pattern)
                    if not next_target:
                        if num_not_detect > num_U*2:
                            next_target = random.choice(self.ship.open_cells)
                        else:
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")
                    if not curr_path:
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                        if not curr_path:
                            isExploring = True     
            if curr_path:
                next_move = curr_path.pop(0)
                
                self.bot.move(next_move)
                Moves += 1
                if self.bot.position == self.intruders[0]:
                    print("Intruder caught!")
                    return Moves
        return Moves

    def get_next_target_based_on_sensed_data(self):
        return self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")

    def get_next_target_based_on_movement_pattern(self, movement_pattern):
        
        new_sightings = [pos for pos, status in movement_pattern.items() if status == "new_sighting"]
        if new_sightings:
            return min(new_sightings, key=lambda cell: self.manhattan_distance(self.bot.position, cell))

        # If no new sightings, consider consistent sightings
        consistent_sightings = [pos for pos, status in movement_pattern.items() if status == "consistent_sighting"]
        if consistent_sightings:
            return min(consistent_sightings, key=lambda cell: self.manhattan_distance(self.bot.position, cell))
        # If no sightings, return None
        return None
    
    def manhattan_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    
    def find_path_to_nearest_cell_with_status(self, start_position, target_status):
        queue = [(start_position, [])]  # Each item is a tuple (position, path_so_far)
        visited = set([start_position])

        while queue:
            (x, y), path = queue.pop(0)
            #print(f"Visiting: {(x, y)} with path: {path}")  # Debug print
            if self.knowledge_grid[x][y] == target_status:
                #print(f"Found target at: {(x, y)} with path: {path}")
                return path

            # Add unvisited neighbors to the queue
            for neighbors in self.ship.get_open_neighbors((x,y)):
                if neighbors not in visited:
                    new_path = path + [neighbors]
                    queue.append((neighbors, new_path))
                    visited.add(neighbors)
        #print(f"No path found from {start_position} to a cell with status '{target_status}'")
        return []  # Return an empty list if no cell with desired status is found
    
    
    def print_ship_state(self):
        print("Ship State:")
        for y in range(self.ship_length):
            row_str = ''
            for x in range(self.ship_length):
                if self.ship.ship[x][y] == 1:
                    row_str += '# '  # Wall or obstacle
                elif (x, y) == self.bot.position:
                    row_str += 'B '  # Bot's position
                elif (x, y) in self.intruders:
                    row_str += 'I '  # Intruder's position
                else:
                    row_str += '. '  # Open space
            print(row_str)
        print("\n")
    
        
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

        return None
    
        
    def calculate_probability(self, target):
       # cell is a tuple
       distance = self.bfs(self.bot.position, target)
       # Apply the formula given in the specifications
       if distance == 1:
           return 1  # If the bot is immediately next to the leak, the probability of receiving a beep is 1
       else:
           return math.exp(-self.alpha * (distance - 1))
       
        
        
    def play_game_four_two(self):
        actions = 0
        curr_path = []
        while self.bot.position != self.intruders[0]:
            beep_heard = self.sense()
            self.update_probabilities_intruder(beep_heard)
            next_target = self.choose_next_move_intruder()
            curr_path = self.pathfinding_choice_intruder(next_target)

            while curr_path:
                move = curr_path.pop(0)
                self.bot.move(move)
                actions += 1
                if self.bot.position == self.intruders[0]:
                    print("Intruder caught!")
                    return actions

        return actions
    
    def pathfinding_choice_intruder(self, target):
        return self.find_path_to_edge(self.bot.position, target)


    def choose_next_move_intruder(self):
        max_prob = max(max(row) for row in self.probability_grid)
        candidate_cells = [(i, j) for i in range(self.ship_length) for j in range(self.ship_length) 
                           if self.probability_grid[i][j] == max_prob]

        # Break ties by choosing the closest cell
        return min(candidate_cells, key=lambda cell: self.manhattan_distance(self.bot.position, cell))



    def update_probabilities_intruder(self, beep_heard):
        for x in range(self.ship_length):
            for y in range(self.ship_length):
                cell = (x, y)
                prob_detect = self.calculate_probability(cell)

                if beep_heard:
                    self.probability_grid[x][y] *= prob_detect
                else:
                    self.probability_grid[x][y] *= (1 - prob_detect)

        self.normalize_probabilities_intruder()


    def normalize_probabilities_intruder(self):
        total_prob = sum(sum(row) for row in self.probability_grid)
        for i in range(self.ship_length):
            for j in range(self.ship_length):
                if self.ship.ship[i][j] != 1:
                    self.probability_grid[i][j] /= total_prob



    
    def run_game(self):
        if self.bot_strategy == 2:
            return self.strategy_bot_two()
        if self.bot_strategy == 4:
            return self.play_game_four_two()
        if self.bot.strategy == 9:
            return 
        
