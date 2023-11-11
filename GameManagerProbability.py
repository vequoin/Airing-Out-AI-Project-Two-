import random
from collections import deque
from Ship import Ship
from prob_strategies import Strategy3Bot
import math
import numpy as np

class GameManager_Probability:
    def __init__(self, ship_size,bot_strategy, alpha):
        """
        Initialize the GameManager class which manages the game's overall state, 
        including the bot, leaks, ship layout, and game elements' positions.

        Args:
            ship_size (int): The size (dimension) of the ship.
            bot_strategy (function): The strategy the bot uses to navigate.
        """
        self.alpha = alpha
        
        
        self.ship = Ship(ship_size)  # Create a ship object with the given size.
        
        self.bot_position = random.choice(self.ship.open_cells)
        
        self.ship_size = ship_size  # Store the ship size.
        
        self.probability_grid = [['#' if cell == 1 else 1/len(self.ship.open_cells) for cell in row] for row in self.ship.ship]
        
        self.bot = Strategy3Bot(self.ship, self.bot_position)
        
        self.visited = {} 
        
        # Create leaks on the ship randomly generated based on bot strategy 
        self.leaks = []

         # Store the selected bot strategy.
        self.bot_strategy = bot_strategy  
        
        self.initialize_leaks()
        
        self.max_entropy = math.log(len(self.ship.open_cells))
        
        self.threshold_std_dev = .1
        self.waiting_factor = 2
        self.threshold_entropy = 0.1 * self.max_entropy
        self.waiting_threshold = 0.2  # Threshold for deciding when to wait
        self.recently_visited = set()
        
        
        
    def get_revisit_priority(self, cell, actions):
    # Assuming 'actions' is the number of actions taken so far
    # This gives a priority to cells that were visited longer ago
        return actions - self.visited.get(cell, 0)
        
        
    
    
    # Define a threshold for standard deviation above which we consider the distribution to be uncertain.
      # This could be 10% of the maximum probability for example, but should be tuned based on your system.
    
    # Define a threshold for entropy.
    # Entropy can range from 0 to log(n), where n is the number of possible outcomes (in your case, the number of open cells).
    # For simplicity, you might start with a fraction of the maximum entropy.
    # 10% of the maximum entropy
    
    # Define a waiting factor which determines how much we increase the waiting time based on uncertainty.
      # This means we double the waiting time for each unit of standard deviation above the threshold.
    
    
    def get_closest_cell(self, candidate_cells):
        # Find the closest candidate cell
        min_distance = float('inf')
        closest_cell = None
        for cell in candidate_cells:
            distance = self.bfs(self.bot.position, cell)
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell
        return closest_cell

    def choose_next_move_avoiding_backtrack(self):
        # Get the list of candidate cells as before
        candidate_cells = ...  # same as before
        
        # Filter out cells that have been recently visited
        unvisited_candidates = [cell for cell in candidate_cells if cell not in self.visited]
        
        if not unvisited_candidates:
            # All candidate cells have been visited, so we need to consider revisiting
            # Sort visited cells by the time since last visit (or by a revisit priority score)
            visited_candidates = sorted(candidate_cells, key=self.get_revisit_priority)
            # Choose the best cell to revisit
            next_move = visited_candidates[0]
            # Optionally, you can choose to remove this cell from the visited set
            # self.visited.remove(next_move)
        else:
            # Choose the closest unvisited candidate cell as before
            next_move = self.get_closest_cell(unvisited_candidates)
        
        # Mark the cell as visited and store the current action number
        self.visited[next_move] = self.actions
        
        return next_move
    
    def selectBot(self):
        if self.bot_strategy == 3:
            return 
        elif self.bot_strategy == 4:
            pass
        elif self.bot_strategy == 7:
            pass
        elif self.bot_strategy == 8:
            pass
        elif self.bot_strategy == 9:
            pass
    
    def initialize_leaks(self):
        open_cells = self.ship.open_cells.copy()
        
        open_cells.remove(self.bot_position) 

        num_leaks = 2 if self.bot_strategy in range(5, 10) else 1
        for _ in range(num_leaks):
            leak_position = random.choice(open_cells)
            self.leaks.append(leak_position)
        return True
    
    
    def strategy_nine(self):
        pass

        

############################################### Stategy 3 #####################################################

    
        
        
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
    


    def calculate_uncertainty(self):
        # Convert probabilities to a flat list excluding walls and visited cells
        probabilities = [prob for row in self.probability_grid for prob in row if isinstance(prob, float)]
        # Calculate the standard deviation or entropy as a measure of uncertainty
        std_dev = np.std(probabilities)
        # You can also calculate entropy as an alternative measure of uncertainty
        entropy = -np.sum([p * np.log(p) for p in probabilities if p > 0])
        return std_dev, entropy

    def dynamic_waiting(self, uncertainty):
        # Determine the number of waiting steps based on the level of uncertainty
        std_dev, entropy = uncertainty
        # This is a simple rule that could be replaced with something more sophisticated
        # The higher the uncertainty, the more we wait (sense)
        if std_dev > self.threshold_std_dev:
            return int(std_dev * self.waiting_factor)
        if entropy > self.threshold_entropy:
            return int(entropy * self.waiting_factor)
        return 1  # Default to one sensing action if uncertainty is low

        
        
    def calculate_probability(self, target):
        # cell is a tuple
        distance = self.bfs(self.bot.position, target)
        # Apply the formula given in the specifications
        if distance == 1:
            return 1  # If the bot is immediately next to the leak, the probability of receiving a beep is 1
        else:
            return math.exp(-self.alpha * (distance - 1))
        
        
        
    def update_probabilities(self, beep_detected):
        # Update the probabilities based on the presence or absence of a beep
        distances = self.bfs_all_distances(self.bot.position)
        
        for cell, distance in distances.items():
            x,y = cell
            if self.probability_grid[x][y] == 0:
                continue
            prob_beep = math.exp(-self.alpha * (distance - 1))
            if beep_detected:
                # print(f"prob bell is: {prob_beep}")
                # print(f"grid before update: {self.probability_grid[x][y]}")
                # if self.probability_grid[x][y] == 0:
                #     user = input("Probability is zero in update_probabilitites True")
                self.probability_grid[x][y] *= prob_beep
            else:
                # print(f"prob bell is: {prob_beep}")
                # print(f"grid before update: {self.probability_grid[x][y]}")
                self.probability_grid[x][y] *= (1 - prob_beep)
                # if self.probability_grid[x][y] == 0:
                #     user = input("Probability is zero in update_probabilitites False")
                
        self.normalize_probabilities()
        
    
    def normalize_probabilities(self):
        # Calculate the sum of all probabilities
        total_prob = 0
        for i in range(len(self.probability_grid)):  
            for j  in range(len(self.probability_grid)):
                if self.probability_grid[i][j] != "#":
                    total_prob += self.probability_grid[i][j]

        # Avoid division by zero
        if total_prob == 0:
            raise ValueError("Total probability in zero, cannot normalize.")

        # Divide each probability by the sum so that the sum of all probabilities will be 1
        for i in range(len(self.probability_grid)):
            for j in range(len(self.probability_grid[i])):
                if self.probability_grid[i][j] == "#" or self.probability_grid[i][j] == 0:
                    continue
                else:
                    self.probability_grid[i][j] /= total_prob
                    # if self.probability_grid[i][j] == 0:
                    #     user = input("Probability is zero in normalization of probabilities ")

    def initialize_probability_matrix(self):
        # Initialize a probability matrix for pairs of cells, each pair representing two potential leak locations
        probability_matrix = {}
        open_cells = self.ship.open_cells.copy()
        for i, cell1 in enumerate(open_cells):
            for cell2 in open_cells[i+1:]:
                probability_matrix[(cell1, cell2)] = 1.0 / self.num_possible_pairs()
        return probability_matrix

    def num_possible_pairs(self):
        # Calculate the number of possible unique pairs
        open_cells_count = len(self.ship.open_cells)
        return open_cells_count * (open_cells_count - 1) // 2

    def update_probability_matrix(self, beep_detected, current_position):
        new_probability_matrix = {}
        for cell_pair in self.probability_matrix:
            prob_leak1, prob_leak2 = self.calculate_beep_probabilities(cell_pair, current_position)
            if beep_detected:
                new_probability = self.probability_matrix[cell_pair] * (1 - (1 - prob_leak1) * (1 - prob_leak2))
            else:
                new_probability = self.probability_matrix[cell_pair] * (1 - prob_leak1) * (1 - prob_leak2)
            new_probability_matrix[cell_pair] = new_probability
        self.normalize_probability_matrix(new_probability_matrix)

    def normalize_probability_matrix(self, probability_matrix):
        # Normalize the probabilities in the matrix so they sum to 1
        total_probability = sum(probability_matrix.values())
        if total_probability == 0:
            raise ValueError("Total probability is zero, cannot normalize.")
        for cell_pair in probability_matrix:
            probability_matrix[cell_pair] /= total_probability
        self.probability_matrix = probability_matrix

    def calculate_beep_probabilities(self, cell_pair, current_position):
        # Calculate the probability of hearing a beep from each of the leaks in the cell pair
        return (
            self.calculate_probability_from_cell(current_position, cell_pair[0]),
            self.calculate_probability_from_cell(current_position, cell_pair[1])
        )

    def choose_next_move(self):
        # Find the maximum numerical probability
        max_probability = max(
            max(cell for cell in row if isinstance(cell, (int, float)))
            for row in self.probability_grid
        )
        print(f"Max probability: {max_probability}")

        # Compile a list of cells that have the maximum probability
        candidate_cells = [
            (i, j) for i in range(self.ship_size) for j in range(self.ship_size)
            if self.probability_grid[i][j] == max_probability
        ]
        
        #print(candidate_cells)

        # Find the closest candidate cell
        min_distance = math.inf
        
        #Breaking ties by distance 
        closest_candidates = sorted(candidate_cells, key=lambda cell: self.bfs(self.bot.position, cell))
        min_distance = self.bfs(self.bot.position, closest_candidates[0])

        # Further break ties randomly
        cells_with_min_distance = [cell for cell in closest_candidates if self.bfs(self.bot.position, cell) == min_distance]
        return random.choice(cells_with_min_distance)


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
    
    
    def bfs_all_distances(self, start_position):
        queue = deque([(start_position, 0)])  # Queue holds tuples of (position, distance)
        visited = {start_position: 0}  # Dictionary to keep track of visited cells and their distances

        while queue:
            cell, curr_distance = queue.popleft()

            # Add unvisited neighbors to the queue
            for neighbor in self.ship.get_open_neighbors(cell):
                if neighbor not in visited:
                    visited[neighbor] = curr_distance + 1
                    queue.append((neighbor, curr_distance + 1))

        return visited
    
    
    
    def sense(self):
        beep_probability = self.calculate_probability(self.leaks[0])
        num  = random.random()
        num /= 10
        #print(f"beep probability: {beep_probability}")
        #print(f"random is: {num}")
        #print(self)
        # if num <= beep_probability:
        #     print(f"beep probability: {beep_probability}")
        #     print(f"random is: {num}")
        #     print(self)
        #     i = input("Enter something...")
        beep_heard = num <= beep_probability
        return beep_heard
    
    
    def sense_two(self):
        for i in range(len(self.leaks)):
            beep_probability = self.calculate_probability(self.leaks[i])
            beep_heard = random.random() < beep_probability
            return beep_heard
        
        
    def play_game(self):
        curr_path = []
        actions = 0
        print(self)
        self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
        
        # Assuming that your bot's position and the leaks are all tuples now.
        while self.bot.position != self.leaks[0]:
            beep_heard = self.sense()
            print(f"Beep is: {beep_heard}")
            #print(self.probability_grid)
            # for i in range(len(self.probability_grid)):
            #     for j in range(len(self.probability_grid)):
            #         if self.probability_grid[i][j] != "#":
            #             if self.probability_grid[i][j] == 0:
            #                 print(f"Marked as zero:{(i,j)}")
            actions += 1
            self.update_probabilities(beep_heard)
            #print(self.probability_grid)
            # for i in range(len(self.probability_grid)):
            #     for j in range(len(self.probability_grid)):
            #         if self.probability_grid[i][j] != "#":
            #             if self.probability_grid[i][j] == 0:
            #                 print(f"Marked as zero: {i,j}")
            next_move = self.choose_next_move()
            curr_path = self.find_path_to_edge(self.bot.position, next_move)
            print(f"bot is at: {self.bot.position}")
            #print("This is the bot representation: ")
            #print(self)
            #print(f"Bot travelling to: {next_move}")
            #pit_stop = input("Press Enter.....")
            while curr_path:
                #print(f"bot position before move : {self.bot.position}")
                move = curr_path.pop(0)
                self.bot.move(move)
                self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
                self.normalize_probabilities()
                #print(f"bot position after move : {self.bot.position}")
                actions += 1   # Update the bot's position
                #print(self)
                #print(f"Bot travelling to: {move}")
                #pit_stop = input("Press Enter.....")
                if(move == self.leaks[0]):
                    return actions
        return actions
    
        # ... within the GameManager_Probability class ...
        
        
    def play_game_nine(self):
        # Implementation of the Bot 9 gameplay loop
        # This will involve moving the bot, updating probabilities, and possibly detecting leaks
        # based on the updated probability matrix
        actions = 0
        self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0

        # Main gameplay loop
        while len(self.leaks) > 0:
            # Choose the next move based on the probability matrix and not recently visited cells
            next_move = self.choose_next_move_avoiding_backtrack()
            
            # If the next move is the current position, the bot decides to wait
            if next_move == self.bot.position:
                self.wait_and_sense()
            else:
                # Move to the next cell and update the bot's position
                self.bot.move(next_move)
                self.update_position()
            
            # After moving, perform sensing
            beep_heard = self.sense()
            self.update_probabilities(beep_heard)
            
            # If a leak is found at the current position, patch it and update leaks list
            if self.bot.position in self.leaks:
                self.leaks.remove(self.bot.position)
                # Update the probability matrix to reflect that a leak has been patched
                self.update_probability_matrix_after_leak_patched(self.bot.position)
            
            actions += 1

            # Break the loop if the bot takes too many actions
            if actions > self.max_actions:
                break

        return actions
    
    def wait_and_sense(self):
        # Implement waiting based on the current uncertainty in the probability distribution
        uncertainty = self.calculate_uncertainty()
        waiting_steps = self.dynamic_waiting(uncertainty)
        for _ in range(waiting_steps):
            beep_heard = self.sense()
            self.update_probabilities(beep_heard)

    def play_game_seven(self):
        actions = 0
        self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
        while self.leaks:
            #print(self)
            beep_heard = self.sense()
            actions += 1
            self.update_probabilities(beep_heard)
            next_move = self.choose_next_move()
            path_to_next_move = self.find_path_to_edge(self.bot.position, next_move)
            while path_to_next_move:
                move = path_to_next_move.pop(0)
                self.bot.move(move)
                self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
                self.normalize_probabilities()
                actions += 1
                if move in self.leaks:
                    self.leaks.remove(move)  # Remove the found leak
                    self.normalize_probabilities() 
                    break
        return actions
    
    
    def play_game_four_two(self):
        curr_path = []
        actions = 0
        print(self)
        self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
        iswaiting = False
        # Assuming that your bot's position and the leaks are all tuples now.
        while self.bot.position != self.leaks[0]:
            uncertainty = self.calculate_uncertainty()
            wait_steps = self.dynamic_waiting(uncertainty)
            action_count = 3
            beep_heard = False
            print(f"Beep is: {beep_heard}")
     
            if not iswaiting:
                beep_heard = self.sense()
            elif iswaiting:
                for i in range(action_count):
                    beep_heard = self.sense()
                    actions += 1
                    if beep_heard:
                        iswaiting = False
                        break
            self.update_probabilities(beep_heard)
            #print(self.probability_grid)
            # for i in range(len(self.probability_grid)):
            #     for j in range(len(self.probability_grid)):
            #         if self.probability_grid[i][j] != "#":
            #             if self.probability_grid[i][j] == 0:
            #                 print(f"Marked as zero: {i,j}")
            next_move = self.choose_next_move()
            curr_path = self.find_path_to_edge(self.bot.position, next_move)
            print(f"bot is at: {self.bot.position}")
            #print("This is the bot representation: ")
            #print(self)
            #print(f"Bot travelling to: {next_move}")
            #pit_stop = input("Press Enter.....")
            while curr_path:
                beep = False
                #print(f"bot position before move : {self.bot.position}")
                move = curr_path.pop(0)
                self.bot.move(move)
                self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
                self.normalize_probabilities()
                if not iswaiting:
                    beep = self.sense()
                    actions += 1
                    self.update_probabilities(beep)
                if beep:
                    iswaiting = True                  
                    
                #print(f"bot position after move : {self.bot.position}")
                actions += 1   # Update the bot's position
                #print(self)
                #print(f"Bot travelling to: {move}")
                #pit_stop = input("Press Enter.....")
                if(move == self.leaks[0]):
                    return actions
        return actions
    
    
    def waiting(self):
        pass
    
    


    
    def run_game(self):
        if self.bot_strategy == 3:
          return self.play_game()
        if self.bot_strategy == 4:
            return self.play_game_four_two()
        if self.bot_strategy == 7:
           return self.play_game_seven()
        
    
    
    def __str__(self):
        grid_representation = ""
        for x in range(self.ship_size):
            for y in range(self.ship_size):
                if (x, y) == self.bot.position:
                    grid_representation += "B "  # B for Bot
                elif self.ship.ship[x][y] == 1:
                    grid_representation += "1 "  # 1 for Wall
                elif self.probability_grid[x][y] == 0:
                    grid_representation += "V "  # V for Visited
                elif (x, y) in self.leaks:
                    grid_representation += "L "  # L for Leak
                else:
                    grid_representation += f"_ "  # Probability for open space
            grid_representation += "\n"

        return grid_representation
