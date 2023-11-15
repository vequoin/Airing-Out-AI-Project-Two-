import heapq
import random
from collections import deque
from Ship import Ship
from Strategy3Bot import Strategy3Bot
import math
import itertools
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
        
        self.bot = Strategy3Bot(self.ship, self.bot_position)
        
        self.visited = set()
        
        # Create leaks on the ship randomly generated based on bot strategy 
        self.leaks = []

         # Store the selected bot strategy.
        self.bot_strategy = bot_strategy 
        
        self.probability_grid = self.initialize_probability_grid() if self.bot_strategy in range(3,8) else self.initialize_probability_grid_multiple()
        
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
    
    
    def initialize_probability_grid(self):
        return [['#' if cell == 1 else 1/len(self.ship.open_cells) for cell in row] for row in self.ship.ship]
        
        
    
    
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
    

    def find_path(self, start_position, edge_cell, avoiding):
        queue = deque([(start_position, [])])  # Each item is a tuple (position, path_so_far)
        visited = set([start_position])

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == edge_cell:
                return path  # Return the path once we've reached the edge cell

            # Add unvisited neighbors to the queue
            if avoiding:
                for neighbor in self.ship.get_open_neighbors((x, y)):
                    if neighbor not in visited and neighbor not in self.visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        visited.add(neighbor)
            else:
                 for neighbor in self.ship.get_open_neighbors((x, y)):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        visited.add(neighbor)

        return []
        
        
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


    def pathfinding_choice(self,target):
        path = self.find_path(self.bot.position,target, True)
        path_two = path = self.find_path(self.bot.position,target, False)
        if not path:
            return path_two
        if len(path) < 1.5*len(path_two):
            return path_two
        else:
            return path 


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
        beep_heard = num <= beep_probability
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

    
    
    #########################################  Bot Strategy 4 ####################################################
    
    
    def play_game_four_two(self):
        curr_path = []
        actions = 0
        #print(self)
        self.probability_grid[self.bot.position[0]][self.bot.position[1]] = 0
        self.visited.add(self.bot.position)
        iswaiting = False
        # Assuming that your bot's position and the leaks are all tuples now.
        while self.bot.position != self.leaks[0]:
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
            #               `print(f"Marked as zero: {i,j}")
            next_target = self.choose_next_move()
            curr_path = self.pathfinding_choice(next_target)
            #curr_path = self.find_path_to_edge(self.bot.position, next_target)
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
                self.visited.add(move)
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
                    print(f"Leak found and removed at {move}. Remaining leaks: {self.leaks}")
                    return actions
        return actions
    
    
    
    ########################################  Bot Strategy 7 ####################################################        

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
    
    
    
    
    ############################################ strategy 8 ##################################################
    
    
    def initialize_probability_grid_multiple(self):
        grid = {}
        open_cells = self.ship.open_cells
        for cell1, cell2 in itertools.combinations(open_cells, 2):
            grid[(cell1, cell2)] = 1 / len(open_cells) ** 2
        return grid
    
    def sense_mutiple(self):
        beep_probability = 0
        beep_probabilities = []
        for leak in self.leaks:
            beep_probability = self.calculate_probability(leak)
            beep_probabilities.append(beep_probability)
        beep_probability = max(beep_probabilities)  # Cap at 1
        num = random.random()
        num /= 10
        return num <= beep_probability
    
    
    def precalculate_probabilities(self, distances):
        prob = {}
        for cell, distance in distances.items():
            prob[cell] = math.exp(-self.alpha * (distance - 1))
        return prob
            
    
    
    def update_probabilities_multiple(self, beep_heard):
        distances = self.bfs_all_distances(self.bot.position)
        all_probabilities = self.precalculate_probabilities(distances)
        
        for ((cell_j), (cell_k)), probability in self.probability_grid.items():
            if probability == 0:
                continue
            probability_cell_j = all_probabilities[cell_j]
            probability_cell_k = all_probabilities[cell_k]
            if beep_heard:
                new_prob = 1 - (1 - probability_cell_j) * (1 - probability_cell_k)
                self.probability_grid[(cell_j, cell_k)] = new_prob * probability
            else:
                new_prob = (1 - probability_cell_j) * (1 - probability_cell_k)
                self.probability_grid[(cell_j, cell_k)] = new_prob * probability
                
        self.normalize_probabilities_multi()
        
        
    def normalize_probabilities_multi(self):
        
        total_probability = sum(self.probability_grid.values())
        
        for key in self.probability_grid:
            self.probability_grid[key] /= total_probability
            
            
    def choose_next_move_multi(self):
        # Find the maximum numerical probability
        max_probability = max(self.probability_grid.values())
        
        print(f"Max probability: {max_probability}")

        # Compile a list of cells that have the maximum probability
        candidate_cells = [
            key for key, val in self.probability_grid.items() if val == max_probability
        ]
        
        #print(candidate_cells)
         # Calculate BFS distance once and store it
        distances = {cell_pair: min(self.bfs(self.bot.position, cell_pair[0]), self.bfs(self.bot.position, cell_pair[1])) for cell_pair in candidate_cells}
        # Find the closest candidate cell
        min_distance = math.inf
        
        #Breaking ties by distance 
        # Sort by distance
        closest_candidates = sorted(candidate_cells, key=lambda cell_pair: distances[cell_pair])

        min_distance = distances[closest_candidates[0]]

        # Filter cells with minimum distance
        cells_with_min_distance = [cell for cell in closest_candidates if distances[cell] == min_distance]
        choice  = random.choice(cells_with_min_distance)
        #print(f"target choice is: {choice}")
        calc_dis_one = self.bfs(self.bot.position, choice[0])
        calc_dis_two = self.bfs(self.bot.position, choice[1])
        if calc_dis_one < calc_dis_two:
            return choice[0]
        return choice[1]
    
    
    def update_probabilities_for_visited(self, visited):
        #print(f"Updating probabilities for visited cell: {visited}")
        total = 0
        for paired_cell, val in self.probability_grid.items():
            if visited in paired_cell:
                self.probability_grid[paired_cell] = 0
                #print(f"Setting pair {paired_cell} to zero")  # Debugging print

        # Count the number of pairs set to zero
        #total_zero_pairs = sum(1 for val in self.probability_grid.values() if val == 0)
        #print(f"Total pairs set to zero: {total_zero_pairs}")

        self.normalize_probabilities_multi()

            
            
    def strategy_eight(self):
        actions = 0
        beep_heard = self.sense_mutiple()
        print(f"bot postion is: {self.bot.position}")
        print(f"total moves: {actions}")
        visited = self.bot.position
        self.update_probabilities_for_visited(visited)
        self.update_probabilities_multiple(beep_heard)
        print("after update multiple")
        target = self.choose_next_move_multi()
        curr_path = self.find_path_to_edge(self.bot.position, target)
        print(f"leaks are in {self.leaks}")
        #i = input("Enter something ...")
        while self.leaks:
            #total = 0
            #total_zero_pairs = sum(1 for val in self.probability_grid.values() if val == 0)
            # print(f"total cells in probability grid: {len(self.probability_grid)}")
            # print(f"total number of zero cells: {total_zero_pairs}")
            # print(f"bot postion is: {self.bot.position}")
            
            if actions > 2000:
                total_zero_pairs = sum(1 for val in self.probability_grid.values() if val == 0)
                print(f"total cells in probability grid: {len(self.probability_grid)}")
                print(f"total number of zero cells: {total_zero_pairs}")
                print(f"bot postion is: {self.bot.position}")
                self.print_ship_state()
                print()
                i = input("Enter someting...")
            # print(f"the target is {target}")
            # print(f"total moves: {actions}")
            # print(f"curr path is: {curr_path}") 
            # h = input("Enter pitstop....") 
            if curr_path:
                #self.print_ship_state()
                # i = input("Enter something ...")
                next_move = curr_path.pop(0)
                self.bot.move(next_move)
                self.update_probabilities_for_visited(next_move)
                actions += 1
                if next_move in self.leaks:
                    self.leaks.remove(next_move)
                    return actions# Remove the found leak
            else:
                beep_heard = self.sense_mutiple()
                actions += 1
                #self.update_probabilities_for_visited(visited)
                self.update_probabilities_multiple(beep_heard)
                target = self.choose_next_move_multi()
                curr_path = self.find_path_to_edge(self.bot.position, target)
        return actions
    
    
    
    
    ############################################ strategy 9 ##################################################
    
    
    def choose_next_move_multi_nine(self):
        # Find the maximum numerical probability
        max_probability = max(self.probability_grid.values())
        
        print(f"Max probability: {max_probability}")

        # Compile a list of cells that have the maximum probability
        candidate_cells = [
            key for key, val in self.probability_grid.items() if val == max_probability
        ]
        
        #print(candidate_cells)
         # Calculate BFS distance once and store it
        distances = {cell_pair: min(self.bfs(self.bot.position, cell_pair[0]), self.bfs(self.bot.position, cell_pair[1])) for cell_pair in candidate_cells}
        # Find the closest candidate cell
        min_distance = math.inf
        
        #Breaking ties by distance 
        # Sort by distance
        closest_candidates = sorted(candidate_cells, key=lambda cell_pair: distances[cell_pair])

        min_distance = distances[closest_candidates[0]]

        # Filter cells with minimum distance
        cells_with_min_distance = [cell for cell in closest_candidates if distances[cell] == min_distance]
        choice  = random.choice(cells_with_min_distance)
        #print(f"target choice is: {choice}")
        # calc_dis_one = self.bfs(self.bot.position, choice[0])
        # calc_dis_two = self.bfs(self.bot.position, choice[1])
        # if calc_dis_one < calc_dis_two:
        #     return choice[0]
        return choice
    
    
    
    
    def print_ship_state(self):
    # Create a copy of the ship's grid to represent the current state
        ship_state = [[cell for cell in row] for row in self.ship.ship]

        # Mark the leaks on the grid
        for leak in self.leaks:
            x, y = leak
            ship_state[x][y] = 'L'  # Representing leaks with 'L'

        # Mark the bot's position on the grid
        bot_x, bot_y = self.bot.position
        ship_state[bot_x][bot_y] = 'B'  # Representing the bot with 'B'

        # Print the current state of the ship
        for row in ship_state:
            print(' '.join(str(cell) for cell in row))
        print("\n")
     
    ########################################## Strategy Nine ###################################################
        
    def strategy_nine(self):
        actions = 0
        beep_heard = self.sense_mutiple()
        print(f"bot postion is: {self.bot.position}")
        print(f"total moves: {actions}")
        visited = self.bot.position
        self.update_probabilities_for_visited(visited)
        print("after update first")
        self.update_probabilities_multiple(beep_heard)
        print("after update multiple")
        steps = 0
        target = None
        patched_leak = None
        choice_target = self.choose_next_move_multi_nine()
        if self.bfs(self.bot.position, choice_target[0]) < self.bfs(self.bot.position, choice_target[1]):
            target = choice_target[0]
        else:
            target = choice_target[1]
        curr_path = self.find_path_to_edge(self.bot.position, target)
        print(f"leaks are in {self.leaks}")
        is_exploiting = False
        is_exploring = False
        if beep_heard:
            is_exploiting = True
            is_exploring = False
        else:
            is_exploring  = True
            is_exploiting = False
        search_space = None
        #i = input("Enter something ...")
        first_leaks = True
        while first_leaks:
            # print(f"total cells in probability grid: {len(self.probability_grid)}")
            # print(f"total number of zero cells: {total_zero_pairs}")
            # print(f"bot postion is: {self.bot.position}")
            #print(f"total moves: {actions}")
            # print(f"curr path is: {curr_path}")
            steps += 1
            if is_exploring and steps % 8 == 0:
                beep_heard = self.sense_mutiple()
                actions += 1
                if beep_heard:
                    is_exploiting = True
                    is_exploring = False
                else:
                    is_exploring = True
                    is_exploiting = False
            if is_exploiting:
                target = self.find_closest_high_prob_cell()
                curr_path = self.find_path_to_edge(self.bot.position, target)
            if self.bot.position in self.leaks:
                self.leaks.remove(self.bot.position)
                self.update_probabilities_for_visited(self.bot.position)
                self.normalize_probabilities()
                first_leaks = False
                break
                if not self.leaks():
                    return actions
            if curr_path:
                #self.print_ship_state()
                # i = input("Enter something ...")
                steps += 1
                next_move = curr_path.pop(0)
                self.bot.move(next_move)
                #self.print_ship_state
                #print(f"total moves: {actions}")
                if self.bot.position in self.leaks:
                        self.leaks.remove(next_move)  # Remove the found leak
                        patched_leak = next_move
                        # if not self.leaks:
                        #     print("All leaks patched")
                        #     return actions
                        print(f"Leak found and removed at {next_move}. Remaining leaks: {self.leaks}") 
                        print(f"total moves: {actions}")
                        print(f"bot postion is: {self.bot.position}")
                        #return actions
                        search_space = self.get_all_leak_cells_with_first_leak(self.bot.position, choice_target)
                        #print(search_space)
                        #i = input("Enter something....")
                        first_leaks = False
                self.visited.add(self.bot.position)
                #print(f"total actions: {actions}")
                self.update_probabilities_for_visited(next_move)
                actions += 1
                #i = input("Break ....")
            else:
                print(f"total actions: {actions}")
                print(f"bot position: {self.bot.position}")
                self.visited.add(self.bot.position)
                #self.print_ship_state()
                #print(f"total moves: {actions}")
                #print(f"bot postion is: {self.bot.position}")
                beep_heard = self.sense_mutiple()
                self.update_probabilities_multiple(beep_heard)
                actions += 1
                steps += 1
                if beep_heard:
                    is_exploiting = True
                    is_exploring  = False
                else:
                    is_exploring = True
                    is_exploiting = False
                if is_exploiting:
                    target = self.find_closest_high_prob_cell()
                    curr_path = self.find_path_to_edge(self.bot.position, target)
                elif is_exploring:
                    target = self.find_unvisited_cell()
                    curr_path = self.find_path_to_edge(self.bot.position, target)
        if search_space:
            beep = self.sense_smaller_space()
            actions += 1
            print(f"total actions: {actions}")
            print(f"beep heard smaller space: {beep}")
            #self.print_ship_state()
            #i = input("Check state...")
            self.update_probabilities_new_search_space(search_space, beep)
            #if beep:
            #    target = target = self.find_closest_high_prob_cell()
            #    curr_path = self.find_path_to_edge(self.bot.position, target)
            choice_target = self.choose_move_search_space(search_space)
            if choice_target[1] == patched_leak:
                target = choice_target[0]
            else:
                target = choice_target[1]
            curr_path = self.find_path_to_edge(self.bot.position, target)
            # if choice_target:
            #     search_space[choice_target] = 0
            # else:
            #     search_space[(target, patched_leak)] = 0
            while self.leaks:
                if curr_path:
                    next_move = curr_path.pop(0)
                    #self.print_ship_state()
                    # i = input("Check state...")
                    # self.bot.move(next_move)
                    if self.bot.position == self.leaks[0]:
                        return actions
                    search_space[choice_target] = 0
                    if next_move in self.leaks:
                        self.leaks.remove(next_move)  # Remove the found leak
                        print(f"Leak found and removed at {next_move}. Remaining leaks: {self.leaks}")
                        return actions
                        #i = input("Enter something....")
                    self.normalize_search_space(search_space)
                    # print(search_space)
                    # i = input("dddd")
                    actions += 1
                    #i = input("stop at self.leaks curr path ....")
                else:
                    beep_heard = self.sense_smaller_space()
                    actions += 1
                    #self.update_probabilities_for_visited(visited)
                    self.update_probabilities_new_search_space(search_space, beep_heard)
                    # if beep:
                    #     target = target = self.find_closest_high_prob_cell()
                    #     curr_path = self.find_path_to_edge(self.bot.position, target)
                    
                    choice_target = self.choose_move_search_space(search_space)
                    #print(f"target is: {choice_target}")
                    if choice_target[0] == patched_leak:
                        target = choice_target[1]
                    else:
                        target = choice_target[0]
                    #print(f"bot position is: {self.bot.position}")
                    curr_path = self.find_path_to_edge(self.bot.position, target)
                    #print(f"curr_path is: {curr_path}")
                    #i = input("else in while leaks ....")
            return actions
    
    
    def sense_smaller_space(self):
        beep_probability = 0
        for leak in self.leaks:
            beep_probability += self.calculate_probability(leak)
        return (random.random()/10) <= beep_probability
    
    
    def choose_move_search_space(self, search_space):
        max_probability = max(search_space.values())
        
        #print(f"Max probability: {max_probability}")
        high_prob_threshold = .90 * max_probability
        high_prob_cells = [cell for cell, prob in search_space.items() if prob > high_prob_threshold]
        flattened_cells = [item for pair in high_prob_cells for item in pair]
        unique_cells = list(set(flattened_cells))
        nearest_cell = self.bfs_target(self.bot.position, unique_cells)
        for key in high_prob_cells:
            if nearest_cell in key:
                return key
        return None
        


    def get_all_leak_cells_with_first_leak(self, position, found_leak):
        new_search_space = {key : val for key, val in self.probability_grid.items() if position in key and val != 0}
        new_search_space[found_leak]  = 0
        
        self.normalize_search_space(new_search_space)
        return new_search_space
    
    def update_probabilities_new_search_space(self, search_space,beep):
        
        for ((cell_j), (cell_k)), probability in search_space.items():
            if probability == 0:
                continue
            probability_cell_j = self.calculate_probability(cell_j)
            probability_cell_k = self.calculate_probability(cell_k)
            if beep:
                new_prob = 1 - (1 - probability_cell_j) * (1 - probability_cell_k)
                search_space[(cell_j, cell_k)] = new_prob * probability
            else:
                new_prob = (1 - probability_cell_j) * (1 - probability_cell_k)
                search_space[(cell_j, cell_k)] = new_prob * probability
                
        self.normalize_search_space(search_space)
        
        
    def normalize_search_space(self, search_space):
        total_probability = sum(search_space.values())
        if total_probability == 0:
            print(self.leaks)
            print(self.bot.position)
        #print(f"total probablity in search space: {total_probability}")
        
        for key in search_space:
            search_space[key] /= total_probability

    def find_closest_high_prob_cell(self):
        Max_probability = max(self.probability_grid.values())
        high_prob_threshold = .80 * Max_probability
        high_prob_cells = [cell for cell, prob in self.probability_grid.items() if prob > high_prob_threshold]
        flattened_cells = [item for pair in high_prob_cells for item in pair]
        unique_cells = list(set(flattened_cells))
        nearest_cell = self.bfs_target(self.bot.position, unique_cells)
        return nearest_cell

    def find_unvisited_cell(self):
        unvisited_cells = [cell for cell in self.ship.open_cells if cell not in self.visited]
        return random.choice(unvisited_cells) if unvisited_cells else None
    
    
    def bfs_target(self, start_position, target):
        queue = [start_position]  # Queue holds tuples of (position, distance)
        visited = set(start_position)

        while queue:
            cell = queue.pop(0)

            if cell in target:
                return cell
            visited.add(cell)

            # Add unvisited neighbors to the queue
            for neighbor in self.ship.get_open_neighbors(cell):
                if neighbor not in visited:
                    queue.append(neighbor)

        return None


    


    
    def run_game(self):
        if self.bot_strategy == 3:
          return self.play_game()
        if self.bot_strategy == 4:
            return self.play_game_four_two()
        if self.bot_strategy == 7:
           return self.play_game_seven()
        if self.bot_strategy == 8:
            return self.strategy_eight()
        if self.bot_strategy == 9:
            return self.strategy_nine()
        
    
    
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
