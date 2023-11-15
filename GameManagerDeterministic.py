import itertools
import random
from collections import deque
from Ship import Ship
from bot import Bot

class GameManager:
    def __init__(self, ship_size,bot_strategy, k):
        """
        Initialize the GameManager class which manages the game's overall state, 
        including the bot, leaks, ship layout, and game elements' positions.

        Args:
            ship_size (int): The size (dimension) of the ship.
            bot_strategy (function): The strategy the bot uses to navigate.
        """
        self.ship = Ship(ship_size)  # Create a ship object with the given size.
        
        self.ship_length = ship_size  # Store the ship size.
        
        self.k = k
        
        # Initialize positions for the bot, fire, and button.
        self.bot_position = random.choice(self.ship.open_cells)
        
        # Create leaks on the ship randomly generated based on bot strategy 
        self.leaks = []
        
        self.covered_grid = []

         # Store the selected bot strategy.
        self.bot_strategy = bot_strategy  

        # Create a Bot instance initialized with the ship, its strategy,
        self.bot = Bot(self.ship, bot_strategy, self.bot_position, k)

        # Initialize a set to keep track of nodes (positions) that have been visited during pathfinding
        self.visited_nodes = set()
        
        #self.knowledge_grid = [['UNKNOWN' for _ in range(ship_size)] for _ in range(ship_size)]
        self.knowledge_grid = [['#' if cell == 1 else 'UNKNOWN' for cell in row] for row in self.ship.ship] if self.bot_strategy in range(0,7) else self.initialize_knowledge_grid_multiple()
        
        self.isleak = self.initialize_leaks()
        
        self.no_leaks = []
        
    
     
    def initialize_leaks(self):
        # Start with a copy of all open cells in the ship
        open_cells = self.ship.open_cells.copy()
        
        # Remove the bot's current position from the list of open cells to avoid placing a leak there
        open_cells.remove(self.bot_position) 
    
        # Determine the number of leaks to initialize based on the bot's strategy
        num_leaks = 2 if self.bot_strategy in range(5, 10) else 1
        for _ in range(num_leaks):
            # Get a random position for the leak
            leak_position = self.get_leak_position()
            if not leak_position:
                # If no valid leak position was found, return False to indicate failure
                return False
            # Add the found leak position to the list of leaks
            self.leaks.append(leak_position)
        # Return True to indicate successful initialization of leaks
        return True
    
    
    
    def initialize_knowledge_grid_multiple(self):
       grid = {}
       open_cells = self.ship.open_cells
       for cell1, cell2 in itertools.combinations(open_cells, 2):
           grid[(cell1, cell2)] = "UNKOWN"
       return grid
           
    def get_leak_position(self):
        # Extract bot's x and y coordinates
        x, y = self.bot.position

        # Determine the bounds of the detection square based on the bot's position and detection range 'k'
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

        # Subtract detection square cells from all open cells to get potential leak positions
        potential_leak_positions = all_open_cells - detection_square_cells
        if potential_leak_positions:
            # Randomly select a leak position from the potential positions
            leak_position = random.choice(list(potential_leak_positions))
            return leak_position
        else:
            # If there are no potential leak positions, return None or handle it as needed
            return None


        

############################################### Stategy 1 #####################################################

    def sense(self):
        sensed_grid = self.generate_sense_grid()
        if self.leaks[0] in sensed_grid:
            for cell in sensed_grid: 
                if self.knowledge_grid[cell[0]][cell[1]] == "UNKNOWN":
                    self.knowledge_grid[cell[0]][cell[1]] = "MIGHT_HAVE_LEAK"
            for i in range(len(self.knowledge_grid)):
                for j in range(len(self.knowledge_grid[i])):
                    if (i,j) not in sensed_grid:
                        self.knowledge_grid[i][j] = "NO LEAK"
            return sensed_grid
        else:
            for cell in sensed_grid:
                self.knowledge_grid[cell[0]][cell[1]] = "NO LEAK"
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
        
        
    def manhattan_distance(self, point1, point2):
        # Returns the Manhattan distance
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        
        
    
    def strategy_bot_one(self):
        sensed = self.sense()
        game_is_on = True
        curr_path = []
        Moves = 0

        while game_is_on:
            print("\n")
            #print(self)
            #print("\n")
            #print(self.bot.position)
            print(f"Moves is: {Moves}")
            
            if self.bot.position == self.leaks[0]:
                Moves += 1
                print("Bot has patched the leak")
                game_is_on = False
                break
            
            #print(curr_path)
            
            # If bot detected a leak, move to the nearest MIGHT_HAVE_LEAK cell
            if curr_path:
                next_move = curr_path.pop(0)
                self.bot.move(next_move)
                Moves += 1
                self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                continue
            
            if sensed and not curr_path:
                curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                continue
            
            if not sensed and not curr_path:
                print("bot is sensing")
                #new_sensed = self.sense()
                #if sensed is None:
                    #sensed = set()
                sensed = self.sense()
                Moves += 1
                if sensed:
                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
            # If no leak is detected, move to the nearest UNKNOWN cell
                else:
                    #print(f"sensed is: {sensed}")
                    #print("not sensed")
                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")   
        return Moves
    


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
                   
    
    
    def strategy_one(self):
        total_moves = self.strategy_bot_one()
        print(f"The bot patches the leak in {total_moves} moves")
        return total_moves
        
        
    ##################################### Strategy 2: Self- made deterministic Bot #################################
    
    def get_nearest_open_cell(self, cell):
        if self.ship.ship[cell[0]][cell[1]] == 0:
            return cell  # The cell is already open, return it

        # Start a BFS to find the nearest open cell
        queue = deque([cell])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if self.ship.ship[current[0]][current[1]] == 0:
                return current
            
            for neighbors in self.ship.get_neighbors(current):
                if neighbors not in visited:
                    queue.append(neighbors)
        return current
    

    
    def remove_duplicates_preserve_order(self,seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
    
    
    def get_inner_strategic_points(self, edge, ship_size, k):
        # Calculate strategic points for a single layer
        strategic_points = [(x, edge) for x in range(edge, ship_size - edge, 2*k + 1)] + \
                           [(x, ship_size - edge - 1) for x in range(edge, ship_size - edge, 2*k + 1)] + \
                           [(edge, y) for y in range(edge, ship_size - edge, 2*k + 1)] + \
                           [(ship_size - edge - 1, y) for y in range(edge, ship_size - edge, 2*k + 1)]
                           
        for cell in strategic_points:
            x,y = cell
            left_bound = x - k
            right_bound = x + k
            upper_bound = y - k
            lower_bound = y + k
            
            total_grid = [(i,j) for i in range(left_bound, right_bound + 1) for j in range(upper_bound, lower_bound + 1)]
            self.covered_grid.extend(total_grid)                    

        strategic_points = self.remove_duplicates_preserve_order(strategic_points)
        return strategic_points
            
    
    
    def get_strategic_points(self):
        all_strategic_points = []
        ship_size = self.ship.get_length()  # Assuming this method returns the size of the ship
        k = self.k
        current_edge = k

        while current_edge < ship_size - k:
            layer_points = self.get_inner_strategic_points(current_edge, ship_size, k)
            for i in range(len(layer_points)):
                cell = layer_points[i]
                if self.ship.ship[cell[0]][cell[1]] == 1:
                    open_cell = self.get_nearest_open_cell(cell)
                    layer_points[i] = open_cell
                    
            all_strategic_points.extend(layer_points)
            # Move to the next inner layer
            current_edge += 2 * k + 1
        
        borderlands = []
        
        for i in range(ship_size):
            for j in range(ship_size):
                if (i, j) not in self.covered_grid and self.ship.ship[i][j] == 0:
                    borderlands.append((i, j))
        all_strategic_points.extend(borderlands)
        return all_strategic_points
    
    
    
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


    
    def infer_edges(self):
        x, y = self.bot.position

        # Define the detection edges relative to the current position
        edges = [
            (x - self.k, y),  # Left edge
            (x + self.k, y),  # Right edge
            (x, y - self.k),  # Top edge
            (x, y + self.k),  # Bottom edge
        ]

        # Filter out any edge points that may be outside the bounds of the ship
        ship_bounds = (self.ship_length, self.ship_length)  # Assuming get_size() returns a tuple (width, height)
        edges = [
            (max(0, min(edge[0], ship_bounds[0] - 1)), max(0, min(edge[1], ship_bounds[1] - 1)))
            for edge in edges
        ]
        
        for i in range(len(edges)):
            cell = edges[i]
            if self.ship.ship[cell[0]][cell[1]] == 1:
                new_cell = self.get_nearest_open_cell(cell)
                edges[i] = new_cell

        # Return the list of edge points
        return edges

######################################### Main  Strategy - Two ################################################## 
    
    def strategy_two(self):
        detection_threshold = 10
        game_is_on = True
        curr_path = []
        # If the detection range is large, start from the center
        total_moves = 0
        
        strategic_points = self.get_strategic_points()
        edge_locater = 0
        edges = []
        sensed = self.sense()
        initial_target_reached = False
        steps = 0
        target = strategic_points.pop(0)
        
        while game_is_on:
            #print(f"sensed is: {sensed}")
            #print(f"curr_path is: {curr_path}")
            print(f"total_moves are: {total_moves}")
            #print(f"bot position is: {self.bot.position}")
            steps += 1
            #print(self)
            if total_moves > 3000:
                x,y  = self.leaks[0]
                print(f"curr_path is: {curr_path}")
                print(f"bot position is: {self.bot.position}")
                print(f"sensed is: {sensed}")
                print(f"leaks in knowledge grid: {self.knowledge_grid[x][y]}")
                print(self)
                user = input("Press Enter....")
            while not initial_target_reached:
                if self.bot.position == target:
                    initial_target_reached = True
                if not curr_path:
                    curr_path = self.find_path_to_edge(self.bot.position, target)
                elif curr_path:
                    next_move = curr_path.pop(0)
                    if(next_move == self.leaks[0]):
                        self.bot.move(next_move)
                        total_moves += 1
                        return total_moves
                    self.bot.move(next_move)
                    self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                    total_moves += 1
                    steps += 1
                    if steps % (2*self.k +1) == 0:
                        sensed = self.sense()
                        total_moves += 1
                        if sensed:
                            curr_path = []
                            initial_target_reached = True
                            break
            #print(f"curr_path after loop: {curr_path}")
            #m = input("Breakpoint....")
            if self.k >= detection_threshold:
                #print(type(curr_path))
                #print(curr_path)
                if self.bot.position == self.leaks[0]:
                    #n = input("Bot is here...")
                    print("Bot has patched the leak")
                    game_is_on = False
                    return total_moves
                if curr_path and not sensed:
                        if steps % (2*self.k + 1) == 0:
                            sensed = self.sense()
                            total_moves += 1
                            if sensed:
                                edges = self.infer_edges()
                                curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                                #print(f"curr_path after edge locator: {curr_path}")
                        if curr_path:
                            #print(curr_path)
                            next_move = curr_path.pop(0)
                            self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                            self.bot.move(next_move)
                            total_moves += 1
                            continue
                if sensed:
                    #print(f"length of edges: {len(edges)}")
                    #m = input("Have sensed...")
                    if curr_path:
                        next_move = curr_path.pop(0)
                        self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                        self.bot.move(next_move)
                        total_moves += 1
                    elif not edges:
                        edges = self.infer_edges()
                        curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                    else:
                        new_sensed = self.sense()
                        if new_sensed:
                            if self.leaks[0] not in new_sensed:
                                print("not in sensed")
                                #k = input("Enter....")
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"MIGHT_HAVE_LEAK")
                        else:
                            #print(f"current edge number: {edge_locater}")
                            #print(new_sensed)
                            if self.leaks[0] in sensed:
                                print("in sensed")
                            if edge_locater >= len(edges):
                                print(f" length of egdes: {len(edges)}")
                                #m = input("Edge Locator Breached!")
                            curr_path = self.find_path_to_edge(self.bot.position,edges[edge_locater])
                            edge_locater += 1
                if not sensed and not curr_path:
                    if strategic_points:
                        point = strategic_points.pop(0)
                        self.find_path_to_edge(self.bot.position, point)
                    else:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")                
            else:
                if self.bot.position == self.leaks[0]:
                    game_is_on = False
                    return total_moves
                if curr_path:
                    if steps % 8 == 0 and not sensed:
                        sensed = self.sense()
                        if sensed:
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                    next_move = curr_path.pop(0)
                    self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                    self.bot.move(next_move)
                    total_moves += 1
                    continue
                if sensed and not curr_path:
                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                if not curr_path:
                    sensed = self.sense()
                    total_moves += 1
                    if sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                if not curr_path and not sensed:
                    #next_target = self.nearest_neighbor(self.bot.position, strategic_points, visited)
                    if strategic_points:
                        next_target = strategic_points.pop(0)
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    else:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"UNKNOWN" )
                    

######################################### Main  Strategy - Five ################################################## 
                    
    def sense_three(self):
        sensed_grid = self.generate_sense_grid()
        leaks = [leaks for leaks in self.leaks if leaks in sensed_grid]
        print(f"leaks in sensed: {leaks}")
        if leaks:
            #print("I am dumb so i find leaks")
            for cell in sensed_grid:
                if self.knowledge_grid[cell[0]][cell[1]] != "NO LEAK":
                    self.knowledge_grid[cell[0]][cell[1]] = "MIGHT_HAVE_LEAK"
            return sensed_grid
        else:
            for cell in sensed_grid:
                self.knowledge_grid[cell[0]][cell[1]] = "NO LEAK"
        return None     
      
         
         
                    
    def strategy_five(self):
        sensed = self.sense()
        game_is_on = True
        curr_path = []
        Moves = 0
        h = 0

        while game_is_on:
            # print("\n")
            # print(self)
            # print("\n")
            # print(self.bot.position)
            h += 1
            print(f"Move Counter: {Moves}")
            
            if Moves > 5000 and not curr_path:
                print(f"bot position is: {self.bot.position}")
                print(self)
                print(len(self.leaks))
                press_key = input("Press Enter...")
            
            
            patched_leaks = [leak for leak in self.leaks if self.bot.position == leak]
            for leak in patched_leaks:
                self.leaks.remove(leak)
                sensed = self.sense_three()
                print(f"Sensed after patch: {sensed}")
                Moves += 1
                print("Bot has patched a leak")
            
            if not self.leaks:
                print(len(self.leaks))
                print("All leaks patched")
                return Moves

            # if len(self.leaks) == 1:
            #     print(len(self.leaks))
            #     print(f"Sensed after each move patch: {sensed}")
            #     press_key = input("Write something to continue...")
        # If bot detected a leak, move to the nearest MIGHT_HAVE_LEAK cell
            if curr_path:
                next_move = curr_path.pop(0)
                self.bot.move(next_move)
                self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                Moves += 1
                continue
                
            if sensed and not curr_path:
                #print(sensed)
                print(f"number of leaks remaing is: {len(self.leaks)}")
                print("has sensed but no path ")
                self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"MIGHT_HAVE_LEAK")
                continue
        
            if not sensed and not curr_path:
                print("bot is sensing")
                #new_sensed = self.sense()
                #if sensed is None:
                    #sensed = set()
                if len(self.leaks) > 1:
                    sensed = self.sense_three()
                    Moves += 1
                    if sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"MIGHT_HAVE_LEAK")
                        print(curr_path)
                        print("I have sensed")
                        print(sensed)
                    else:
                        print(f"sensed is: {sensed}")
                        print("not sensed")
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"UNKNOWN") 
                else:
                    print("Next else triggered")
                    sensed = self.sense_three()
                    Moves += 1
                    if sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
            # If no leak is detected, move to the nearest UNKNOWN cell
                    else:
                        print(f"sensed is: {sensed}")
                        print("not sensed")
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")
                        if not curr_path:
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                            
                            
                            
  ######################################### Main  Strategy - Six ##################################################                           
     
    def update_knowledge_grid(self, cell):
        for (cell1, cell2), status in self.knowledge_grid.items():
            if cell == cell1 or cell == cell2:
                self.knowledge_grid[(cell1, cell2)] = "MIGHT_HAVE_LEAK"
            else:
                self.knowledge_grid[(cell1, cell2)] = "NO_LEAK" 

    
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
                

    def sense_six(self):
        sensed_grid = self.generate_sense_grid()  # This should generate a list of cells that the bot can sense
        leaks_detected = [leak for leak in self.leaks if leak in sensed_grid]

        print(f"Leaks in sensed: {leaks_detected}")
        # Update pairs with 'MIGHT_HAVE_LEAK' or 'NO_LEAK' based on sensing
        for (cell1, cell2), status in self.knowledge_grid.items():
            if cell1 in sensed_grid or cell2 in sensed_grid:
                # If any of the leaks are in the sensed grid, mark the pair as 'MIGHT_HAVE_LEAK'
                if any(leak in leaks_detected for leak in (cell1, cell2)):
                    self.knowledge_grid[(cell1, cell2)] = "MIGHT_HAVE_LEAK"
                else:
                    # If none of the leaks are in the sensed grid, keep the current status
                    continue
            else:
                # If neither cell in the pair is in the sensed grid, mark the pair as 'NO_LEAK'
                self.knowledge_grid[(cell1, cell2)] = "NO_LEAK"

        # Return the list of cells that were sensed to have a leak, if any
        return leaks_detected if leaks_detected else None

                
                
    def refine_knowledge_after_leak(self, found_leak):
        for pair in list(self.knowledge_grid.keys()):
            if found_leak not in pair:
                del self.knowledge_grid[pair]

    def choose_next_move_MHL(self):
        candidate_pairs = [pair for pair, status in self.knowledge_grid.items() if status == "MIGHT_HAVE_LEAK"]
        candidate_cells = set([cell for pair in candidate_pairs for cell in pair])

        # Find the closest cell from candidate cells
        closest_cell = self.find_closest_cell(self.bot.position, candidate_cells)
        print(closest_cell)
        return closest_cell

    def choose_next_move_UK(self):
        candidate_pairs = [pair for pair, status in self.knowledge_grid.items() if status == "UNKNOWN"]
        candidate_cells = set([cell for pair in candidate_pairs for cell in pair])

        # Find the closest cell from candidate cells
        closest_cell = self.find_closest_cell(self.bot.position, candidate_cells)
        print(closest_cell)
        return closest_cell

    def find_closest_cell(self, current_position, candidate_cells):
        min_distance = float('inf')
        closest_cell = None

        for cell in candidate_cells:
            distance = self.bfs(current_position, cell)
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell

        return closest_cell

        
        
    def strategy_six_two(self):
        detection_threshold = 10
        game_is_on = True
        curr_path = []
        # If the detection range is large, start from the center
        total_moves = 0
        edge_moves = 0
        strategic_points = self.get_strategic_points()
        edge_locater = 0
        edges = []
        sensed = self.sense_six()
        initial_target_reached = False
        steps = 0
        target = strategic_points.pop(0)
        while game_is_on:
            steps += 1
            #print(f"curr path: {curr_path}")
            #print(f"sensed is: {sensed}")
            print(f"total_moves are: {total_moves}")
            print(f"bot position is: {self.bot.position}")
            #print(self)
            # if total_moves > 5000:
            #     print(f"curr path: {curr_path}")
                # print(f"sensed is: {sensed}")
                # print(self)
                # user = input("Press Enter....")
            while not initial_target_reached:
                if self.bot.position == target:
                    sensed = self.sense_six()
                    initial_target_reached = True
                if not curr_path:
                    curr_path = self.find_path_to_edge(self.bot.position, target)
                elif curr_path:
                    # print(f"in loop curr path: {curr_path}")
                    # m = input("Press Enter...")
                    next_move = curr_path.pop(0)
                    self.bot.move(next_move)
                    if next_move in self.leaks:
                        self.bot.move(next_move)
                        self.leaks.remove(next_move)
                        total_moves += 1
                        if len(self.leaks) == 0:
                            return total_moves
                    self.update_knowledge_grid(self.bot.position)
                    total_moves += 1
                    steps += 1
                    if steps % (2*self.k +1) == 0:
                        sensed = self.sense_six()
                        total_moves += 1
                        if sensed:
                            curr_path = []
                            self.update_knowledge_grid(self.bot.position)
                            initial_target_reached = True
                            break                      
            patched_leaks = [leak for leak in self.leaks if self.bot.position == leak]
            for leak in patched_leaks:
                self.leaks.remove(leak)
                sensed = self.sense_six()
                edges.clear()
                #print(f"Sensed after patch: {sensed}")
                total_moves += 1
                #print("Bot has patched a leak")
            
            if not self.leaks:
                print(len(self.leaks))
                print("All leaks patched")
                return total_moves
            
            if self.k >= detection_threshold:
                if curr_path and not sensed:
                        if steps % (2*self.k + 1) == 0:
                            sensed = self.sense_six()
                            total_moves += 1
                            if sensed:
                                edges = self.infer_edges()
                                self.update_knowledge_grid(self.bot.position)
                                if edge_locater >= 4:
                                    edge_locater = 0
                                    curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                        if curr_path:
                            next_move = curr_path.pop(0)
                            self.update_knowledge_grid(self.bot.position)
                            self.bot.move(next_move)
                            total_moves += 1
                            continue
                if sensed:
                    #print(f"length of edges: {len(edges)}")
                    #m = input("Have sensed...")
                    if curr_path:
                        next_move = curr_path.pop(0)
                        self.update_knowledge_grid(self.bot.position)
                        self.bot.move(next_move)
                        total_moves += 1
                    elif not edges:
                        edges = self.infer_edges()
                        edge_locater = 0
                        curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                    else:
                        #print("tryin to resense")
                        new_sensed = self.sense_six()
                        #print(f"new sensed is: {sensed}")
                        self.update_knowledge_grid(self.bot.position)
                        if new_sensed:
                                #print("not in sensed")
                                #k = input("Enter....")
                            next_target = self.choose_next_move_MHL()
                            curr_path = self.find_path_to_edge(self.bot.position, next_target)
                        else:
                            #print(f"current edge number: {edge_locater}")
                            #print(new_sensed)
                            if edge_locater >= len(edges):
                                edge_locater = 0
                                # print(f"length of edges: {len(edges)}")
                                # print(f"edge list: {edges}")
                                # print(self)
                                # i = input("Press Enter...")
                                # print(f" length of egdes: {len(edges)}")
                                # m = input("Edge Locator Breached!")
                            curr_path = self.find_path_to_edge(self.bot.position,edges[edge_locater])
                            edge_locater += 1
                            edge_moves += 1
                            if edge_moves > 10:
                                next_target = self.choose_next_move_MHL()
                                curr_path = self.find_path_to_edge(self.bot.position, next_target)
                                if not curr_path:
                                    next_target = self.choose_next_move_UK()
                                    curr_path = self.find_path_to_edge(self.bot.position, next_target)
                if not curr_path and not sensed:
                    self.update_knowledge_grid(self.bot.position)
                    if strategic_points and len(self.leaks) > 1:
                        next_target = strategic_points.pop(0)
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    else:
                        next_target = self.choose_next_move_UK()
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                        if not curr_path:
                            next_target = self.choose_next_move_MHL()
                            curr_path = self.find_path_to_edge(self.bot.position, next_target)
            else:
                if curr_path:
                    if steps % self.k == 0 and not sensed:
                        cell = curr_path[0]
                        sensed = self.sense_six()
                        if sensed:
                            #print(self.knowledge_grid[cell[0]][cell[1]])
                            self.update_knowledge_grid(self.bot.position)
                            next_target = self.choose_next_move_MHL()
                            curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    if curr_path:
                        next_move = curr_path.pop(0)
                        self.update_knowledge_grid(self.bot.position)
                        self.bot.move(next_move)
                        total_moves += 1
                        continue
                if sensed and not curr_path:
                    self.update_knowledge_grid(self.bot.position)
                    next_target = self.choose_next_move_MHL()
                    curr_path = self.find_path_to_edge(self.bot.position, next_target)
                if not curr_path:
                    print("no curr path, tryin to resense")
                    sensed = self.sense_six()
                    print(f"sensed is: {sensed}")
                    print(f"bot postion: {self.bot.position}")
                    i = input("Enter...")
                    total_moves += 1
                    if sensed:
                        next_target = self.choose_next_move_MHL()
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                if not curr_path and not sensed:
                    #next_target = self.nearest_neighbor(self.bot.position, strategic_points, visited)
                    print("finding strategic route")
                    i = input("Enter...")
                    if strategic_points:
                        next_target = strategic_points.pop(0)
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    else:
                        next_target = self.choose_next_move_UK()
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                        if not curr_path:
                            next_target = self.choose_next_move_MHL()
                            curr_path = self.find_path_to_edge(self.bot.position, next_target)
    


    def strategy_six(self):
        detection_threshold = 10
        game_is_on = True
        curr_path = []
        # If the detection range is large, start from the center
        total_moves = 0
        edge_moves = 0
        strategic_points = self.get_strategic_points()
        edge_locater = 0
        edges = []
        sensed = self.sense_three()
        initial_target_reached = False
        steps = 0
        target = strategic_points.pop(0)
        while game_is_on:
            steps += 1
            #print(f"curr path: {curr_path}")
            #print(f"sensed is: {sensed}")
            print(f"total_moves are: {total_moves}")
            print(f"bot position is: {self.bot.position}")
            #print(self)
            # if total_moves > 5000:
            #     print(f"curr path: {curr_path}")
                # print(f"sensed is: {sensed}")
                # print(self)
                # user = input("Press Enter....")
            while not initial_target_reached:
                if self.bot.position == target:
                    self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                    initial_target_reached = True
                if not curr_path:
                    curr_path = self.find_path_to_edge(self.bot.position, target)
                elif curr_path:
                    # print(f"in loop curr path: {curr_path}")
                    # m = input("Press Enter...")
                    next_move = curr_path.pop(0)
                    self.bot.move(next_move)
                    if next_move in self.leaks:
                        self.bot.move(next_move)
                        self.leaks.remove(next_move)
                        total_moves += 1
                        if len(self.leaks) == 0:
                            return total_moves
                    self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                    total_moves += 1
                    steps += 1
                    if steps % (2*self.k +1) == 0:
                        sensed = self.sense_three()
                        total_moves += 1
                        if sensed:
                            curr_path = []
                            self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                            initial_target_reached = True
                            break                      
            patched_leaks = [leak for leak in self.leaks if self.bot.position == leak]
            for leak in patched_leaks:
                self.leaks.remove(leak)
                sensed = self.sense_three()
                edges.clear()
                #print(f"Sensed after patch: {sensed}")
                total_moves += 1
                #print("Bot has patched a leak")
            
            if not self.leaks:
                print(len(self.leaks))
                print("All leaks patched")
                return total_moves
            
            if self.k >= detection_threshold:
                if curr_path and not sensed:
                        if steps % (2*self.k + 1) == 0:
                            sensed = self.sense_three()
                            total_moves += 1
                            if sensed:
                                edges = self.infer_edges()
                                self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                                if edge_locater >= 4:
                                    edge_locater = 0
                                    curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                        if curr_path:
                            next_move = curr_path.pop(0)
                            self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                            self.bot.move(next_move)
                            total_moves += 1
                            continue
                if sensed:
                    #print(f"length of edges: {len(edges)}")
                    #m = input("Have sensed...")
                    if curr_path:
                        next_move = curr_path.pop(0)
                        self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                        self.bot.move(next_move)
                        total_moves += 1
                    elif not edges:
                        edges = self.infer_edges()
                        edge_locater = 0
                        curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                    else:
                        #print("tryin to resense")
                        new_sensed = self.sense_three()
                        #print(f"new sensed is: {sensed}")
                        self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                        if new_sensed:
                                #print("not in sensed")
                                #k = input("Enter....")
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"MIGHT_HAVE_LEAK")
                        else:
                            #print(f"current edge number: {edge_locater}")
                            #print(new_sensed)
                            if edge_locater >= len(edges):
                                edge_locater = 0
                                # print(f"length of edges: {len(edges)}")
                                # print(f"edge list: {edges}")
                                # print(self)
                                # i = input("Press Enter...")
                                # print(f" length of egdes: {len(edges)}")
                                # m = input("Edge Locator Breached!")
                            curr_path = self.find_path_to_edge(self.bot.position,edges[edge_locater])
                            edge_locater += 1
                            edge_moves += 1
                            if edge_moves > 10:
                                curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                                if not curr_path:
                                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"UNKNOWN" )
                if not curr_path and not sensed:
                    self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                    if strategic_points and len(self.leaks) > 1:
                        next_target = strategic_points.pop(0)
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    else:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"UNKNOWN" )
                        if not curr_path:
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
            else:
                if curr_path:
                    if steps % self.k == 0 and not sensed:
                        cell = curr_path[0]
                        sensed = self.sense_three()
                        if sensed:
                            #print(self.knowledge_grid[cell[0]][cell[1]])
                            self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                    next_move = curr_path.pop(0)
                    self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                    self.bot.move(next_move)
                    total_moves += 1
                    continue
                if sensed and not curr_path:
                    self.knowledge_grid[self.bot.position[0]][self.bot.position[1]] = "NO LEAK"
                    print(self.knowledge_grid[self.bot.position[0]][self.bot.position[1]])
                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                if not curr_path:
                    print("no curr path, tryin to resense")
                    sensed = self.sense_three()
                    total_moves += 1
                    if sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                if not curr_path and not sensed:
                    #next_target = self.nearest_neighbor(self.bot.position, strategic_points, visited)
                    print("finding strategic route")
                    if strategic_points:
                        next_target = strategic_points.pop(0)
                        curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    else:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"UNKNOWN" )
                        if not curr_path:
                            curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
    
    
######################################### Running and Execution  ################################################## 
    
    
    def run_game(self):
        if self.bot_strategy == 1:
            return self.strategy_one()
        if self.bot_strategy == 2:
            return self.strategy_two()
        if self.bot_strategy == 5:
            return self.strategy_five()
        if self.bot_strategy == 6:
            return self.strategy_six()
        if self.bot_strategy == 8:
            return self.strategy_six_two()
    
    
    def __str__(self):
        output = []
        for i in range(self.ship_length):
            row = []
            for j in range(self.ship_length):
                # Check for the bot's position
                if (i, j) == self.bot.position:
                    row.append('B')
                # Check for leaks
                elif (i, j) in self.leaks:
                    row.append('L')
                # Check for walls
                elif self.ship.ship[i][j] == 1:
                    row.append('#')  # Representing walls with '#'
                # If it's not a wall, bot, or leak, check the status
                else:
                    status = self.knowledge_grid[i][j]
                    if status == "UNKNOWN":
                        row.append('?')
                    elif status == "MIGHT_HAVE_LEAK":
                        row.append('M')
                    elif status == "NO LEAK":
                        row.append('N')
                    else:
                        row.append('X')  # Default in case of unexpected statuses
            output.append(' '.join(row))
        return '\n'.join(output)
