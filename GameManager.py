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
        self.knowledge_grid = [['#' if cell == 1 else 'UNKNOWN' for cell in row] for row in self.ship.ship]
        self.initialize_leaks()
        
    
     
    def initialize_leaks(self):
        open_cells = self.ship.open_cells.copy()
        
        open_cells.remove(self.bot_position) 

        num_leaks = 2 if self.bot_strategy in range(5, 10) else 1
        for _ in range(num_leaks):
            leak_position = self.get_leak_position()
            self.leaks.append(leak_position)
      
            
    def get_leak_position(self):
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
        print("All open cells:", all_open_cells)
        print("Detection square cells:", detection_square_cells)


        # Subtract detection square cells from all open cells to get potential leak positions
        potential_leak_positions = all_open_cells - detection_square_cells
        print(potential_leak_positions)

        # Randomly select a leak position
        leak_position = random.choice(list(potential_leak_positions))
        return leak_position

        

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
                if self.knowledge_grid[cell[0]][cell[1]] == "UNKNOWN":
                    self.knowledge_grid[cell[0]][cell[1]] = "NO LEAK"
            return None
        
        
    def sense_two(self):
        sensed_grid = self.generate_sense_grid()
        if self.leaks[0] in sensed_grid or self.leaks[1] in sensed_grid:
            for cell in sensed_grid: 
                self.knowledge_grid[cell[0]][cell[1]] = "MIGHT_HAVE_LEAK"
            return sensed_grid
        else:
            for cell in sensed_grid:
                if self.knowledge_grid[cell[0]][cell[1]] == "UNKNOWN":
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
        
        
    
    def dfs_bot_one(self):
        sensed = self.sense()
        game_is_on = True
        curr_path = []
        Moves = 0

        while game_is_on:
            print("\n")
            print(self)
            print("\n")
            print(self.bot.position)
            Moves += 1
            
            if self.bot.position == self.leaks[0]:
                print("Bot has patched the leak")
                game_is_on = False
                break
            
            print(curr_path)
            
            # If bot detected a leak, move to the nearest MIGHT_HAVE_LEAK cell
            if curr_path:
                next_move = curr_path.pop(0)
                self.bot.move(next_move)
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
                if sensed:
                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
            # If no leak is detected, move to the nearest UNKNOWN cell
                else:
                    print(f"sensed is: {sensed}")
                    print("not sensed")
                    curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")   
        return Moves
    


    def find_path_to_nearest_cell_with_status(self, start_position, target_status):
        queue = [(start_position, [])]  # Each item is a tuple (position, path_so_far)
        visited = set([start_position])

        while queue:
            (x, y), path = queue.pop(0)
            if self.knowledge_grid[x][y] == target_status:
                return path

            # Add unvisited neighbors to the queue
            for neighbors in self.ship.get_open_neighbors((x,y)):
                if neighbors not in visited:
                    new_path = path + [neighbors]
                    queue.append((neighbors, new_path))
                    visited.add(neighbors)

        return []  # Return an empty list if no cell with desired status is found
                   
    
    
    def strategy_one(self):
        total_moves = self.dfs_bot_one()
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
  
    def nearest_neighbor(self, start, points, visited):
        unvisited = set(points) - set(visited)
        tour = [start]
        current_point = start

        while unvisited:
            next_point = min(unvisited, key=lambda point: self.manhattan_distance(current_point, point))
            tour.append(next_point)
            unvisited.remove(next_point)
            current_point = next_point

        return tour
    
    
    def nearest_neighbor(self, start, points, visited):
    # Filter out the visited points first
        unvisited = set(points) - set(visited)
        # Start with the current point
        current_point = start

        if unvisited:
            # Find the nearest unvisited point
            next_point = min(unvisited, key=lambda point: self.manhattan_distance(current_point, point))
            return next_point
        else:
            # If there are no unvisited points, return None or some indication that all points have been visited
            return None

    
    def strategy_two(self):
        detection_threshold = 10
        game_is_on = True
        curr_path = []
        # If the detection range is large, start from the center
        total_moves = 0
        visited = set()
        
        strategic_points = self.get_strategic_points()
        edge_locater = 0
        edges = []
        sensed = self.sense()
        
        while game_is_on:
            print(f"sensed is: {sensed}")
            print(f"curr_path is: {curr_path}")
            print(total_moves)
            print(self)
            print("\n")
            if self.k > detection_threshold:
                if self.bot.position == self.leaks[0]:
                    game_is_on = False
                    return total_moves
                if curr_path:
                        next_move = curr_path.pop(0)
                        self.knowledge_grid[next_move[0]][next_move[1]] = "NO LEAK"
                        self.bot.move(next_move)
                        total_moves += 1
                        continue
                if not curr_path and not sensed:
                    sensed = self.sense()
                    total_moves += 1
                    if not sensed:
                        #next_target = self.nearest_neighbor(self.bot.position, strategic_points, visited)
                        next_target = strategic_points.pop(0)
                        if next_target:
                            visited.add(next_target)  # Add the next target to visited
                            curr_path = self.find_path_to_edge(self.bot.position, next_target)
                    elif sensed:
                        edges = self.infer_edges()
                        curr_path = self.find_path_to_edge(self.bot.position, edges[edge_locater])
                elif sensed and not curr_path:
                    new_sensed = self.sense()
                    if new_sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position,"MIGHT_HAVE_LEAK")
                    else:
                        edge_locater += 1
                        curr_path = self.find_path_to_edge(self.bot.position,edges[edge_locater])       
            else:
                if self.bot.position == self.leaks[0]:
                    game_is_on = False
                    return total_moves
                if curr_path:
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
                    next_target = strategic_points.pop(0)
                    curr_path = self.find_path_to_edge(self.bot_position, next_target)
                    
    def strategy_five(self):
        sensed = self.sense()
        game_is_on = True
        curr_path = []
        Moves = 0

        while game_is_on:
            print("\n")
            print(self)
            print("\n")
            print(self.bot.position)
            Moves += 1
            
            
            patched_leaks = [leak for leak in self.leaks if self.bot.position == leak]
            for leak in patched_leaks:
                self.leaks.remove(leak)
                print("Bot has patched a leak")
            
            if not self.leaks:
                return Moves
            
            # If bot detected a leak, move to the nearest MIGHT_HAVE_LEAK cell
            if curr_path:
                next_move = curr_path.pop(0)
                self.bot.move(next_move)
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
                if len(self.leaks) > 1:
                    sensed = self.sense_two()
                    if sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
                    else:
                        print(f"sensed is: {sensed}")
                        print("not sensed")
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")  
                else:
                    sensed = self.sense()
                    if sensed:
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "MIGHT_HAVE_LEAK")
            # If no leak is detected, move to the nearest UNKNOWN cell
                    else:
                        print(f"sensed is: {sensed}")
                        print("not sensed")
                        curr_path = self.find_path_to_nearest_cell_with_status(self.bot.position, "UNKNOWN")   
        return Moves
                    

    
    def run_game(self):
        if self.bot_strategy == 1:
            return self.strategy_one()
        if self.bot_strategy == 2:
            return self.strategy_two()
        if self.bot_strategy == 5:
            return self.strategy_five()
    
    
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



        
            
