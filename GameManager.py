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
        
        open_cells.remove(self.bot_position)  # Ensure bot's position isn't an option for a leak

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

        # Subtract detection square cells from all open cells to get potential leak positions
        potential_leak_positions = all_open_cells - detection_square_cells

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
                   
    
    
    def stretegy_one(self):
        total_moves = self.dfs_bot_one()
        print(f"The bot patches the leak in {total_moves} moves")
        return total_moves
        
        
    ##################################### Strategy 2: Self- made deterministic Bot #################################
    
    def move_center(self):
        center = self.ship.find_nearest_open_center()
        """
         Uses Breadth-First Search (BFS) to find the shortest path to the button.
        The bot tries to move toward the button while avoiding already visited cells or cells on fire.
        """
        visited = set()
        queue = deque([[self.bot.position]])  # queue to hold all paths; initially it has one path with only the start node
    
        while queue:
            path = queue.popleft()  # getting the first path from the queue
            current = path[-1] # getting the last cell

            if current == center: 
                return path  # return the entire path if we found the button

            if current in visited:  # if we already visited this node in another path, skip it
                continue 
            
            visited.add(current)  # mark the node as visited
            neighbours = [cell for cell in self.ship.get_open_neighbors(current) if cell not in visited]
            for neighbour in neighbours:  
                new_path = list(path)  # create a new path extending the current one
                new_path.append(neighbour)  # add the neighbor to the new path
                queue.append(new_path)  # enqueue the new path

        return None  # return None if there is no path to the button
    
    
    def get_strategic_points(self):
        strategic_points = []
        ship_size = self.ship.get_length()  # Assuming this method returns the size of the ship
        edge = self.k

        # Define the boundary coordinates considering 'k' blocks away from the actual boundary
        for x in range(edge, ship_size - edge):
            strategic_points.append((x, edge))  # Top boundary
            strategic_points.append((x, ship_size - edge - 1))  # Bottom boundary
        for y in range(edge, ship_size - edge):
            strategic_points.append((edge, y))  # Left boundary
            strategic_points.append((ship_size - edge - 1, y))  # Right boundary
        
        
            
        pass
    
    
    
    def strategy_two(self):
        detection_threshold = 8
        game_is_on = True
        sensed = self.sense()
        # If the detection range is large, start from the center
        total_moves = 0
        while game_is_on:
            if not sensed:
                if self.k > detection_threshold:
                    path_to_center = self.move_center()
                    while path_to_center:
                        self.bot.move(path_to_center.pop(0))
                        total_moves += 1
                    sensed = self.sense()
                    total_moves += 1

                # Then move to other strategic points, e.g., midpoints of ship quadrants
                for point in self.get_strategic_points():
                    self.bot.move(point)
                    self.sense_and_update()
                    # Check if the leak is found and update strategy accordingly

            # If the detection range is small, traverse the boundary
            else:
                self.traverse_boundary()
            pass

    def boundary_sensing_strategy(self):
        # Prioritize boundary cells within the detection square
        boundary_cells = self.get_boundary_cells(self.bot.position)
        for cell in boundary_cells:
            self.bot.move_to(cell)
            self.sense_and_update()
            # If a leak is detected within a certain boundary, search that area

# Assuming sense_and_update and other helper methods exist

    
    
    
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



        
            
