import random
from collections import deque
from Ship import Ship
from prob_strategies import Strategy3Bot

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
        
        self.ship_length = ship_size  # Store the ship size.
        
        # Initialize positions for the bot, fire, and button.
        self.bot_position = random.choice(self.ship.open_cells)
        
        self.bot = Strategy3Bot(self.ship, ship_size, alpha,self.bot_position)
        
        # Create leaks on the ship randomly generated based on bot strategy 
        self.leaks = []
        
        self.covered_grid = []

         # Store the selected bot strategy.
        self.bot_strategy = bot_strategy  
        
        self.isleak = self.initialize_leaks()
        self.no_leaks = []
        
    
    def selectBot(self):
        if self.bot_strategy == 3:
            return Strategy3Bot(self.ship, self.ship_size, self.alpha,self.bot_position)
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
            if not leak_position:
                return False
            self.leaks.append(leak_position)
        return True

        

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
                print(neighbor)
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)

        return []  # Return an empty list if no path is found
        
        
    def playy_game(self):
        curr_path = []
        actions = 0
        # Assuming that your bot's position and the leaks are all tuples now.
        while True:
            print(self)
            beep_probability = self.bot.calculate_probability(self.leaks[0])
            beep_heard = random.random() < beep_probability 
            if beep_heard:
                self.bot.update_probabilities()
            next_move = self.bot.choose_next_move()
            print(f"leaks are {self.leaks}")
            print(f"bot is at: {self.bot.position}")
            print(f"next_move is: {next_move}")# Get the next move from your Strategy3Bot
            curr_path = self.find_path_to_edge(self.bot.position, next_move)
            print(f"Curr_path is: {curr_path}")
            n = input("Enter ...")
            while curr_path:
                print(f"bot position before move : {self.bot.position}")
                move = curr_path.pop(0)
                self.bot.update_position(move)
                print(f"bot position after move : {self.bot.position}")
                actions += 1   # Update the bot's position

            if self.leaks[0] == self.bot.position:
                print(f"Leak found at {self.bot.position} in {actions} actions.")
                break

            if actions > 100:  # Some maximum number of actions to prevent an infinite loop
                print("Leak not found within 100 actions.")
                break

    def strategy_seven():
        pass
    
    
    def strategy_eight():
        pass
    
    def run_game(self):
        if self.bot_strategy == 3:
            self.playy_game()
        if self.bot_strategy == 7:
            return self.strategy_seven()
        if self.bot_strategy == 8:
            return self.strategy_eight()
        
        
    '''def __str__(self):
        grid_representation = ""
        for y in range(self.ship_length):
            for x in range(self.ship_length):
                if self.ship.ship[y][x] == 1:
                    grid_representation += "# "  # 1 for Wall
                elif (x, y) == self.bot.position:
                    grid_representation += "B "  # B for Bot
                elif (x, y) in self.leaks:
                    grid_representation += "L "  # L for Leak
                else:
                    grid_representation += ". "  # . for open space
            grid_representation += "\n"
        #grid_representation += "\nProbability Grid:\n"
        #grid_representation += self.get_probability_grid_str()
        return grid_representation'''
    
    
    def __str__(self):
        grid_representation = ""
        for y in range(self.ship_length):
            for x in range(self.ship_length):
                if (x, y) == self.bot.position:
                    grid_representation += "B "  # B for Bot
                elif self.ship.ship[y][x] == 1:
                    grid_representation += "1 "  # 1 for Wall
                elif (x, y) in self.leaks:
                    grid_representation += "L "  # L for Leak
                else:
                    grid_representation += ". "  # . for open space
            grid_representation += "\n"

        return grid_representation



    def get_probability_grid_str(self):
        prob_str = ""
        for row in self.bot.probability_grid:
            prob_str += ' '.join('{:.2f}'.format(cell) for cell in row) + "\n"
        return prob_str
        
