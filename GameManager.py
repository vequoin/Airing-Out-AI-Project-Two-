import random
from Ship import Ship
from bot import Bot

class GameManager:
    def __init__(self, ship_size,bot_strategy):
        """
        Initialize the GameManager class which manages the game's overall state, 
        including the bot, leaks, ship layout, and game elements' positions.

        Args:
            ship_size (int): The size (dimension) of the ship.
            bot_strategy (function): The strategy the bot uses to navigate.
        """
        self.ship = Ship(ship_size)  # Create a ship object with the given size.
        
        self.ship_length = ship_size  # Store the ship size.
        
        # Initialize positions for the bot, fire, and button.
        self.bot_position = random.choice(self.ship.open_cells)
        
        # Create leaks on the ship randomly generated based on bot strategy 
        self.leaks = []

         # Store the selected bot strategy.
        self.bot_strategy = bot_strategy  

        # Create a Bot instance initialized with the ship, its strategy,
        self.bot = Bot(self.ship, bot_strategy, self.bot_position)

        # Initialize a set to keep track of nodes (positions) that have been visited during pathfinding
        self.visited_nodes = set()
        
        
    def initialize_leaks(self):
        open_cells = self.ship.open_cells.copy()
   
        open_cells.discard(self.bot_position)  # Ensure bot's position isn't an option for a leak

        num_leaks = 2 if self.bot_strategy in range(5, 9) else 1
        for _ in range(num_leaks):
            leak_position = random.choice(list(open_cells))  # Convert set to list for random.choice
            self.leaks.append(leak_position)
            open_cells.discard(leak_position)  # Using discard instead of remove to avoid KeyErrors
        

############################################### Stategy 1 #####################################################

    def stretegy_one():
        pass
