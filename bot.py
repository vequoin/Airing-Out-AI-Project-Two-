import random
from collections import deque
from Ship import Ship

class Bot:
    def __init__(self, ship, bot_number,position, button_position):
        # Initializes a Bot with its attributes.
        self.ship = ship  # Assigns the ship on which the bot operates.
        self.position = position  # Sets the bot's current position.
        self.strategy = bot_number  # Sets the bot's strategy identifier.
        self.is_alive = True  # Indicates whether the bot is alive or not.
    
    def move(self, position):
        # Moves the bot to a new position on the ship.
        self.position = position
        
   
    def get_possible_moves(self):
        # Determines the possible moves for the bot.
        open_neighbors = self.ship.get_open_neighbors()
        
    

