import random
import math
from collections import deque

class Strategy3Bot:
    def __init__(self,ship, initial_position):
        self.ship = ship
        self.position = initial_position  # This is now a tuple
        # Setting the initial position's probability to 0
        
    
    def move(self,move):
        self.position = move

    

