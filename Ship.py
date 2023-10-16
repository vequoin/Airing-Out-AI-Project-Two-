import random


class Ship:
    def __init__(self, D):
        """Initializes the Ship with dimension D x D and generates its structure."""
        self.name = "Archaeopteryx"  # Assigns a name to the ship.
        self.D = D  # Sets the dimension of the ship.
        self.ship = self.generate_ship()  # Generates the ship's structure.
        self.fire_instance = None  # Initializes a variable for tracking fire on the ship.
        self.open_cells = self.get_open_cells()  # Gets a list of open cells on the ship.

    def generate_ship(self):
        """Generates the ship's structure with rooms and pathways."""
        # Create a D x D grid filled with 1's (walls/rooms)
        ship = [[1 for _ in range(self.D)] for _ in range(self.D)]

        start_cell = (random.randint(0, self.D - 1), random.randint(0, self.D - 1))
        ship[start_cell[0]][start_cell[1]] = 0

        fringe_cells = set(self.get_neighbors(start_cell))

        while fringe_cells:
            #filter Cells which are not valid, valids cells have one open neighbor
            valid_cells = [cell for cell in fringe_cells if ship[cell[0]][cell[1]] and sum([1 - ship[nx][ny] for nx, ny in self.get_neighbors(cell)]) == 1]

            if not valid_cells:
                break

            chosen_cell = random.choice(valid_cells)
            ship[chosen_cell[0]][chosen_cell[1]] = 0
            

            fringe_cells.remove(chosen_cell)
            fringe_cells.update(self.get_neighbors(chosen_cell))
            
        self.eliminate_dead_ends(ship)
        return ship
    

    def eliminate_dead_ends(self, ship):
        """Eliminates dead ends from the ship by making some of them open."""
        # Find all cells that are dead-ends
        dead_ends = [(row, col) for row in range(self.D) for col in range(self.D) if
                     ship[row][col] == 0 and sum([1 - ship[nx][ny] for nx, ny in self.get_neighbors((row, col))]) == 1]
        
        required_length = len(dead_ends)//2

        while required_length < len(dead_ends):
            random_dead_end = random.choice(dead_ends)
            neighbours = self.get_neighbors(random_dead_end)
            closed_neighbors = [(xi,yi) for xi,yi in neighbours if ship[xi][yi] == 1]
            
            if not closed_neighbors:
                return
            cell_to_open = random.choice(closed_neighbors)
            ship[cell_to_open[0]][cell_to_open[1]] = 0
            dead_ends = [(row, col) for row in range(self.D) for col in range(self.D) if
                     ship[row][col] == 0 and sum([1 - ship[nx][ny] for nx, ny in self.get_neighbors((row, col))]) == 1]

    def get_open_cells(self):
        """Returns a list of open cells on the ship."""
        return [(i, j) for i, row in enumerate(self.ship) for j, cell in enumerate(row) if cell == 0]

    def get_neighbors(self, cell):
        """Returns the neighboring cells of a given cell."""
        x, y = cell
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(xi, yi) for xi, yi in neighbors if 0 <= xi < self.D and 0 <= yi < self.D]

    def get_open_neighbors(self, cell):
        """Returns neighboring cells that are open."""
        x, y = cell
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_cells = [(xi, yi) for xi, yi in neighbors if 0 <= xi < self.D and 0 <= yi < self.D and self.ship[xi][yi] == 0]
        return valid_cells
        
        
    def get_length(self):
        """Returns the dimension of the ship."""
        return self.D
    
    def __str__(self):
        """Returns a string representation of the ship's structure."""
        grid_str = ""
        for i, row in enumerate(self.ship):
            for j, cell in enumerate(row):
                if self.fire_instance and (i, j) in self.fire_instance.get_cells_on_fire():
                    grid_str += "F "
                elif cell == 0:
                    grid_str += "0 "
                else:
                    grid_str += "1 "
            grid_str += "\n"
        return grid_str

